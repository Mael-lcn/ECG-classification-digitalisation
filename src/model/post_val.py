import os
import sys
import json
import argparse
import numpy as np
import torch
import multiprocessing as mp
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, f1_score, hamming_loss
from sklearn.utils import resample
from tqdm import tqdm
import wandb

# configuration des chemins relatifs
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from evaluation import compute_challenge_metric, load_weights
from model_factory import get_shared_parser, build_model, create_dataloader
from core_utils import (
    setup_global_environment,
    setup_wandb,
    run_inference,
    load_checkpoint
)


# variables globales pour le multiprocessing
_worker_probs = None
_worker_labels = None
_worker_weights = None
_worker_classes = None
_worker_sr = None


def _init_worker(probs, labels, weights, classes, sr):
    """Initialise la mémoire partagée pour chaque processus travailleur."""
    global _worker_probs, _worker_labels, _worker_weights, _worker_classes, _worker_sr
    _worker_probs = probs
    _worker_labels = labels
    _worker_weights = weights
    _worker_classes = classes
    _worker_sr = sr


def validate_bootstrapping(labels, probs, weights, classes, nsr_index, thresholds, n_iterations=100):
    """
    Évalue la robustesse statistique des métriques d'évaluation via la méthode de bootstrapping. 
    
    Calcule les intervalles de confiance à 95 % pour le score du challenge, le F1-score macro et la perte de Hamming,
    en effectuant des tirages aléatoires avec remise sur les données de validation.

    Args:
        labels (numpy.ndarray): Matrice des étiquettes réelles.
        probs (numpy.ndarray): Matrice des probabilités prédites.
        weights (numpy.ndarray): Matrice des pondérations pour le score du challenge.
        classes (list): Liste des désignations des classes.
        nsr_index (int): Index de la classe correspondant au rythme sinusal normal.
        thresholds (numpy.ndarray): Vecteur des seuils de décision optimisés par classe.
        n_iterations (int): Nombre d'itérations pour le tirage de bootstrapping.

    Returns:
        tuple: Un tuple contenant :
            - results (dict): Les métriques statistiques agrégées (moyenne et intervalles de confiance pour chaque métrique).
            - stats (dict): Les listes des scores bruts calculés pour chaque itération de la boucle.
    """
    n_samples = len(labels)
    nsr_name = classes[nsr_index]

    stats = {
        'challenge': [],
        'macro_f1': [],
        'hamming_loss': []
    }

    print(f"\n[analyse] Lancement du bootstrapping statistique ({n_iterations} itérations)...")
    for _ in tqdm(range(n_iterations), desc="Bootstrapping", leave=False):
        # Tirage avec remise
        indices = resample(np.arange(n_samples), replace=True)
        sample_labels = labels[indices]
        sample_probs = probs[indices]

        sample_preds = (sample_probs >= thresholds).astype(bool)

        stats['challenge'].append(compute_challenge_metric(weights, sample_labels, sample_preds, classes, nsr_name))
        stats['macro_f1'].append(f1_score(sample_labels, sample_preds, average='macro', zero_division=0))
        stats['hamming_loss'].append(hamming_loss(sample_labels, sample_preds))

    results = {}
    for metric, values in stats.items():
        results[f"stabilité/{metric}_mean"] = np.mean(values)
        results[f"stabilité/{metric}_ci_lower"] = np.percentile(values, 2.5)
        results[f"stabilité/{metric}_ci_upper"] = np.percentile(values, 97.5)

    return results, stats


def compute_binary_metrics(labels, probs, thresholds, nsr_index):
    """
    Calcule les métriques de classification binaire pour la sous-tâche distinguant les patients sains des patients malades.
    
    Un patient est considéré comme malade s'il présente au moins une pathologie autre que le rythme sinusal normal (NSR).

    Args:
        labels (numpy.ndarray): Matrice binaire des étiquettes réelles.
        probs (numpy.ndarray): Matrice des probabilités prédites.
        thresholds (numpy.ndarray): Tableau des seuils de décision par classe.
        nsr_index (int): Index de la classe représentant l'état sain.

    Returns:
        tuple: Un tuple contenant :
            - metrics (dict): Dictionnaire comprenant la sensibilité, spécificité, F1-score, coefficient de corrélation de Matthews (MCC) et la prévalence de la maladie.
            - real_sick (numpy.ndarray): Vecteur booléen indiquant la présence réelle d'une pathologie.
            - pred_sick (numpy.ndarray): Vecteur booléen indiquant la prédiction d'une pathologie par le modèle.
    """
    num_classes = labels.shape[1]
    other_indices = [i for i in range(num_classes) if i != nsr_index]

    real_sick = np.any(labels[:, other_indices] == 1, axis=1)
    preds = (probs >= thresholds).astype(bool)
    pred_sick = np.any(preds[:, other_indices], axis=1)

    tp = np.sum((real_sick == 1) & (pred_sick == 1))
    tn = np.sum((real_sick == 0) & (pred_sick == 0))
    fp = np.sum((real_sick == 0) & (pred_sick == 1))
    fn = np.sum((real_sick == 1) & (pred_sick == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_binary = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    mcc = matthews_corrcoef(real_sick, pred_sick)

    return {
        "binaire_malade/sensibilité": sensitivity,
        "binaire_malade/spécificité": specificity,
        "binaire_malade/f1_score": f1_binary,
        "binaire_malade/mcc": mcc,
        "binaire_malade/prevalence": np.mean(real_sick)
    }, real_sick, pred_sick


def _val_batch(args):
    """
    Évalue les performances d'une configuration de seuils spécifique sur un sous-ensemble de données. 
    
    Fonction conçue pour être exécutée en parallèle par les processus travailleurs lors de la recherche par quadrillage.

    Args:
        args (tuple): Un tuple contenant :
            - th (float): Le seuil testé pour la classe cible.
            - class_idx (int): L'index de la classe en cours d'optimisation.
            - base_thresholds (numpy.ndarray): Le vecteur des seuils de base pour les autres classes.

    Returns:
        tuple: Un tuple contenant le score du challenge, le F1-score macro, le score MCC, et la valeur du seuil testé (th).
    """
    th, class_idx, base_thresholds = args
    test_thresholds = base_thresholds.copy()
    test_thresholds[class_idx] = th

    binary_preds = (_worker_probs >= test_thresholds).astype(bool)

    challenge = compute_challenge_metric(_worker_weights, _worker_labels, binary_preds, _worker_classes, _worker_sr)
    f1 = f1_score(_worker_labels, binary_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(_worker_labels.flatten(), binary_preds.flatten())

    return challenge, f1, mcc, th


def optimize_all_metrics(val_labels, val_probs, weights, classes, nsr_index, num_workers, epochs=3):
    """
    Effectue une recherche multivariée itérative pour déterminer les vecteurs de seuils optimisant de manière indépendante plusieurs métriques cibles.
    
    Utilise le multiprocessing pour accélérer l'exploration de l'espace des paramètres. À chaque époque, la fonction itère sur chaque classe pour affiner le seuil de décision tout en maintenant les autres fixes.

    Args:
        val_labels (numpy.ndarray): Matrice des étiquettes réelles de l'ensemble de validation.
        val_probs (numpy.ndarray): Matrice des probabilités issues de l'inférence du modèle.
        weights (numpy.ndarray): Matrice des poids pour le calcul de la métrique du challenge.
        classes (list): Liste des noms des pathologies.
        nsr_index (int): Index associé au rythme sinusal normal.
        num_workers (int): Nombre de processus parallèles à allouer pour l'évaluation.
        epochs (int): Nombre de passages complets d'optimisation sur l'ensemble des classes.

    Returns:
        dict: Un dictionnaire associant chaque métrique cible (score du challenge, F1-score macro, MCC) à son meilleur score obtenu et à son vecteur de seuils optimal.
    """
    nsr_name = classes[nsr_index]
    best_thresholds = np.ones(len(classes)) * 0.5
    candidate_thresholds = np.arange(0.05, 0.41, 0.01)

    best_states = {
        "challenge_score": {"score": -np.inf, "thresholds": best_thresholds.copy()},
        "macro_f1": {"score": -np.inf, "thresholds": best_thresholds.copy()},
        "mcc": {"score": -np.inf, "thresholds": best_thresholds.copy()}
    }

    pool_args = (val_probs, val_labels, weights, classes, nsr_name)

    print("\n[optimisation] lancement de la recherche multivariée...")
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=pool_args) as pool:
        for epoch in range(epochs):
            for i in tqdm(range(len(classes)), desc=f"optimisation - époque {epoch+1}/{epochs}"):
                tasks = [(th, i, best_thresholds) for th in candidate_thresholds]

                best_chal_local = -np.inf
                best_th_local = best_thresholds[i]

                for chal, f1, mcc, th in pool.imap_unordered(_val_batch, tasks):
                    current_th_vector = best_thresholds.copy()
                    current_th_vector[i] = th
    
                    if chal > best_chal_local:
                        best_chal_local = chal
                        best_th_local = th

                    if chal > best_states["challenge_score"]["score"]:
                        best_states["challenge_score"].update({"score": chal, "thresholds": current_th_vector.copy()})
                    if f1 > best_states["macro_f1"]["score"]:
                        best_states["macro_f1"].update({"score": f1, "thresholds": current_th_vector.copy()})
                    if mcc > best_states["mcc"]["score"]:
                        best_states["mcc"].update({"score": mcc, "thresholds": current_th_vector.copy()})

                best_thresholds[i] = best_th_local

    return best_states


def run(args):
    """
    Orchestre le processus global de post-validation et d'optimisation des seuils de décision.

    Configure l'environnement, charge le modèle depuis un point de sauvegarde (checkpoint), exécute l'inférence sur le jeu de validation, et lance la recherche multivariée. 
    Évalue ensuite la robustesse des résultats via bootstrapping, extrait les performances de la sous-tâche binaire, génère des graphiques d'analyse, exporte les configurations optimales au format JSON et synchronise l'ensemble des artefacts avec Weights & Biases.

    Args:
        args (argparse.Namespace): Arguments contenant la configuration système, les hyperparamètres et les chemins d'accès aux fichiers.
    """
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(args.workers)

    device, use_amp, amp_dtype = setup_global_environment(args)

    with open(args.class_map) as f:
        classes = json.load(f)
    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes
    nsr_index = classes.index("NSR")

    wandb_id = setup_wandb(
        args=args,
        job_type="post_val",
        run_name=f"validation_{args.model_name}_{wandb.util.generate_id()[:6]}",
        tags=["validation", args.model_name]
    )

    model, _, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    load_checkpoint(checkpoint_path, model, device=device)

    val_loader = create_dataloader(args, args.val_data, Dataset_fun, gen_fun, is_train=False)

    labels, probs, is_valid = run_inference(
        model, val_loader, device, use_amp, amp_dtype, desc="[Inférence Val]"
    )

    if not is_valid:
        print("[WARNING] Valeurs invalides (NaNs) détectées lors de l'inférence. L'optimisation risque d'échouer.")

    try:
        auroc_macro = roc_auc_score(labels, probs, average='macro')
        auprc_macro = average_precision_score(labels, probs, average='macro')
    except ValueError:
        auroc_macro, auprc_macro = float('nan'), float('nan')

    wandb.log({"validation_brute/auroc_macro": auroc_macro, "validation_brute/auprc_macro": auprc_macro})

    # Optimisation ciblée des seuils
    best_configs = optimize_all_metrics(labels, probs, weights, classes, nsr_index, epochs=3, num_workers=args.workers)

    # Bootstrapping avec les seuils optimaux
    best_challenge_th = best_configs["challenge_score"]["thresholds"]
    robustness_metrics, raw_bootstrap_stats = validate_bootstrapping(labels, probs, weights, classes, nsr_index, best_challenge_th)
    wandb.log(robustness_metrics)

    # Extraction binaire
    binary_results, real_sick, pred_sick = compute_binary_metrics(labels, probs, best_challenge_th, nsr_index)
    wandb.log(binary_results)

    # A. Histograms du Bootstrapping
    wandb.log({
        "distribution_bootstrap/challenge": wandb.Histogram(raw_bootstrap_stats['challenge']),
        "distribution_bootstrap/macro_f1": wandb.Histogram(raw_bootstrap_stats['macro_f1'])
    })

    # B. F1-Score par Classe
    best_preds_f1 = (probs >= best_configs["macro_f1"]["thresholds"]).astype(bool)
    f1_per_class = f1_score(labels, best_preds_f1, average=None, zero_division=0)
    f1_table = wandb.Table(
        columns=["Classe", "F1 Score"], 
        data=[[classes[i], f1_per_class[i]] for i in range(len(classes))]
    )
    wandb.log({"analyse/f1_par_classe": wandb.plot.bar(f1_table, "Classe", "F1 Score", title="F1 Score par Pathologie")})

    # C. Matrice de confusion pour la sous-tâche "Sain vs Malade"
    wandb.log({
        "analyse/matrice_confusion_binaire": wandb.plot.confusion_matrix(
            preds=pred_sick.astype(int),
            y_true=real_sick.astype(int),
            class_names=["Sain (NSR)", "Malade"]
        )
    })

    # D. Préparation et Log du fichier JSON
    final_config = {
        "metadata": {
            "model_name": args.model_name,
            "tache_binaire": binary_results
        },
        "configurations_optimales": {}
    }

    for metric, data in best_configs.items():
        wandb.run.summary[f"optimal_{metric}"] = data["score"]
        final_config["configurations_optimales"][metric] = {
            "score": float(data["score"]),
            "seuils": {cls: float(t) for cls, t in zip(classes, data["thresholds"])}
        }

    config_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_config_opti.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=4)

    artifact = wandb.Artifact(name="configuration_seuils", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)
    wandb.finish()


def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="optimisation des seuils multivariée", parents=[shared_parser])
    parser.add_argument('--val_data', default="../../../output/final_data/val")
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    args = parser.parse_args()

    args.workers = max(4, mp.cpu_count()-1)
    run(args)

if __name__ == "__main__":
    main()
