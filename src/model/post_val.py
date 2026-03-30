import os
import shutil
import sys
import json
import argparse
import numpy as np
import torch
import multiprocessing as mp

from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from TurboDataset import TurboDataset
from model_factory import get_shared_parser, build_model



torch.set_float32_matmul_precision('high')

# Variables globales pour le multiprocessing
_worker_probs = None    # Stockera la matrice des prédictions (val_probs)
_worker_labels = None   # Stockera la matrice des vraies étiquettes (val_labels)
_worker_weights = None  # Stockera la matrice des coûts du challenge PhysioNet
_worker_classes = None  # Stockera la liste des noms de classes
_worker_sr = None       # Stockera le nom de la classe de référence (rythme sinusal)


def _init_worker(probs, labels, weights, classes, sr):
    """
    Initialise la mémoire de chaque worker une seule fois au lancement du pool.
    Cela évite de transférer de larges matrices à chaque calcul.

    Args:
        probs (np.ndarray): Matrice des probabilités prédites.
        labels (np.ndarray): Matrice des labels réels.
        weights (np.ndarray): Matrice des poids de la métrique.
        classes (list): Liste des noms de classes.
        sr (str): Nom de la classe de rythme sinusal normal.
    """
    global _worker_probs, _worker_labels, _worker_weights, _worker_classes, _worker_sr
    _worker_probs = probs
    _worker_labels = labels
    _worker_weights = weights
    _worker_classes = classes
    _worker_sr = sr

def _eval_batch(args):
    """
    Fonction map exécutée par imap_unordered pour évaluer un seuil spécifique.

    Args:
        args (tuple): Contient le seuil à tester, l'index de la classe, 
                      et le tableau des seuils de base.

    Returns:
        tuple: (score obtenu, seuil testé)
    """
    th, class_idx, base_thresholds = args
    
    test_thresholds = base_thresholds.copy()
    test_thresholds[class_idx] = th

    # Utilisation des variables globales du worker pour l'inférence
    binary_preds = (_worker_probs >= test_thresholds).astype(bool)
    score = compute_challenge_metric(
        _worker_weights, _worker_labels, binary_preds, _worker_classes, _worker_sr
    )
    return score, th



def compute_accuracy(labels, outputs):
    """Calcule l'accuracy globale sur l'ensemble des enregistrements."""
    num_recordings = labels.shape[0]
    correct = np.all(labels == outputs, axis=1)
    return float(np.sum(correct)) / float(num_recordings)

def compute_confusion_matrices(labels, outputs, normalize=False):
    """Calcule les matrices de confusion binaires pour chaque classe."""
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, 2, 2))

    if not normalize:
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1: A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1: A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0: A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0: A[j, 0, 0] += 1
                else: raise ValueError('Erreur lors du calcul de la matrice de confusion.')
    else:
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1: A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1: A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0: A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0: A[j, 0, 0] += 1.0 / normalization
                else: raise ValueError('Erreur lors du calcul de la matrice de confusion.')
    return A

def compute_f_measure(labels, outputs):
    """Calcule la F-mesure macro et par classe."""
    num_recordings, num_classes = np.shape(labels)
    A = compute_confusion_matrices(labels, outputs)
    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn: 
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else: 
            f_measure[k] = float('nan')
    macro_f_measure = np.nanmean(f_measure) if np.any(np.isfinite(f_measure)) else float('nan')
    return macro_f_measure, f_measure

def compute_auc(labels, outputs):
    """Calcule l'AUROC et l'AUPRC macro et par classe."""
    num_recordings, num_classes = labels.shape
    auroc, auprc = np.zeros(num_classes), np.zeros(num_classes)
    for k in range(num_classes):
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)[::-1]
        tp, fp, fn, tn = np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds))
        fn[0], tn[0] = np.sum(labels[:, k] == 1), np.sum(labels[:, k] == 0)
        idx = np.argsort(outputs[:, k])[::-1]
        i = 0
        for j in range(1, len(thresholds)):
            tp[j], fp[j], fn[j], tn[j] = tp[j-1], fp[j-1], fn[j-1], tn[j-1]
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]: tp[j] += 1; fn[j] -= 1
                else: fp[j] += 1; tn[j] -= 1
                i += 1
        tpr = tp / np.maximum(tp + fn, 1)
        tnr = tn / np.maximum(fp + tn, 1)
        ppv = tp / np.maximum(tp + fp, 1)
        for j in range(len(thresholds) - 1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]
    return np.nanmean(auroc), np.nanmean(auprc), auroc, auprc

def compute_modified_confusion_matrix(labels, outputs):
    """Calcule la matrice de confusion modifiée pour la métrique du challenge."""
    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        union_count = np.sum(labels[i] | outputs[i])
        if union_count > 0:
            A += np.outer(labels[i], outputs[i]) / float(union_count)
    return A

def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    """Calcule le score normalisé du challenge PhysioNet."""
    num_recordings, num_classes = labels.shape
    if sinus_rhythm in classes: 
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else: 
        raise ValueError('La classe de rythme sinusal n\'est pas disponible.')

    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else: 
        normalized_score = 0.0

    return normalized_score


def is_number(x):
    """Vérifie si la valeur est un nombre."""
    try: 
        float(x)
        return True
    except (ValueError, TypeError): 
        return False

def is_finite_number(x):
    """Vérifie si la valeur est un nombre fini."""
    return np.isfinite(float(x)) if is_number(x) else False

def load_table(table_file):
    """Charge un tableau CSV avec noms de lignes et colonnes."""
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            table.append([arr.strip() for arr in l.split(',')])
    num_rows = len(table)-1
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    num_cols = min(row_lengths)
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            values[i, j] = float(value) if is_finite_number(value) else float('nan')
    return rows, cols, values

def load_weights(weights_file):
    """Charge la matrice des poids d'évaluation."""
    rows, cols, values = load_table(weights_file)
    assert(rows == cols)
    return rows, values



def get_predictions(model, dataloader, device):
    """
    Effectue une passe unique sur les données pour extraire les probabilités brutes.
    
    Args:
        model (torch.nn.Module): Le modèle PyTorch chargé.
        dataloader (DataLoader): Le chargeur de données de validation.
        device (torch.device): Le périphérique cible (CPU/GPU).
        
    Returns:
        tuple: (labels sous forme de booléens, probabilités brutes)
    """
    all_labels, all_probs = [], []
    model.eval()

    with torch.no_grad():
        for x, y, batch_mask in tqdm(dataloader, desc="inférence sur le set de validation"):
            x = x.to(device)
            y = y.to(device)
            batch_mask = batch_mask.to(device)

            if x.shape[-1] == 0:
                print(f"\n[Avertissement] Donnée invalide détectée et ignorée : shape={x.shape}.")
                continue 

            probs = torch.sigmoid(model(x, batch_mask=batch_mask))

            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    labels = torch.cat(all_labels).numpy().astype(bool)
    probs = torch.cat(all_probs).numpy()

    return labels, probs


def optimize_coordinate_descent(val_labels, val_probs, weights, classes, sinus_rhythm="NSR", epochs=3, num_workers=4):
    """
    Trouve les 27 seuils optimaux en maximisant le Challenge Score.
    Utilise la descente de coordonnées parallélisée avec mémoire partagé.
    
    Args:
        val_labels (np.ndarray): Les labels réels.
        val_probs (np.ndarray): Les probabilités prédites.
        weights (np.ndarray): La matrice des poids.
        classes (list): Liste des classes.
        sinus_rhythm (str, optional): Nom de la classe saine.
        epochs (int, optional): Nombre d'itérations complètes.
        num_workers (int, optional): Nombre de processus parallèles.
        
    Returns:
        np.ndarray: Vecteur des seuils optimisés.
    """
    print(f"\n[Optimisation] Lancement sur {epochs} epochs avec {num_workers} workers...")

    # Initialisation des seuils à 0.5 pour chaque classe
    best_thresholds = np.ones(27) * 0.5 
    candidate_thresholds = np.arange(0.10, 0.75, 0.05) 
    best_global_score = -np.inf
    step = 0
    total_steps = epochs * 27

    # Préparation des arguments partagés pour le pool
    pool_args = (val_probs, val_labels, weights, classes, sinus_rhythm)
    
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=pool_args) as pool:
        with tqdm(total=total_steps, desc="recherche des seuils", unit="cls") as pbar:
            for epoch in range(epochs):
                for i in range(27):      
                    # Préparation des tâches contenant uniquement les données légères
                    tasks = [(th, i, best_thresholds) for th in candidate_thresholds]

                    best_th_for_class_i = best_thresholds[i]
                    best_score_for_class_i = -np.inf

                    # Récupération asynchrone des résultats et identification du meilleur seuil
                    for score, th in pool.imap_unordered(_eval_batch, tasks, chunksize=1):
                        if score > best_score_for_class_i:
                            best_score_for_class_i = score
                            best_th_for_class_i = th

                    # Mise à jour une fois que tout le batch de seuils est évalué
                    best_thresholds[i] = best_th_for_class_i
                    best_global_score = best_score_for_class_i

                    # Traçabilité WandB
                    wandb.log({
                        "optim/step": step,
                        "optim/epoch": epoch + 1,
                        "optim/current_class": classes[i],
                        "optim/best_challenge_score": best_global_score
                    })
                    step += 1

                    pbar.set_postfix({
                        "epoch": f"{epoch+1}/{epochs}", 
                        "meilleur_score": f"{best_global_score:.4f}"
                    })
                    pbar.update(1)

    print(f"\n[Succès] Score maximum atteint lors de l'optimisation : {best_global_score:.4f}")
    return best_thresholds


def run(args):
    """
    Exécute le pipeline complet : inférence, optimisation et sauvegarde de configuration.
    """
    mp.set_start_method("spawn", force=True)     
    torch.set_num_threads(args.workers)         

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement des classes et des poids
    with open(args.class_map) as f:
        classes = json.load(f)
    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes, "L'ordre des classes ne correspond pas à weights.csv"

    # Configuration WandB
    wandb_id = wandb.util.generate_id()
    os.makedirs(args.output, exist_ok=True)
    base_wandb_path = os.path.join(args.output, "wandb_logs")
    os.makedirs(base_wandb_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = base_wandb_path
    os.environ["WANDB_CACHE_DIR"] = os.path.join(base_wandb_path, "cache")
    os.environ["WANDB_DATA_DIR"] = base_wandb_path 
    os.environ["TMPDIR"] = os.path.join(base_wandb_path, "tmp")
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    base_name = args.checkpoint.split('_ep')[0] 
    group_id = base_name.split('-')[-1]

    wandb.init(
        project="ECG_Classification_Experiments",
        group=group_id,
        job_type="eval_optimization",
        name=f"opt_{args.model_name}_{wandb_id[:6]}",
        id=wandb_id,
        config={
            "checkpoint_source": args.checkpoint,
            "test_batch_size": args.batch_size_theoric,
            "model_tested": args.model_name,
            "optimization_epochs": 3,
            "workers": args.workers
        },
        tags=["optimization", args.model_name, "offline"]
    )

    print(f"Début de l'évaluation et optimisation pour le modèle : {args.checkpoint}")

    # Préparation du dataset
    mb_size = args.batch_size_theoric * args.mega_batch_factor
    val_ds = TurboDataset(
        data_path=args.data,
        batch_size=args.batch_size_accumulat,
        mega_batch_size=mb_size,
        use_static_padding=args.use_static_padding
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0), 
        prefetch_factor=2
    )

    # Chargement du modèle
    model, _ = build_model(args)
    model = model.to(device)
    full_checkpoint_path = os.path.abspath(os.path.join(args.checkpoint_dir, args.checkpoint))
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=True)

    # Étape 1 : Inférence globale
    labels, probs = get_predictions(model, val_loader, device)

    # Étape 2 : Optimisation
    optimal_thresholds = optimize_coordinate_descent(
        labels, probs, weights, classes, sinus_rhythm="NSR", epochs=3, num_workers=args.workers
    )

    # Étape 3 : Calcul des métriques finales
    print('\n[Évaluation] Calcul des métriques avec les seuils optimaux...')
    binary = (probs >= optimal_thresholds).astype(bool)

    auroc, auprc, auroc_c, auprc_c = compute_auc(labels, probs)
    acc = compute_accuracy(labels, binary)
    f1, f1_c = compute_f_measure(labels, binary)

    A = compute_confusion_matrices(labels, binary)
    sensitivity = np.zeros(len(classes))
    specificity = np.zeros(len(classes))
    for k in range(len(classes)):
        tp, fp, fn, tn = A[k,1,1], A[k,1,0], A[k,0,1], A[k,0,0]
        sensitivity[k] = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity[k] = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    final_challenge_score = compute_challenge_metric(weights, labels, binary, classes, "NSR")
    
    print(f"score final obtenu (challenge metric) : {final_challenge_score:.4f}")

    # Étape 4 : Sauvegarde de la configuration
    print('\n[Sauvegarde] Génération du fichier de configuration JSON...')

    config_dict = {
        "model_name": args.model_name,
        "checkpoint_path": full_checkpoint_path,
        "final_challenge_score": float(final_challenge_score),
        "optimal_thresholds_vector": optimal_thresholds.tolist(),
        "thresholds_per_class": {cls: float(th) for cls, th in zip(classes, optimal_thresholds)}
    }

    config_filename = f"{args.model_name}_{group_id}_opt_config.json"
    config_filepath = os.path.join(args.output, config_filename)

    with open(config_filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)

    print(f"configuration enregistrée dans : {config_filepath}")

    # Étape 5 : Journalisation WandB
    wandb.log({
        "eval/AUROC":           auroc,
        "eval/AUPRC":           auprc,
        "eval/Accuracy":        acc,
        "eval/Macro_F1":        f1,
        "eval/Final_Challenge_Score": final_challenge_score,
    })

    wandb.run.summary["Final_Challenge_Score"] = final_challenge_score

    class_table = wandb.Table(columns=["Class", "Optimal_Threshold", "F1", "AUROC", "AUPRC", "Sensitivity", "Specificity"])
    for i, cls in enumerate(classes):
        class_table.add_data(
            cls, 
            float(optimal_thresholds[i]),
            float(f1_c[i]) if np.isfinite(f1_c[i]) else None,
            float(auroc_c[i]) if np.isfinite(auroc_c[i]) else None,
            float(auprc_c[i]) if np.isfinite(auprc_c[i]) else None,
            float(sensitivity[i]) if np.isfinite(sensitivity[i]) else None,
            float(specificity[i]) if np.isfinite(specificity[i]) else None
        )
    wandb.log({"eval/per_class_metrics": class_table})

    artifact = wandb.Artifact(f"model-config-{group_id}", type="thresholds-config")
    config_interne = os.path.join(wandb.run.dir, config_filename)
    shutil.copy2(config_filepath, config_interne)

    artifact.add_file(config_interne, name=config_filename)
    wandb.log_artifact(artifact)

    wandb.finish()
    print("[Terminé] L'ensemble du processus a été exécuté et journalisé.")


def main():
    """Point d'entrée du script."""
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="Script d'optimisation des seuils", parents=[shared_parser])

    parser.add_argument('--data', default="../../../output/final_data/val", help="Répertoire HDF5 contenant les données de validation")
    parser.add_argument('-c', '--checkpoint', required=True, help="Point de contrôle (checkpoint) du modèle entraîné")
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="Fichier weights.csv (PhysioNet)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()
