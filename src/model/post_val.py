import os
import shutil
import sys
import json
import argparse
import numpy as np
import torch
import multiprocessing as mp
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, f1_score, hamming_loss
from sklearn.utils import resample
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# configuration des chemins relatifs
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from evaluation import compute_challenge_metric, load_weights
from model_factory import get_shared_parser, build_model



torch.set_float32_matmul_precision('high')

# variables globales pour le multiprocessing
_worker_probs = None
_worker_labels = None
_worker_weights = None
_worker_classes = None
_worker_sr = None


def _init_worker(probs, labels, weights, classes, sr):
    """
    initialise la mémoire partagée pour chaque processus travailleur.
    """
    global _worker_probs, _worker_labels, _worker_weights, _worker_classes, _worker_sr
    _worker_probs = probs
    _worker_labels = labels
    _worker_weights = weights
    _worker_classes = classes
    _worker_sr = sr


def get_predictions(model, dataloader, device, use_amp, amp_dtype):
    """
    effectue l'inférence sur le jeu de validation
    """
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, y, batch_mask in tqdm(dataloader, desc="inférence"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
                probs = torch.sigmoid(model(x, batch_mask=batch_mask))

            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    return torch.cat(all_labels).to(torch.float32).numpy().astype(bool), torch.cat(all_probs).to(torch.float32).numpy()

def validate_bootstrapping(labels, probs, weights, classes, nsr_index, thresholds, n_iterations=100):
    """
    Évalue la robustesse statistique via Bootstrapping pour obtenir des intervalles de confiance à 95%.
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
    th, class_idx, base_thresholds = args
    test_thresholds = base_thresholds.copy()
    test_thresholds[class_idx] = th

    binary_preds = (_worker_probs >= test_thresholds).astype(bool)

    challenge = compute_challenge_metric(_worker_weights, _worker_labels, binary_preds, _worker_classes, _worker_sr)
    f1 = f1_score(_worker_labels, binary_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(_worker_labels.flatten(), binary_preds.flatten())

    return challenge, f1, mcc, th

def optimize_all_metrics(val_labels, val_probs, weights, classes, nsr_index, num_workers, epochs=3):
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
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(args.workers)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)

    # Configuration matérielle et amp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(f"cuda:{args.gpu}")
        use_amp = not args.not_use_amp

        if use_amp and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("[init] matériel compatible ampere détecté : bfloat16 activé.")
        else:
            amp_dtype = torch.float16
            print("[init] bfloat16 non supporté : fallback sur float16.")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        device = torch.device("cpu")
        use_amp = False
        amp_dtype = torch.float32
        print("[init] mode cpu forcé.")

    with open(args.class_map) as f:
        classes = json.load(f)
    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes
    nsr_index = classes.index("NSR")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = "offline" 
    os.environ["WANDB_DIR"] = os.path.join(args.output, "wandb_logs")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    wandb_id = wandb.util.generate_id()
    wandb.init(
        project="ecg_classification_experiments",
        job_type="post_val",
        name=f"validation_{args.model_name}_{wandb_id[:6]}",
        config=vars(args),
        id=wandb_id,
        tags=["validation", args.model_name, "offline"]
    )

    model, _, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()})

    mb_size = args.batch_size_theoric * args.mega_batch_factor
    dataset_kwargs = {
        "batch_size": args.batch_size_accumulat,
        "mega_batch_size": mb_size,
        "use_static_padding": args.use_static_padding
    }

    if gen_fun is not None:
        dataset_kwargs["generate_img"] = gen_fun

    val_ds = Dataset_fun(
        data_path=args.data,
        **dataset_kwargs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2
    )

    # 1. inférence avec amp
    labels, probs = get_predictions(model, val_loader, device, use_amp, amp_dtype)
    
    # 2. Remplacement par Sklearn (beaucoup plus rapide et safe)
    try:
        auroc_macro = roc_auc_score(labels, probs, average='macro')
        auprc_macro = average_precision_score(labels, probs, average='macro')
    except ValueError:
        auroc_macro, auprc_macro = float('nan'), float('nan')
    wandb.log({"validation_brute/auroc_macro": auroc_macro, "validation_brute/auprc_macro": auprc_macro})

    # 3. Optimisation ciblée des seuils
    best_configs = optimize_all_metrics(labels, probs, weights, classes, nsr_index, epochs=3, num_workers=args.workers)

    # 4. Bootstrapping avec les seuils optimaux (vrais intervalles de confiance)
    best_challenge_th = best_configs["challenge_score"]["thresholds"]
    robustness_metrics, raw_bootstrap_stats = validate_bootstrapping(labels, probs, weights, classes, nsr_index, best_challenge_th)
    wandb.log(robustness_metrics)

    # 5. Extraction binaire avec le meilleur modèle pour le challenge
    binary_results, real_sick, pred_sick = compute_binary_metrics(labels, probs, best_challenge_th, nsr_index)
    wandb.log(binary_results)

    
    # A. Histograms du Bootstrapping
    wandb.log({
        "distribution_bootstrap/challenge": wandb.Histogram(raw_bootstrap_stats['challenge']),
        "distribution_bootstrap/macro_f1": wandb.Histogram(raw_bootstrap_stats['macro_f1'])
    })

    # B. F1-Score par Classe (Bar Chart interactif sur Wandb)
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

    # préparation du fichier json
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
    parser.add_argument('--data', default="../../../output/final_data/val")
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    args = parser.parse_args()

    args.workers = max(4, mp.cpu_count()-1)

    run(args)


if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()
