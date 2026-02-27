import os
import sys
import json
import argparse
import numpy as np
import torch
from functools import partial

from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import csv

import wandb

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from Dataset import LargeH5Dataset, ecg_collate_wrapper
from Sampler import MegaBatchSortishSampler
from Cnn import CNN
from Cnn_TimeFreq import CNN_TimeFreq



torch.set_float32_matmul_precision('high')  # Test d'optimisation

# ==============================================================================
# CONFIGURATION SYSTÈME
# ==============================================================================
# Liste des modèles disponibles
model_list = [CNN, CNN_TimeFreq]
# Supprime la limite de recompilation pour éviter les crashs avec torch.compile
torch._dynamo.config.recompile_limit = 6000


# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings = labels.shape[0]
    correct = np.all(labels == outputs, axis=1)
    return float(np.sum(correct)) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A


# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = labels.shape
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]

        tp = np.zeros(len(thresholds))
        fp = np.zeros(len(thresholds))
        fn = np.zeros(len(thresholds))
        tn = np.zeros(len(thresholds))

        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        idx = np.argsort(outputs[:, k])[::-1]
        i = 0

        for j in range(1, len(thresholds)):
            tp[j], fp[j], fn[j], tn[j] = tp[j-1], fp[j-1], fn[j-1], tn[j-1]
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        tpr = tp / np.maximum(tp + fn, 1)
        tnr = tn / np.maximum(fp + tn, 1)
        ppv = tp / np.maximum(tp + fp, 1)

        for j in range(len(thresholds) - 1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    return np.nanmean(auroc), np.nanmean(auprc), auroc, auprc


def compute_modified_confusion_matrix(labels, outputs):
    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))

    for i in range(num_recordings):
        # Calculate union count for this specific recording
        union_count = np.sum(labels[i] | outputs[i])
        if union_count > 0:
            # Use outer product to fill the matrix for this recording
            # (replaces 'for j' and 'for k' loops)
            A += np.outer(labels[i], outputs[i]) / float(union_count)
            
    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = labels.shape
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


def evaluate(model, dataloader, device, threshold):
    model.eval()
    all_labels, all_probs, all_binary = [], [], []

    with torch.no_grad():
        for x, y, _ in tqdm(dataloader, desc="[EVAL]"):
            x = x.to(device)
            y = y.to(device)

            if x.shape[-1] == 0:
                print(f"\n[SKIP] Donnée invalide détectée : shape={x.shape}. Vérifiez le prétraitement.")
                continue # Passe à l'ECG suivant au lieu de faire crash le modèle

            #probs = model(x)
            probs = torch.sigmoid(model(x))
            binary = (probs >= threshold).int()

            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            all_binary.append(binary.cpu())

    labels = torch.cat(all_labels).numpy().astype(bool)
    probs = torch.cat(all_probs).numpy()
    binary = torch.cat(all_binary).numpy().astype(bool)

    return labels, binary, probs

"""
Some utility functions for loading tables and weights
"""
# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

def load_weights(weights_file):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weights_file)

    assert(rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights



def main():
    options = ", ".join([f"{i}: {model.__name__}" for i, model in enumerate(model_list)])

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="../output/final_data/test", help="HDF5 test dataset directory")
    parser.add_argument('-c', '--checkpoint', default="checkpoints/best_model_ep49.pt", help="Trained model checkpoint")
    parser.add_argument('--class_map', default='../../ressources/final_class.json', help="JSON ordered class list")
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    parser.add_argument('-o', '--output', type=str, default="../output/evaluation")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--use_static_padding', action='store_true', default=False,
                        help="Force une taille de padding fixe (universelle).")

    parser.add_argument('--model', type=int, default=0,
                        help=f"Quel modèle voulez-vous évaluer: {options}")

    parser.add_argument("-w", "--workers", type=int, default=min(8, multiprocessing.cpu_count()-1))

    args = parser.parse_args()

    # multiprocessing set up
    multiprocessing.set_start_method("spawn", force=True)     
    torch.set_num_threads(args.workers)         

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    with open(args.class_map) as f:
        class_list = json.load(f)
    classes = class_list

    # Load weights
    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes, "Class order mismatch with weights.csv"


    # ================= CONFIGURATION WANDB ================= 

    os.makedirs(args.output, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.path.join(args.output, "wandb_logs")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    checkpoint_basename = os.path.basename(args.checkpoint)
    # On extrait la partie "EXP_..." avant le numéro d'époque
    group_id = checkpoint_basename.split('_ep')[0].replace('best_model_', '')

    # 2. INITIALISATION WANDB CLEAN
    wandb.init(
        project="ECG_Classification_Experiments",
        group=group_id,
        job_type="eval",
        name=f"test_thr{args.threshold}",
        config={
            "eval_threshold": args.threshold,
            "checkpoint_source": checkpoint_basename,
            "test_batch_size": args.batch_size,
            "use_static_padding": args.use_static_padding
        },
        tags=["eval", "final_test", "offline"]
    )

    # Définition des axes pour des graphiques cohérents
    wandb.define_metric("eval/*")

    print(f"Début de l'évaluation : {checkpoint_basename}")

    # Dataset
    dataset = LargeH5Dataset(args.data, classes_list=class_list,  use_static_padding=False)

    collate_fn = partial(ecg_collate_wrapper, use_static_padding=args.use_static_padding)

    # Création des Samplers et Loaders
    train_sampler = MegaBatchSortishSampler(dataset, batch_size=args.batch_size, mega_batch_factor=50, shuffle=True)

    loader = DataLoader(
        dataset, collate_fn=collate_fn, batch_sampler=train_sampler, 
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )


    # Model
    model = model_list[args.model](num_classes=len(class_list)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Nettoyage des clés '_orig_mod' si modèle compilé
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)

    model = torch.compile(model)

    print('Evaluating model...')
    # Evaluate
    labels, binary, probs = evaluate(model, loader, device, args.threshold)

    # Metrics
    print('- AUROC and AUPRC...')
    auroc, auprc, auroc_c, auprc_c = compute_auc(labels, probs)
    print('- Accuracy...')
    acc = compute_accuracy(labels, binary)
    print('- F-measure...')
    f1, f1_c = compute_f_measure(labels, binary)
    print('- Challenge metric...')
    challenge = compute_challenge_metric(
        # uses the specific "Sinus Rhythm" (NSR) baseline to see if the model is actually learning 
        # heart pathologies or just guessing the most common class
        weights, labels, binary, classes, "NSR") # normal ecg

    print('Done.')

    # ================= LOGGING WANDB ================= 
    # Métriques globales
    metrics = {
        "eval/AUROC":           auroc,
        "eval/AUPRC":           auprc,
        "eval/Accuracy":        acc,
        "eval/Macro_F1":        f1,
        "eval/Challenge_Score": challenge,
        "eval/threshold":       args.threshold,
    }
    wandb.log(metrics)

    # Mise à jour du résumé WandB (visible au premier coup d'oeil sur le dashboard)
    wandb.run.summary["AUROC"]           = auroc
    wandb.run.summary["AUPRC"]           = auprc
    wandb.run.summary["Accuracy"]        = acc
    wandb.run.summary["Macro_F1"]        = f1
    wandb.run.summary["Challenge_Score"] = challenge
    wandb.run.summary["checkpoint"]      = args.checkpoint

    # Table des métriques par classe (F1 et AUROC)
    class_table = wandb.Table(columns=["Class", "F1", "AUROC", "AUPRC"])
    for i, cls in enumerate(classes):
        f1_val    = float(f1_c[i])    if np.isfinite(f1_c[i])    else None
        auroc_val = float(auroc_c[i]) if np.isfinite(auroc_c[i]) else None
        auprc_val = float(auprc_c[i]) if np.isfinite(auprc_c[i]) else None
        class_table.add_data(cls, f1_val, auroc_val, auprc_val)

    wandb.log({"eval/per_class_metrics": class_table})

    # Upload du checkpoint évalué comme artifact
    print("[WANDB] Upload du checkpoint évalué en cours...")
    artifact = wandb.Artifact(f"eval-{wandb.run.id}", type="evaluation")

    if os.path.exists(args.checkpoint):
        artifact.add_file(args.checkpoint)
    wandb.log_artifact(artifact)

    wandb.finish()

    # ================= Save CSV ================= 

    name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # Overall metrics
    output_csv = os.path.join(args.output, f"{name}_metrics.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AUROC', 'AUPRC', 'Accuracy', 'Macro_F1', 'Challenge_Score'])
        writer.writerow([auroc, auprc, acc, f1, challenge])
    print(f"Overall metrics saved to {output_csv}")

    # Per-class metrics
    output_csv_per_class = os.path.join(args.output, f"{name}_metrics_per_class.csv")
    with open(output_csv_per_class, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'F1', 'AUROC', 'AUPRC'])
        for i, cls in enumerate(classes):
            f1_val    = float(f1_c[i])    if np.isfinite(f1_c[i])    else 'nan'
            auroc_val = float(auroc_c[i]) if np.isfinite(auroc_c[i]) else 'nan'
            auprc_val = float(auprc_c[i]) if np.isfinite(auprc_c[i]) else 'nan'
            writer.writerow([cls, f1_val, auroc_val, auprc_val])
    print(f"Per-class metrics saved to {output_csv_per_class}")


if __name__ == "__main__":
    # Configuration PyTorch pour éviter la fragmentation mémoire CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    main()