import os
import shutil
import sys
import json
import argparse
import numpy as np
import torch

import multiprocessing
import csv

import wandb

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser, build_model, create_dataloader
from core_utils import (
    setup_global_environment,
    setup_wandb,
    run_inference,
    load_checkpoint
)



def compute_accuracy(labels, outputs):
    num_recordings = labels.shape[0]
    correct = np.all(labels == outputs, axis=1)
    return float(np.sum(correct)) / float(num_recordings)

def compute_confusion_matrices(labels, outputs, normalize=False):
    num_recordings, num_classes = np.shape(labels)
    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: A[j, 0, 0] += 1
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: A[j, 0, 0] += 1.0/normalization
    return A

def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)
    A = compute_confusion_matrices(labels, outputs)
    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn: f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else: f_measure[k] = float('nan')

    macro_f_measure = np.nanmean(f_measure) if np.any(np.isfinite(f_measure)) else float('nan')
    return macro_f_measure, f_measure

def compute_auc(labels, outputs):
    num_recordings, num_classes = labels.shape
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]

        tp, fp, fn, tn = np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds))
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
        union_count = np.sum(labels[i] | outputs[i])
        if union_count > 0:
            A += np.outer(labels[i], outputs[i]) / float(union_count)
    return A

def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = labels.shape
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

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
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def is_finite_number(x):
    return np.isfinite(float(x)) if is_number(x) else False

def load_table(table_file):
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    num_rows = len(table)-1
    if num_rows<1: raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1: raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1: raise Exception('The table {} is empty.'.format(table_file))

    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            values[i, j] = float(value) if is_finite_number(value) else float('nan')

    return rows, cols, values

def load_weights(weights_file):
    rows, cols, values = load_table(weights_file)
    assert(rows == cols)
    return rows, values

def round_val(val, digits=5):
    try:
        if val is not None and np.isfinite(val): return round(float(val), digits)
        return val
    except:
        return val


def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="Script d'évaluation finale", parents=[shared_parser])
    parser.add_argument('--data', default="../../../output/final_data/test", help="HDF5 test dataset directory")
    parser.add_argument('-c', '--checkpoint', required=True, help="path to checkpoint.pt")
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    parser.add_argument('--threshold', type=float, default=0.5, help="Seuil de décision du modèle")

    args = parser.parse_args()

    # Multiprocessing
    multiprocessing.set_start_method("spawn", force=True)     
    torch.set_num_threads(args.workers)         

    device, use_amp, amp_dtype = setup_global_environment(args)

    # Classes & Poids
    with open(args.class_map) as f:
        classes = json.load(f)
    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes, "Class order mismatch with weights.csv"

    base_name = args.checkpoint.split('_ep')[0]
    group_id = base_name.split('-')[-1]
    run_name = f"test_{args.model_name}_{wandb.util.generate_id()[:6]}"

    # On ajoute des infos supplémentaires spécifiques à l'eval dans l'objet args pour qu'elles soient logguées
    args.checkpoint_source = args.checkpoint

    setup_wandb(
        args=args, 
        job_type="eval", 
        run_name=run_name, 
        group=group_id, 
        tags=["eval", "final_test"]
    )

    print(f"Début de l'évaluation du checkpoint : {args.checkpoint}")

    model, _, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    load_checkpoint(checkpoint_path, model, device=device)

    # Dataloaders
    # Dataloaders
    test_loader = create_dataloader(args, args.data, Dataset_fun, gen_fun, is_train=False)

    print("Inférence en cours...")
    labels, probs, _ = run_inference(
        model, test_loader, device, use_amp, amp_dtype, 
        desc="[EVAL]", squeeze_batch=True
    )

    # Binarisation selon le seuil
    binary = (probs >= args.threshold).astype(bool)

    # 6. Calcul des métriques
    print('- AUROC and AUPRC...')
    auroc, auprc, auroc_c, auprc_c = compute_auc(labels, probs)

    print('- Accuracy...')
    final_acc = compute_accuracy(labels, binary)
    print('- F-measure...')
    final_f1, final_f1_c = compute_f_measure(labels, binary)

    A = compute_confusion_matrices(labels, binary)
    sensitivity = np.zeros(len(classes))
    specificity = np.zeros(len(classes))
    for k in range(len(classes)):
        tp, fp, fn, tn = A[k,1,1], A[k,1,0], A[k,0,1], A[k,0,0]
        sensitivity[k] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[k] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print('- Challenge metric...')
    final_challenge = compute_challenge_metric(weights, labels, binary, classes, "NSR")
    print('Terminé.')

    # 7. LOGGING WANDB & SAUVEGARDE CSV
    metrics = {
        "eval/AUROC": round_val(auroc), "eval/AUPRC": round_val(auprc),
        "eval/Accuracy": round_val(final_acc), "eval/Macro_F1": round_val(final_f1),
        "eval/Challenge_Score": round_val(final_challenge),
    }
    wandb.log(metrics)

    wandb.run.summary.update(metrics)
    wandb.run.summary["checkpoint"] = checkpoint_path

    class_table = wandb.Table(columns=["Class", "Applied_Threshold", "F1", "AUROC", "AUPRC", "Sensitivity", "Specificity"])
    for i, cls in enumerate(classes):
        class_table.add_data(
            cls, args.threshold, round_val(final_f1_c[i]), round_val(auroc_c[i]), 
            round_val(auprc_c[i]), round_val(sensitivity[i]), round_val(specificity[i])
        )
    wandb.log({"eval/per_class_metrics": class_table})

    name = args.checkpoint.replace('.pt', '')
    output_csv = os.path.join(args.output, f"{name}_metrics.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AUROC', 'AUPRC', 'Accuracy', 'Macro_F1', 'Challenge_Score'])
        writer.writerow([round_val(auroc), round_val(auprc), round_val(final_acc), round_val(final_f1), round_val(final_challenge)])

    output_csv_per_class = os.path.join(args.output, f"{name}_metrics_per_class.csv")
    with open(output_csv_per_class, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Applied_Threshold', 'F1', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity'])
        for i, cls in enumerate(classes):
            writer.writerow([cls, args.threshold, round_val(final_f1_c[i]), round_val(auroc_c[i]), round_val(auprc_c[i]), round_val(sensitivity[i]), round_val(specificity[i])])

    try:
        csv_interne_1 = os.path.join(wandb.run.dir, os.path.basename(output_csv))
        csv_interne_2 = os.path.join(wandb.run.dir, os.path.basename(output_csv_per_class))
        shutil.copy2(output_csv, csv_interne_1)
        shutil.copy2(output_csv_per_class, csv_interne_2)

        artifact = wandb.Artifact(f"eval-results-{wandb.run.id}", type="evaluation-data")
        artifact.add_file(csv_interne_1, name=os.path.basename(output_csv))
        artifact.add_file(csv_interne_2, name=os.path.basename(output_csv_per_class))
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"[WARN] Artifact upload failed: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
