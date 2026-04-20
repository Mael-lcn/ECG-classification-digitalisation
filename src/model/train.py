import os
import sys
import time
import math
import glob
import shutil
import argparse
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser, build_model
from core_utils import (
    setup_global_environment, 
    setup_wandb, 
    run_inference, 
    load_pos_weight, 
    load_checkpoint,
    create_attention_mask,
    NEED_COMPILE
)

import warnings



warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")


def generate_exp_name(args, valid_kwargs, wandb_id):
    """Génère un nom d'expérience structuré et lisible pour WandB."""
    pad_status = "UnivPad" if args.use_static_padding else "MaxPad"
    exp_parts = [
        args.model_name, pad_status, 
        f"bs{args.batch_size_theoric}", f"lr{args.lr}", f"backbone_lr{args.backbone_lr}",
        "AMP_F" if args.not_use_amp else "AMP_T"
    ]

    abbrv = {
        'kernel_size': 'k', 'window_size1D': 'w1D', 'n_fft': 'fft', 'context_length': 'ctx',
        'patch_length': 'pt', 'd_model': 'dm', 'num_heads': 'hd', 'patch_size': 'ps'
    }

    for key, value in valid_kwargs.items():
        if key in ["num_classes", "in_channels"]: continue
        name_key = abbrv.get(key, key)
        if isinstance(value, bool): exp_parts.append(f"{name_key}{'T' if value else 'F'}")
        elif isinstance(value, list): exp_parts.append(f"{name_key}{'x'.join(map(str, value))}")
        else: exp_parts.append(f"{name_key}{value}")

    exp_parts.append(wandb_id[:6])
    return "-".join(exp_parts)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    total_epochs,
    use_amp,
    amp_dtype,
    accum_steps
):
    """
    Exécute une époque complète d'entraînement du modèle avec prise en charge de l'accumulation de gradients et de la précision mixte automatique (AMP).

    Args:
        model (torch.nn.Module): Le modèle à entraîner.
        dataloader (torch.utils.data.DataLoader): Le chargeur de données d'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur utilisé pour mettre à jour les poids.
        criterion (callable): La fonction de perte pour évaluer les prédictions.
        scaler (torch.amp.GradScaler or None): Le scaler de gradient utilisé si l'AMP est activé, sinon None.
        device (torch.device): Le périphérique cible pour l'entraînement (CPU ou GPU).
        epoch (int): L'indice de l'époque en cours.
        total_epochs (int): Le nombre total d'époques prévues.
        use_amp (bool): Indique si l'entraînement doit utiliser la précision mixte automatique.
        amp_dtype (torch.dtype): Le type de données pour la précision mixte.
        accum_steps (int): Le nombre d'étapes d'accumulation de gradients avant chaque mise à jour des poids.

    Returns:
        float: La valeur moyenne de la perte calculée sur l'ensemble de l'époque.
    """
    model.train()
    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]", dynamic_ncols=False)
    total_loss, count = 0, 0

    optimizer.zero_grad(set_to_none=True)

    for i, (tracings, targets, batch_lens) in enumerate(loop):
        tracings = tracings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_mask = create_attention_mask(batch_lens, tracings.shape[2], device)

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            predictions = model(tracings, batch_mask=batch_mask)
            loss = criterion(predictions, targets) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * accum_steps
        if math.isfinite(loss_val):
            total_loss += loss_val
            count += 1
            loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / max(1, count)


def run_training_loop(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scaler, 
    device,
    use_amp,
    amp_dtype,
    exp_name,
    start_epoch,
    best_val_pr_auc,
    fold=-1
):
    """
    Orchestre le processus global d'entraînement sur plusieurs époques, incluant l'évaluation périodique, la sauvegarde des meilleurs modèles (checkpoints) et l'arrêt prématuré (early stopping) en cas de stagnation.

    Args:
        args (argparse.Namespace): Arguments contenant les hyperparamètres d'entraînement.
        model (torch.nn.Module): Le modèle à entraîner et évaluer.
        train_loader (torch.utils.data.DataLoader): Le chargeur de données pour l'entraînement.
        val_loader (torch.utils.data.DataLoader): Le chargeur de données pour la validation.
        optimizer (torch.optim.Optimizer): L'algorithme d'optimisation.
        criterion (callable): La fonction de perte.
        scaler (torch.amp.GradScaler or None): Le scaler pour l'AMP.
        device (torch.device): Le périphérique matériel utilisé.
        use_amp (bool): Activation de la précision mixte automatique.
        amp_dtype (torch.dtype): Type de tenseur pour l'AMP.
        exp_name (str): Le nom de l'expérience utilisé pour formater les fichiers de sauvegarde.
        start_epoch (int): L'époque de départ (utile pour la reprise de l'entraînement).
        best_val_pr_auc (float): Le meilleur score PR-AUC précédemment atteint pour initialiser la sauvegarde de modèle.
        fold (int): L'index du fold actuel (utile pour le K-Fold, vaut -1 par défaut).
    """
    prefix = f"fold_{fold}/" if fold != -1 else ""

    wandb.define_metric(f"{prefix}val/pr_auc", step_metric=f"{prefix}epoch")
    wandb.define_metric(f"{prefix}train/loss", step_metric=f"{prefix}epoch")

    print(f"\n[TRAIN] Démarrage : Epoch {start_epoch} -> {args.epochs}")
    stagnation_counter = 0
    best_model_path = None
    accum_steps = max(1, args.batch_size_theoric // args.batch_size_accumulat)

    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        # 1. Phase d'entraînement
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            device, epoch, args.epochs, use_amp, amp_dtype, accum_steps
        )

        metrics = {
            f"{prefix}epoch": epoch,
            f"{prefix}train/loss": train_loss,
            f"{prefix}train/lr": optimizer.param_groups[0]['lr'], 
        }

        # 2. Phase de Validation (via le moteur universel core_utils)
        if epoch >= args.val_start_epoch:
            # On utilise le même moteur d'inférence que l'évaluation et le post_val !
            labels, probs, is_valid = run_inference(model, val_loader, device, use_amp, amp_dtype, desc=f"Ep {epoch} [VAL]")

            if is_valid:
                try:
                    pr_auc = average_precision_score(labels, probs, average='macro')
                except ValueError:
                    pr_auc = 0.0 # Sécurité si classes absentes

                metrics["val/pr_auc"] = pr_auc

                if pr_auc > best_val_pr_auc:
                    previous = best_val_pr_auc
                    best_val_pr_auc = pr_auc
                    stagnation_counter = 0

                    # Nettoyage et Sauvegarde
                    for f in glob.glob(os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep*.pt")):
                        try: os.remove(f)
                        except OSError: pass

                    best_model_path = os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep{epoch}.pt")
                    torch.save(model.state_dict(), best_model_path)
                    wandb.run.summary["best_val_pr_auc"] = best_val_pr_auc

                    print(f"    *** NEW RECORD *** {previous:.4f} -> {best_val_pr_auc:.4f} (Saved)")
                else:
                    stagnation_counter += 1
                    print(f"    [PATIENCE] {stagnation_counter}/{args.patience} (Best: {best_val_pr_auc:.4f})")
                    if stagnation_counter >= args.patience:
                        print(f"\n[STOP] Early Stopping déclenché.")
                        wandb.log(metrics)
                        break
            else:
                print(f"    [WARNING] NaNs détectés, métrique ignorée pour l'époque {epoch}.")

        # 3. Sauvegarde de sécurité
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_pr_auc': best_val_pr_auc
            }
            if scaler: checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"backup_{exp_name}_ep{epoch}.pt"))

        wandb.log(metrics)
    
    total_train_time_hours = (time.time() - total_start_time) / 3600.0
    wandb.run.summary["total_train_time_hours"] = total_train_time_hours

    # 4. Upload du modèle final sur WandB
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n[WANDB] Upload du modèle final ({os.path.basename(best_model_path)})...")
        shutil.copy2(best_model_path, os.path.join(wandb.run.dir, os.path.basename(best_model_path)))

        # On ajoute le numéro de fold au nom de l'artefact pour éviter les écrasements
        artifact_name = f"model-{args.model_name}-{wandb.run.id}"
        if fold != -1:
            artifact_name += f"-fold{fold}"

        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(os.path.join(wandb.run.dir, os.path.basename(best_model_path)))
        wandb.log_artifact(artifact)

    print(f"\n[FIN] Entraînement terminé en {total_train_time_hours:.2f} heures. Meilleur PR-AUC : {best_val_pr_auc:.4f}")


def run(args):
    """
    Point d'entrée principal du pipeline d'entraînement.
    Gère l'initialisation de l'environnement, des chargeurs de données, du modèle,
    de l'optimiseur et déclenche la boucle d'entraînement.

    Args:
        args (argparse.Namespace): Configuration globale issue de la ligne de commande.
    """
    # 1. Setup Global
    device, use_amp, amp_dtype = setup_global_environment(args)

    # 2. WandB (Gère la reprise d'ID automatiquement)
    wandb_id = wandb.util.generate_id()
    resume_mode = "allow"
    id_file = os.path.join(args.checkpoint_dir, "wandb_run_id.txt")

    if args.resume_from and os.path.exists(id_file):
        with open(id_file, "r") as f: wandb_id = f.read().strip() or wandb_id
        resume_mode = "must"

    # Build Model pour obtenir les kwargs valides nécessaires au nommage
    model, valid_kwargs, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)
    exp_name = generate_exp_name(args, valid_kwargs, wandb_id)

    setup_wandb(args, job_type="train", run_name=f"run_{exp_name}", wandb_id=wandb_id, resume_mode=resume_mode)
    with open(id_file, "w") as f: f.write(wandb.run.id)


    # 3. Dataloaders
    dataset_kwargs = {
        "batch_size": args.batch_size_accumulat,
        "mega_batch_size": args.batch_size_theoric * args.mega_batch_factor,
        "use_static_padding": args.use_static_padding,
        "h": args.input_h, 
        "w": args.input_w 
    }
    if gen_fun: dataset_kwargs["generate_img"] = gen_fun

    train_loader = DataLoader(
        Dataset_fun(args.train_data, **dataset_kwargs),
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        Dataset_fun(args.val_data, is_train=False,  **dataset_kwargs),
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2
    )

    # 4. Compilation PyTorch 2.0
    try:
        if args.model_name in NEED_COMPILE or args.use_static_padding:
            model = torch.compile(model)
    except Exception as e:
        print(f"[INFO] Torch Compile ignoré: {e}")

    # 5. Optimiseur & Loss
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n], "lr": args.backbone_lr, "name": "backbone"},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n], "lr": args.lr, "name": "head"}
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=load_pos_weight(args.pos_weight_path, args.num_classes, device))
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

    # 6. Reprise
    start_epoch, best_score = load_checkpoint(args.resume_from if args.resume_from else "", model, device, optimizer, scaler)
    if not args.resume_from and wandb.run.summary.get("best_val_pr_auc"):
        best_score = float(wandb.run.summary["best_val_pr_auc"])

    # 7. Lancement
    run_training_loop(args, model, train_loader, val_loader, optimizer, criterion, scaler, device, use_amp, amp_dtype, exp_name, start_epoch, best_score)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Script d'entraînement standard", parents=[get_shared_parser()])
    parser.add_argument('--train_data', type=str, default="../../../output/final_data/train")
    parser.add_argument('--val_data', type=str, default="../../../output/final_data/val")

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_start_epoch', type=int, default=15)

    # Arguments Système
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement")

    # Argumentspos_weight per class
    parser.add_argument(
        '--pos_weight_path', type=str, default="../ressources/pos_weight.pt",
        help=(
            "Path to the pre-computed pos_weight .pt file "
            "(output of compute_pos_weight.py). "
            "If omitted, BCEWithLogitsLoss runs without class weighting."
        )
    )

    # Argument for Cnn Image
    parser.add_argument('--input_h', type=int, default=512)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--cnn_mode', type=str, default='square', choices=['square', 'rectangle'],
                        help="Mode for CNN_Image: 'square' (3x3) or 'rectangle' (half-height)")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
