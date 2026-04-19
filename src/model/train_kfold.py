import os, time
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

# Importation des briques modulaires
from model_factory import get_shared_parser, build_model
from core_utils import (
    setup_global_environment, 
    setup_wandb, 
    load_pos_weight, 
    NEED_COMPILE
)
# On importe la boucle d'entraînement du script de base
from train import run_training_loop, generate_exp_name 



def create_symlink_fold(fold_dir, train_files, val_files):
    """
    Génère une structure de répertoires temporaire pour l'itération de validation croisée et crée des liens symboliques
    vers les fichiers de données originaux.

    Args:
        fold_dir (str): Le chemin du répertoire temporaire racine dédié au pli actuel.
        train_files (list or numpy.ndarray): Les chemins absolus des fichiers alloués à l'ensemble d'entraînement.
        val_files (list or numpy.ndarray): Les chemins absolus des fichiers alloués à l'ensemble de validation.

    Returns:
        tuple: Un tuple contenant les chemins absolus des deux sous-répertoires créés :
            - train_dir (str): Le répertoire contenant les liens symboliques d'entraînement.
            - val_dir (str): Le répertoire contenant les liens symboliques de validation.
    """
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        os.symlink(f, os.path.join(train_dir, os.path.basename(f)))
    for f in val_files:
        os.symlink(f, os.path.join(val_dir, os.path.basename(f)))

    return train_dir, val_dir


def run_kfold_pipeline(args):
    """
    Orchestre le pipeline complet de validation croisée à K plis avec une isolation stricte des sorties par dossier.

    Réalise la collecte récursive des données, définit un groupe d'expérience unique et itère sur les plis de validation. 
    Pour chaque pli, un sous-répertoire de sauvegarde dédié est créé et la boucle d'entraînement 
    est exécutée de manière cloisonnée. Le processus gère également la redirection dynamique des chemins 
    de sauvegarde et calcule le temps total d'exécution de l'expérience à l'issue du processus.

    Args:
        args (argparse.Namespace): Les arguments de configuration contenant les hyperparamètres d'entraînement,
                                    les chemins des données et les paramètres de la validation croisée.

    Raises:
        ValueError: Si aucun fichier de données correspondant au motif de recherche n'est trouvé dans les répertoires spécifiés.
    """
    # 1. Collecte des fichiers
    print(f"[K-FOLD] Recherche dans : {args.data_dirs}")
    all_files = np.array([str(p.resolve()) for p in Path(args.data_dirs).rglob('*signal.npy') if p.is_file()])
    if len(all_files) == 0:
        raise ValueError(f"Aucun fichier trouvé dans {args.data_dirs}")

    # 2. Setup global
    device, use_amp, amp_dtype = setup_global_environment(args)

    # Nom unique pour le groupe d'expérience
    group_id = wandb.util.generate_id()[:6]
    group_name = f"kfold_{args.model_name}_{group_id}"

    # Création du dossier racine de l'expérience dans les checkpoints
    kfold_checkpoint_root = os.path.join(args.checkpoint_dir, group_name)
    os.makedirs(kfold_checkpoint_root, exist_ok=True)

    # Dossier pour les données temporaires
    base_tmp_data = os.path.join(args.output, f"tmp_kfold_{group_id}")

    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    total_exp_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n{'='*60}\nDÉMARRAGE FOLD {fold}/{args.k - 1}\n{'='*60}")

        # Dossier spécifique pour les checkpoints de ce fold
        fold_ckpt_dir = os.path.join(kfold_checkpoint_root, f"fold_{fold}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        # Dossier spécifique pour les données de ce fold
        fold_data_dir = os.path.join(base_tmp_data, f"fold_{fold}")
        train_path, val_path = create_symlink_fold(fold_data_dir, all_files[train_idx], all_files[val_idx])

        # Setup WandB
        wandb_id = setup_wandb(args, job_type=f"fold_{fold}", run_name=f"f{fold}_{group_id}", group=group_name)

        # Build Model
        model, valid_kwargs, Dataset_fun, gen_fun = build_model(args)
        model = model.to(device)

        if args.model_name in NEED_COMPILE or args.use_static_padding:
            try: model = torch.compile(model)
            except Exception as e: print(f"[WARN] Torch compile failed: {e}")

        # Optim & Loss
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.backbone_lr},
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.lr}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=load_pos_weight(args.pos_weight_path, args.num_classes, device))
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

        # DataLoaders
        dataset_kwargs = {
            "batch_size": args.batch_size_accumulat,
            "mega_batch_size": args.batch_size_theoric * args.mega_batch_factor,
            "use_static_padding": args.use_static_padding
        }
        if gen_fun: dataset_kwargs["generate_img"] = gen_fun

        train_loader = DataLoader(Dataset_fun(train_path, **dataset_kwargs), batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0), prefetch_factor=2)
        val_loader = DataLoader(Dataset_fun(val_path, **dataset_kwargs), batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0), prefetch_factor=2)

        # On sauvegarde le chemin original pour le restaurer après
        original_ckpt_dir = args.checkpoint_dir
        args.checkpoint_dir = fold_ckpt_dir # On injecte le dossier du fold

        # Lancement
        run_training_loop(
            args=args, model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, scaler=scaler, device=device,
            use_amp=use_amp, amp_dtype=amp_dtype, 
            exp_name=generate_exp_name(args, valid_kwargs, wandb_id),
            start_epoch=1, best_val_pr_auc=-1.0, fold=fold
        )

        # Restauration et Nettoyage
        args.checkpoint_dir = original_ckpt_dir
        wandb.finish()
        shutil.rmtree(fold_data_dir)

    # Temps total de l'expérience
    total_time = (time.time() - total_exp_start) / 3600
    print(f"\n[K-FOLD FINISHED] Temps total pour les {args.k} folds : {total_time:.2f} heures.")
    
    if os.path.exists(base_tmp_data):
        shutil.rmtree(base_tmp_data)


def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="K-Fold Training", parents=[shared_parser])
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('--data_dirs', type=str, default="../../../output/final_data/")
    parser.add_argument('--pos_weight_path', type=str, default="../ressources/pos_weight.pt")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_start_epoch', type=int, default=15)
    
    args = parser.parse_args()
    run_kfold_pipeline(args)

if __name__ == "__main__":
    main()
