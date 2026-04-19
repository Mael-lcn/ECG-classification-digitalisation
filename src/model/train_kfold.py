import os
import sys
import glob
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

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser, build_model

# On importe les bases solides qu'on a créées dans core_utils.py
from core_utils import setup_global_environment, setup_wandb, load_pos_weight
from train import run_fold, generate_exp_name  # On réutilise la logique de boucle de train.py



def create_symlink_fold(fold_dir, train_files, val_files):
    """
    Génère une structure de répertoires temporaire pour l'itération de validation croisée
    et crée des liens symboliques vers les fichiers de données originaux. 

    Args:
        fold_dir (str): Le chemin du répertoire temporaire racine dédié au pli actuel.
        train_files (list ou numpy.ndarray): Les chemins absolus des fichiers alloués à l'ensemble d'entraînement.
        val_files (list ou numpy.ndarray): Les chemins absolus des fichiers alloués à l'ensemble de validation.

    Returns:
        tuple: Un tuple contenant les chemins absolus des deux sous-répertoires créés :
            - train_dir (str): Le répertoire contenant les liens symboliques d'entraînement.
            - val_dir (str): Le répertoire contenant les liens symboliques de validation.
    """
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    # Nettoyage si le dossier existe déjà
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Création des liens (symlinks) pour le Train
    for f in train_files:
        filename = os.path.basename(f)
        os.symlink(f, os.path.join(train_dir, filename))

    # Création des liens (symlinks) pour le Val
    for f in val_files:
        filename = os.path.basename(f)
        os.symlink(f, os.path.join(val_dir, filename))

    return train_dir, val_dir


def run_kfold_pipeline(args):
    """
    Orchestre le pipeline complet d'entraînement basé sur une validation croisée à K plis.
    La fonction effectue la recherche récursive des fichiers de données,
    configure l'environnement d'exécution matériel, et procède à la division stochastique des données.
    Pour chaque itération (pli), elle réinitialise de manière stricte le modèle,
    l'optimiseur et les chargeurs de données, puis lance la boucle d'apprentissage.
    Elle intègre également le suivi des métriques via Weights & Biases de façon cloisonnée pour chaque pli et gère
    le nettoyage des données temporaires en fin d'exécution.

    Args:
        args (argparse.Namespace): Les arguments de configuration contenant les hyperparamètres d'entraînement, les chemins des données et les paramètres de la validation croisée.

    Raises:
        ValueError: Si aucun fichier de données correspondant à l'extension ciblée n'est trouvé dans les répertoires spécifiés.
    """
    # 1. Collecte de toutes les données
    print("[K-FOLD] Collecte des données depuis :", args.data_dirs)
    all_files = np.array([str(p.resolve()) for p in Path(args.data_source).rglob('*signal.npy') if p.is_file()])

    if len(all_files) == 0:
        raise ValueError("Aucun fichier trouvé dans les dossiers spécifiés.")
    print(f"[K-FOLD] Total de fichiers trouvés : {len(all_files)}")

    # 2. Configuration système globale
    device, use_amp, amp_dtype = setup_global_environment(args)

    # Génération d'un nom de groupe WandB unique pour relier tous les folds de cette expérience
    group_name = f"kfold_{args.model_name}_{wandb.util.generate_id()[:6]}"

    # Dossier parent temporaire pour héberger les sous-dossiers des symlinks
    base_tmp_kfold = os.path.join(args.output, "tmp_kfold_data")

    # 3. Préparation du Splitter
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    # 4. Boucle sur les Folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n{'='*50}\nLANCEMENT DU FOLD {fold}/{args.k - 1}\n{'='*50}")

        train_files = all_files[train_idx]
        val_files = all_files[val_idx]
        print(f"[FOLD {fold}] Train: {len(train_files)} fichiers | Val: {len(val_files)} fichiers")

        # Création des dossiers symlinks à la volée
        fold_dir = os.path.join(base_tmp_kfold, f"fold_{fold}")
        train_path, val_path = create_symlink_fold(fold_dir, train_files, val_files)

        # Setup WandB spécifique à ce fold
        wandb_id = setup_wandb(args, job_type=f"train_fold{fold}", run_name=f"run_f{fold}_{args.model_name}", group=group_name)

        # RAZ model et optimzer
        model, valid_kwargs, Dataset_fun, gen_fun = build_model(args)
        model = model.to(device)
        
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n], "lr": args.backbone_lr},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n], "lr": args.lr}
        ]

        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
        pos_weight_tensor = load_pos_weight(args.pos_weight_path, args.num_classes, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

        dataset_kwargs = {
            "batch_size": args.batch_size_accumulat,
            "mega_batch_size": args.batch_size_theoric * args.mega_batch_factor,
            "use_static_padding": args.use_static_padding
        }
        if gen_fun is not None:
            dataset_kwargs["generate_img"] = gen_fun

        train_ds = Dataset_fun(data_path=train_path, **dataset_kwargs)
        val_ds = Dataset_fun(data_path=val_path, **dataset_kwargs)

        train_loader = DataLoader(train_ds, batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0), prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0), prefetch_factor=2)

        exp_name = generate_exp_name(args, wandb_id)

        # Lancement du train pour ce fold
        run_fold(
            args=args, model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, scaler=scaler, device=device,
            use_amp=use_amp, amp_dtype=amp_dtype, exp_name=exp_name,
            start_epoch=1, best_score=-1.0, fold=fold
        )

        wandb.finish()
        shutil.rmtree(fold_dir)

    print(f"\n[K-FOLD] Expérience terminée. Nettoyage global.")
    if os.path.exists(base_tmp_kfold):
        shutil.rmtree(base_tmp_kfold)


def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="Script K-Fold d'entraînement", parents=[shared_parser])

    # Nouveaux arguments spécifiques au K-Fold
    parser.add_argument('-k', type=int, default=5, help="Nombre de splits (folds) pour la cross-validation")
    parser.add_argument('--data_dirs', type=str,default="../../../output/final_data/",
                        help="Liste des dossiers contenant les données brutes à regrouper avant le split K-Fold")

    # Hyperparamètres
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_start_epoch', type=int, default=15)

    parser.add_argument('--pos_weight_path', type=str, default="../ressources/pos_weight.pt")
    args = parser.parse_args()

    run_kfold_pipeline(args)

if __name__ == "__main__":
    main()
