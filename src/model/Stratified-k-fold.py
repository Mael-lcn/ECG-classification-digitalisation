import os, sys
import argparse
import re
import time
import math
from tqdm import tqdm
import glob
import shutil
import warnings

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser, build_model
from train import generate_exp_name, load_pos_weight, run_training_loop



# On dit à Python tkt c'est pas grave pour ce warning précis
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

TOTAL_NB_FILE = 27  # Nombre total de file dans le dataset complet (train + val + test)
torch.set_float32_matmul_precision('high')

need_compile = set(['PatchTSTModel', 'DinoTraceTemporal', 'ViT_TimeFreq', 'ViT_Image'])


def create_fold(train_path, val_path, eval_path):
    """
    
    """
    repartition = {
        "train": [],
        "val": [],
        "eval": []
    }

    return repartition


def run(args):
    # 1. Configuration Matérielle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        device = torch.device(f"cuda:{args.gpu}")
        use_amp = not args.not_use_amp

        if use_amp and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("Matériel compatible détecté (Ampere+) : BFloat16 activé !")
        else:
            amp_dtype = torch.float16
            print("BFloat16 non supporté sur ce GPU : Fallback sur Float16.")

        # Mode bench pour conv rapide
        torch.backends.cudnn.benchmark = args.use_static_padding

        # Gère flash attention test dans cette ordre qui correspond aux implémentations les plus rapides
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        print(f"[INIT] Mode: CUDA AMP is {use_amp}")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU")


    # 2. Configuration des dossiers
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # Configuration WandB pour environnement restreint (Offline)
    os.environ["WANDB_MODE"] = "offline" 
    os.environ["WANDB_DIR"] = os.path.join(args.output, "wandb_logs")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # 3. Gestion de l'ID WandB (Pour la reprise/resume)
    wandb_id = wandb.util.generate_id()
    resume_mode = "allow"
    id_file_path = os.path.join(args.checkpoint_dir, "wandb_run_id.txt")

    if args.resume_from and os.path.exists(id_file_path):
        with open(id_file_path, "r") as f:
            old_id = f.read().strip()
            if old_id:
                wandb_id = old_id
                resume_mode = "must"
                print(f"[WANDB] Reprise du run ID : {wandb_id}")

    # 4. Créer le model
    model, valid_kwargs, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)
    model_name = args.model_name

    # 5. Génération du nom d'expérience dynamique
    exp_name = generate_exp_name(args, valid_kwargs, wandb_id)

    # 6. Initialisation WandB
    wandb.init(
        project="ECG_Classification_Experiments",
        group=exp_name,
        job_type="train",
        name=f"run_{model_name}_{wandb_id[:6]}",
        config=args,
        id=wandb_id,
        resume=resume_mode,
        tags=["scientific", args.model_name, "offline"]
    )

    # Sauvegarde de l'ID pour une future reprise
    with open(id_file_path, "w") as f:
        f.write(wandb.run.id)

    # Définition des axes pour des graphiques cohérents
    wandb.define_metric("val/pr_auc", step_metric="epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("perf/*", step_metric="epoch")

    print(f"Début de l'expérience : {exp_name}")
    print(f"[INIT] Préparation des TurboDatasets (Format .npy)...")

    mb_size = args.batch_size_theoric * args.mega_batch_factor
    dataset_kwargs = {
        "batch_size": args.batch_size_accumulat,
        "mega_batch_size": mb_size,
        "use_static_padding": args.use_static_padding
    }

    if gen_fun is not None:
        dataset_kwargs["generate_img"] = gen_fun

    # Création des Datasets
    train_ds = Dataset_fun(
        data_path=args.train_data,
        **dataset_kwargs
    )

    val_ds = Dataset_fun(
        data_path=args.val_data, 
        **dataset_kwargs
    )

    test_ds = Dataset_fun(
        data_path=args.val_data, 
        **dataset_kwargs
    )

    # Création des DataLoaders
    # IMPORTANT : batch_size=None car le Dataset renvoie déjà des batchs formés
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0), 
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2
    )

    tes_loader = DataLoader(
        test_ds,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2
    )

    # Compilation PyTorch 2.0
    try:
        if model_name in need_compile or args.use_static_padding:
            model = torch.compile(model)
    except Exception as e:
        print(f"[INFO] Torch Compile ignoré ou échoué: {e}")


    # On sépare les paramètres en deux groupes
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # On identifie les paramètres du backbone par leur nom
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Configuration des groupes pour l'optimiseur
    param_groups = [
        {
            "params": backbone_params, 
            "lr": args.backbone_lr,
            "name": "backbone"
        },
        {
            "params": head_params, 
            "lr": args.lr, # LR plein pour les nouvelles couches
            "name": "head"
        }
    ]

    print(f"[INIT] Params Backbone: {len(backbone_params)} | Params Head: {len(head_params)}")
    print(f"[INIT] Backbone LR Factor: {args.backbone_lr}")

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)

    # MODIFIED: Build criterion with optional pos_weight
    pos_weight_tensor = load_pos_weight(
        pos_weight_path=args.pos_weight_path,
        num_classes=args.num_classes, 
        device=device
    )

    # BCEWithLogitsLoss accepts pos_weight=None natively (falls back to no weighting)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    scaler_amp = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

    # 7. Logique de Reprise
    start_epoch = 1
    best_val_pr_auc = -1

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[RESUME] Chargement du checkpoint depuis : {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Chargement strict du nouveau format de Checkpoint robuste
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_pr_auc = checkpoint.get('best_val_pr_auc', -1)
        
        if scaler_amp and 'scaler_state_dict' in checkpoint:
            scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"[RESUME] Reprise confirmée à l'époque {start_epoch} (Best score précédent: {best_val_pr_auc:.4f})")

    # Si reprise WandB SANS args.resume_from (ex: on a juste l'ID mais on repart de zéro)
    elif wandb.run.summary.get("best_val_pr_auc"):
        saved_best = wandb.run.summary["best_val_pr_auc"]
        if isinstance(saved_best, (int, float)) and math.isfinite(saved_best):
            best_val_pr_auc = saved_best
            print(f"[RESUME] Best score précédent récupéré de WandB: {best_val_pr_auc:.4f}")


    # 8.Boucle des epochs
    run_training_loop(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_ds=train_ds,
        optimizer=optimizer,
        criterion=criterion,
        scaler_amp=scaler_amp,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        start_epoch=start_epoch,
        best_val_pr_auc=best_val_pr_auc,
        exp_name=exp_name,
        model_name=model_name,
        fold=-1
    )

    # Clôture du run W&B
    wandb.finish()



def main():
    """
    Point d'entrée du script. Gestion des arguments CLI.
    """
    # On récupère le parser de base
    shared_parser = get_shared_parser()

    # On crée le parser de train en héritant du shared_parser
    parser = argparse.ArgumentParser(
        description="Script d'entraînement", 
        parents=[shared_parser]
    )

    # Arguments Dossiers & Fichiers
    parser.add_argument('--train_data', type=str, default="../../../output/cross_val_data/train", 
                        help="Dossier contenant les fichiers H5 de train")
    parser.add_argument('--val_data', type=str, default="../../../output/cross_val_data/val", 
                        help="Dossier contenant les fichiers H5 de validation")
    parser.add_argument('--test_data', type=str, default="../../../output/cross_val_data/test", 
                        help="Dossier contenant les fichiers H5 de validation")

    # Hyperparamètres
    parser.add_argument('-k', type=int, default=5, help="Nombre de plis (folds) pour la validation croisée")
    parser.add_argument('--epochs', type=int, default=80, help="Nombre max d'époques")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate initial")
    parser.add_argument('--backbone_lr', type=float, default=1e-4, help="Learning Rate initial for the backbone")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="pénaliter pour la régularisation du model")
    parser.add_argument('--patience', type=int, default=10, help="Nb époques sans amélioration avant arrêt")
    parser.add_argument('--val_start_epoch', type=int, default=10, help="epoch ou commencer la validation")

    # Arguments Système
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement")

    # NEW ARGUMENT for pos_weight per class
    parser.add_argument(
        '--pos_weight_path', type=str, default="../ressources/pos_weight.pt",
        help=(
            "Path to the pre-computed pos_weight .pt file "
            "(output of compute_pos_weight.py). "
            "If omitted, BCEWithLogitsLoss runs without class weighting."
        )
    )

    if TOTAL_NB_FILE % args.k != 0:
        print(f"Attention: 27 fichiers n'est pas divisible par k={args.k}. Les plis seront inégaux.")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # Configuration PyTorch pour éviter la fragmentation mémoire CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()
