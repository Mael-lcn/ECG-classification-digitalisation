import os
import json
import argparse
import re
import time

import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb  # Librairie de monitoring

from torch.utils.data import DataLoader
from Dataset import LargeH5Dataset, ecg_collate_wrapper
from Sampler import MegaBatchSortishSampler

from Cnn import CNN



def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, total_epochs, use_amp):
    """
    Exécute une époque d'entraînement complète (Forward + Backward pass).

    Args:
        model (nn.Module): Le réseau de neurones à entraîner.
        dataloader (DataLoader): Le générateur de batchs d'entraînement.
        optimizer (torch.optim): L'optimiseur (ex: AdamW).
        criterion (nn.Module): La fonction de perte (ex: BCEWithLogitsLoss).
        scaler (torch.amp.GradScaler): Scaler pour la précision mixte (FP16). Peut être None.
        device (torch.device): 'cuda' ou 'cpu'.
        epoch (int): Numéro de l'époque actuelle (pour l'affichage).
        total_epochs (int): Nombre total d'époques.
        use_amp (bool): Si True, active l'Automatic Mixed Precision (plus rapide).

    Returns:
        float: La perte moyenne (loss) sur cette époque.
    """
    model.train()  # Active le mode entraînement (active Dropout, BatchNorm, etc.)

    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]")
    total_loss = 0
    count = 0

    for batch in loop:
        # Récupération des données et envoi sur GPU
        tracings, targets = batch
        tracings = tracings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Remise à zéro des gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward Pass avec AMP
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            predictions = model(tracings)
            loss = criterion(predictions, targets)

        # Backward Pass
        if scaler:
            # Si AMP activé : on scale la loss pour éviter les underflows en FP16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Mode standard (FP32)
            loss.backward()
            optimizer.step()

        # Calcul des stats
        loss_val = loss.item()
        total_loss += loss_val
        count += 1

        # Mise à jour de la barre de progression
        loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device, use_amp):
    """
    Évalue le modèle sur le jeu de validation.
    
    Note: Désactive le calcul des gradients pour économiser la mémoire.

    Args:
        model (nn.Module): Le modèle à évaluer.
        dataloader (DataLoader): Le générateur de batchs de validation.
        criterion (nn.Module): La fonction de perte.
        device (torch.device): 'cuda' ou 'cpu'.
        use_amp (bool): Si True, utilise l'autocast.

    Returns:
        float: La perte moyenne (loss) sur le jeu de validation.
    """
    model.eval()  # Active le mode évaluation (fige Dropout, BatchNorm)

    loop = tqdm(dataloader, desc="[VAL]")
    total_loss = 0
    count = 0

    # torch.no_grad() empêche PyTorch de stocker les calculs pour la backprop
    with torch.no_grad():
        for batch in loop:
            tracings, targets = batch
            tracings = tracings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
                predictions = model(tracings)
                loss = criterion(predictions, targets)

            total_loss += loss.item()
            count += 1
            loop.set_postfix(val_loss=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


def run(args):
    """
    Fonction principale qui orchestre tout le processus d'entraînement.
    
    Étapes :
    1. Init WandB.
    2. Chargement des données (Train/Val).
    3. Création du modèle et de l'optimiseur.
    4. Boucle d'entraînement (Train -> Val -> Log -> Save).
    """
    # On désactive toute tentative de connexion réseau
    os.environ["WANDB_MODE"] = "offline"

    # On s'assure que les dossiers de logs pointent vers le disque temporaire
    os.environ["WANDB_DIR"] = os.path.join(args.output, "wandb_logs")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)


    pad_status = "UnivPad" if args.use_static_padding else "MaxPad"

    exp_name = f"EXP_{ model_list[args.model].__name__}_bs{args.batch_size}_lr{args.lr} \
            _ep{args.epochs}_{pad_status}_{args.patience}"

    # Construction du nom d'expérience
    wandb.init(
        project="ECG_Classification_Experiments",
        config=args,
        name=exp_name
    )

    print(f"Début de {exp_name}")

    # Préparation des données
    print(f"[INIT] Chargement des classes depuis : {args.class_map}")
    with open(args.class_map, 'r') as f:
        loaded_classes = json.load(f)
    num_classes = len(loaded_classes)

    print(f"[DATA] Train: {args.train_data}")
    print(f"[DATA] Val:   {args.val_data}")

    # Dataset
    train_ds = LargeH5Dataset(
        input_dir=args.train_data,
        classes_list=loaded_classes,
        use_static_padding=args.use_static_padding
    )

    val_ds = LargeH5Dataset(
        input_dir=args.val_data,
        classes_list=loaded_classes,
        use_static_padding=args.use_static_padding
    )

    # Sampler
    train_sampler = MegaBatchSortishSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    val_sampler = MegaBatchSortishSampler(val_ds, batch_size=args.batch_size, shuffle=False)

    collate_fn = partial(ecg_collate_wrapper, 
                     use_static_padding=args.use_static_padding)

    # Dataloader de Training
    train_loader = DataLoader(
        dataset=train_ds,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 
    )

    # Dataloader de Validation
    val_loader = DataLoader(
        dataset=val_ds,
        collate_fn=collate_fn,
        batch_sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Setup Matériel & Modèle
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        torch.backends.cudnn.benchmark = True # Optimise les algos de convolution via triton
        print("[INIT] Mode: CUDA (NVIDIA)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU (Lent)")

    print(f"[MODEL] Création du modèle {model_list[args.model].__name__} pour {num_classes} classes...")
    model = model_list[args.model](num_classes=num_classes).to(device)

    try:
        model = torch.compile(model, dynamic=True)
    except Exception as e:
        print(f"[INFO] torch.compile ignoré : {e}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # Gestion de la Reprise
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[RESUME] Chargement du checkpoint : {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        # Gestion des clés si le modèle a été compilé (préfixe '_orig_mod.')
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)

        # Tentative de récupérer l'époque depuis le nom du fichier
        match = re.search(r'ep(\d+)', args.resume_from)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"[RESUME] Reprise à l'époque {start_epoch}")

    # Boucle d'Entraînement
    print(f"\n[TRAIN] Début de l'entraînement pour {args.epochs} époques.")
    stagnation_counter = 0
    
    train_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        # A. Entraînement
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
        }

        val_loss = None
        # Validation dès 15 epoch. Inutile avant
        if epoch > 15:
            # B. Validation
            val_loss = validate(model, val_loader, criterion, device, use_amp)

            # C. Sauvegarde du Meilleur Modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stagnation_counter = 0

                # Nom du fichier
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")

                # Sauvegarde locale
                torch.save(model.state_dict(), save_path)
                print(f"    *** NEW RECORD *** Modèle sauvegardé : {save_path}")
            else:
                stagnation_counter += 1
                print(f"    [PATIENCE] Pas d'amélioration ({stagnation_counter}/{args.patience})")

                # D. Early Stopping
                if stagnation_counter >= args.patience:
                    print(f"\n[STOP] Arrêt précoce déclenché (Patience {args.patience} atteinte).")
                    print(f"       Meilleur score : {best_val_loss:.4f}")
                    break

        # E. Backup périodique
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"backup_ep{epoch}.pt"))

        if val_loss:
            metrics["val/loss"] =  val_loss

        # F. Envoi des stats à WandB
        wandb.log(metrics)


    # Enregistre le temps total et les meilleurs résultats dans le résumé du run
    wandb.run.summary["total_train_time_hours"] = (time.time() - train_start_time) / 3600
    wandb.run.summary["best_val_loss"] = best_val_loss

    # Fin du script
    wandb.finish()
    print("[FIN] Script terminé avec succès.")



model_list = [CNN]


def main():
    """
    Point d'entrée du script. Parse les arguments CLI.
    """
    options = ", ".join([f"{i}: {model}" for i, model in enumerate(model_list)])


    parser = argparse.ArgumentParser(description="Script d'entraînement ECG avec WandB")

    # Arguments du script
    parser.add_argument('--train_data', type=str, default="../output/final_data/train", 
                        help="Dossier contenant les fichiers H5 de train")
    parser.add_argument('--val_data', type=str, default="../output/final_data/val", 
                        help="Dossier contenant les fichiers H5 de validation")
    parser.add_argument('--class_map', type=str, default='../../ressources/final_class.json',
                        help="Chemin JSON mappant les indices aux noms de classes")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="Dossier où sauvegarder les poids (.pt)")
    parser.add_argument('--output', type=str, default='../output/',
                        help="Dossier de sortie standart")
    parser.add_argument('--model', type=int, default=0,
                        help=f"Quel modèle voulez-vous entrainer: {options}")

    # Arguments du modèle (Hyperparamètres)
    parser.add_argument('--batch_size', type=int, default=64, help="Taille du batch")
    parser.add_argument('--epochs', type=int, default=50, help="Nombre max d'époques")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate initial")
    parser.add_argument('--patience', type=int, default=6, help="Nb époques sans amélioration avant arrêt")
    parser.add_argument(
        '--use_static_padding', 
        action='store_true', 
        default=False,
        help="Si activé, force une taille de padding fixe (universelle) pour tous les batchs. "
            "Par défaut, le padding est dynamique (ajusté à la longueur max du lot actuel)."
    )

    # Arguments Système
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help="Nombre de processus pour charger les données")
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement")

    args = parser.parse_args()


    # Config PyTorch pour éviter la fragmentation CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Lancement
    run(args)


if __name__ == "__main__":
    main()
