import os, sys
import json
import argparse
import re
import time
import math
from tqdm import tqdm
import glob

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
from torch.utils.data import DataLoader


project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from TurboDataset import TurboDataset
from model_factory import get_shared_parser, build_model

import warnings



# On dit à Python tkt c'est pas grave pour ce warning précis
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# Supprime la limite de recompilation pour éviter les crashs avec torch.compile
torch._dynamo.config.recompile_limit = 6000

def generate_exp_name(args, valid_kwargs, wandb_id):
    pad_status = "UnivPad" if args.use_static_padding else "MaxPad"
    exp_parts = [args.model_name, pad_status]

    exp_parts.append(f"bs{args.batch_size}")
    exp_parts.append(f"lr{args.lr}")

    amp_status = "AMP_F" if args.not_use_amp else "AMP_T"
    exp_parts.append(amp_status)

    # Simplifie le nommage
    abbrv = {
        'in_channels': 'in', 'num_classes': 'cls', 'ch1': 'c1', 'ch2': 'c2', 'ch3': 'c3', 
        'kernel_size': 'k', 'window_size1D': 'w1D', 'n_fft': 'fft', 'context_length': 'ctx',
        'patch_length': 'pt', 'd_model': 'dm', 'num_heads': 'hd', 'use_cross_att': 'cross'
    }

    for key, value in valid_kwargs.items():
        if key in ["num_classes", "in_channels"]:
            continue

        name_key = abbrv.get(key, key) # Utilise le raccourci si dispo

        if isinstance(value, bool):
            val_str = "T" if value else "F"
            exp_parts.append(f"{name_key}{val_str}")
        elif isinstance(value, list):
            val_str = "x".join(map(str, value))
            exp_parts.append(f"{name_key}{val_str}")
        else:
            exp_parts.append(f"{name_key}{value}")

    exp_parts.append(wandb_id[:6])
    return "_".join(exp_parts)



def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, total_epochs, use_amp):
    """
    Exécute une époque d'entraînement complète avec tolérance aux pannes numériques.

    Cette fonction itère sur le DataLoader, effectue les passes avant/arrière et met à jour les poids.
    Elle intègre une gestion robuste des erreurs : si une perte (Loss) devient NaN ou Inf, 
    le batch incriminé est ignoré et consigné dans un rapport d'erreur WandB, évitant ainsi 
    le crash complet de l'entraînement.

    Args:
        model (nn.Module): Le réseau de neurones à entraîner.
        dataloader (DataLoader): Le chargeur de données (doit renvoyer tracings, targets, lengths).
        optimizer (torch.optim.Optimizer): L'optimiseur (ex: AdamW).
        criterion (nn.Module): La fonction de perte (ex: BCEWithLogitsLoss).
        scaler (torch.amp.GradScaler | None): Scaler pour la gestion de la précision mixte (AMP).
            Si None, l'entraînement se fait en précision standard (FP32).
        device (torch.device): Le périphérique de calcul (CPU ou CUDA).
        epoch (int): L'index de l'époque courante (pour l'affichage/logging).
        total_epochs (int): Nombre total d'époques prévues.
        use_amp (bool): Indique si l'Automatic Mixed Precision est activée.

    Returns:
        float: La perte moyenne de l'époque. Retourne `float('inf')` si tous les batchs 
        de l'époque ont échoué (cas critique).
    """
    model.train()

    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]")
    total_loss = 0
    count = 0

    for batch in loop:
        tracings, targets, batch_mask = batch
        tracings = tracings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            predictions = model(tracings, batch_mask)
            loss = criterion(predictions, targets)

        # Backward Pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Logging
        loss_val = loss.item()

        if math.isfinite(loss_val):
            total_loss += loss_val
            count += 1
            loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")
        else:
            # Enregistrement des preuves pour debug
            try:
                crash_table = wandb.Table(
                    columns=["Epoch", "Batch", "Loss", "Targets Sample"],
                    data=[[epoch, count, loss_val, str(targets[0].tolist())]]
                )
                wandb.log({"errors/train_crash_report": crash_table})
            except Exception:
                pass

    # Retourne la moyenne ou l'infini si tout a échoué
    return total_loss / count if count > 0 else float('inf')



def validate(model, dataloader, criterion, device, use_amp, epoch):
    """
    Évalue le modèle sur le jeu de validation en mode inférence.

    Cette fonction calcule la perte moyenne sur le dataset de validation sans calculer les gradients.
    Elle utilise également le contexte AMP pour accélérer l'inférence et surveille l'apparition
    de valeurs aberrantes (NaN/Inf) pour générer des rapports de diagnostic sans interrompre le processus.

    Args:
        model (nn.Module): Le modèle à évaluer (sera passé en mode .eval()).
        dataloader (DataLoader): Le chargeur de données de validation.
        criterion (nn.Module): La fonction de perte pour calculer le score.
        device (torch.device): Le périphérique de calcul.
        use_amp (bool): Indique si l'inférence doit utiliser la précision mixte.
        epoch (int): Numéro de l'époque actuelle (utilisé pour taguer les logs d'erreurs).

    Returns:
        float: La perte moyenne de validation. Retourne `float('inf')` si le calcul échoue globalement.
    """
    model.eval()
    loop = tqdm(dataloader, desc=f"Ep {epoch} [VAL]")
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loop:
            tracings, targets, batch_mask = batch
            tracings = tracings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
                predictions = model(tracings, batch_mask)
                loss = criterion(predictions, targets)

            loss_val = loss.item()

            # Vérification Mathématique
            if math.isfinite(loss_val):
                total_loss += loss_val
                count += 1
                loop.set_postfix(val_loss=f"{total_loss/count:.4f}")
            else:
                try:
                    crash_table = wandb.Table(
                        columns=["Epoch", "Loss", "Targets Sample"],
                        data=[[epoch, loss_val, str(targets[0].tolist())]]
                    )
                    wandb.log({"errors/val_crash_report": crash_table})
                except Exception:
                    pass

    # Si count > 0, on retourne la moyenne. Sinon, on retourne l'INFINI
    return total_loss / count if count > 0 else float('inf')



def run(args):
    """
    Orchestre le cycle de vie complet de l'expérience d'entraînement (Pipeline principal).

    Cette fonction configure l'environnement, initialise les composants et gère la boucle d'entraînement.
    Les étapes clés incluent :

    1. Setup Système : Création des dossiers, configuration de WandB (mode Offline/Online) et 
    gestion de la reprise (Resume) via ID de run.
    2. Data Loading : Instanciation du `LargeH5Dataset` et du `MegaBatchSortishSampler` 
    pour optimiser le débit I/O.
    3. Optimisation : Compilation du modèle via `torch.compile` (PyTorch 2.0+) et configuration
    de l'optimiseur AdamW.
    4. Boucle Train/Val : Exécution séquentielle avec calcul de métriques en temps réel.
    5. Sauvegarde : Checkpoints périodiques, sauvegarde du "Best Model" sur amélioration de la 
    validation, et mécanisme d'Early Stopping.

    Args:
        args (argparse.Namespace): Objet contenant tous les hyperparamètres et configurations 
        (batch_size, lr, paths, etc.) parsés depuis la ligne de commande.
    """

    # 1. Configuration Matérielle
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = args.not_use_amp
        torch.backends.cudnn.benchmark = args.use_static_padding
        print("[INIT] Mode: CUDA")
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
    print(f"[INIT] Instanciation du modèle {args.model_name}...")
    model, valid_kwargs = build_model(args)
    model = model.to(device)

    # 5. Génération du nom d'expérience dynamique
    exp_name = generate_exp_name(args, valid_kwargs, wandb_id)

    # 6. Initialisation WandB
    wandb.init(
        project="ECG_Classification_Experiments",
        group=exp_name,
        job_type="train",
        name=f"run_{wandb_id[:6]}",
        config=args,
        id=wandb_id,
        resume=resume_mode,
        tags=["scientific", args.model_name, "offline"]
    )

    # Sauvegarde de l'ID pour une future reprise
    with open(id_file_path, "w") as f:
        f.write(wandb.run.id)

    # Définition des axes pour des graphiques cohérents
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("perf/*", step_metric="epoch")

    print(f"Début de l'expérience : {exp_name}")
    print(f"[INIT] Préparation des TurboDatasets (Format .npy)...")

    # Création du Dataset d'entraînement
    mb_size = args.batch_size * args.mega_batch_factor

    # Création du Dataset d'entraînement
    train_ds = TurboDataset(
        data_path=args.train_data,
        batch_size=args.batch_size,
        mega_batch_size=mb_size,
        use_static_padding=args.use_static_padding
    )

    # Création du Dataset de validation
    val_ds = TurboDataset(
        data_path=args.val_data, 
        batch_size=args.batch_size,
        mega_batch_size=mb_size,
        use_static_padding=args.use_static_padding
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

    # Log les gradients et poids tous les 500 batchs pour diagnostic mais casse torch.compile !!!!
    # wandb.watch(model, log="all", log_freq=500)

    # Compilation PyTorch 2.0
    try:
        if args.use_static_padding:
            model = torch.compile(model)
    except Exception as e:
        print(f"[INFO] Torch Compile ignoré ou échoué: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # 7. Logique de Reprise
    start_epoch = 1
    best_val_loss = float('inf')

    # Si reprise WandB, on tente de récupérer le meilleur score connu
    if wandb.run.summary.get("best_val_loss"):
        saved_best = wandb.run.summary["best_val_loss"]
        if isinstance(saved_best, (int, float)) and math.isfinite(saved_best):
            best_val_loss = saved_best
            print(f"[RESUME] Best score précédent récupéré de WandB: {best_val_loss}")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[RESUME] Chargement des poids depuis : {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        # Nettoyage des clés '_orig_mod' si modèle compilé
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)

        # Récupération de l'époque
        match = re.search(r'ep(\d+)', args.resume_from)
        if match:
            start_epoch = int(match.group(1)) + 1


    # 8.Boucle des epochs
    print(f"\n[TRAIN] Démarrage : Epoch {start_epoch} -> {args.epochs}")
    stagnation_counter = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # A. Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )

        # Calcul temps & débit
        epoch_duration = time.time() - epoch_start
        samples_per_sec = len(train_ds) / epoch_duration if epoch_duration > 0 else 0

        # Métriques brutes
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/lr": optimizer.param_groups[0]['lr'], 
            "perf/epoch_duration": epoch_duration,
            "perf/samples_per_sec": samples_per_sec
        }

        # B. Validation
        if epoch >= 15:
            val_loss = validate(model, val_loader, criterion, device, use_amp, epoch)
            metrics["val/loss"] = val_loss

            # C. Si la val a fonctionnée
            if math.isfinite(val_loss):
                # C.1 Nouveau Record
                if val_loss < best_val_loss:
                    previous = best_val_loss
                    best_val_loss = val_loss
                    stagnation_counter = 0
                    
                    # Supprime l'ancienne best
                    old_files = glob.glob(os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep*.pt"))
                    for f in old_files:
                        try:
                            os.remove(f)
                        except OSError as e:
                            print(f"Erreur lors de la suppression de {f}: {e}")

                    # Sauvegarde disque du model
                    save_path = os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep{epoch}.pt")
                    torch.save(model.state_dict(), save_path)

                    # Mise à jour du résumé WandB
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch

                    print(f"    *** NEW RECORD *** {previous:.4f} -> {best_val_loss:.4f} (Saved)")

                # C.2 Pas d'amélioration
                else:
                    stagnation_counter += 1
                    print(f"    [PATIENCE] {stagnation_counter}/{args.patience} (Best: {best_val_loss:.4f})")

                    # Early Stopping
                    if stagnation_counter >= args.patience:
                        print(f"\n[STOP] Early Stopping déclenché (Patience {args.patience} atteinte).")
                        wandb.log(metrics) # Log final avant de quitter
                        break

            else:
                print(f"    [WARNING] Val Loss invalide ({val_loss}). Ignoré.")

        # D. Backup Périodique
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"backup_ep{epoch}.pt"))

        # E. Envoi des métriques à WandB
        wandb.log(metrics)

    # 9. Fin du script
    total_duration_hours = (time.time() - total_start_time) / 3600
    wandb.run.summary["total_train_time_hours"] = total_duration_hours

    # Versionning du modèle final
    print(f"\n[WANDB] Upload du modèle final ({exp_name}) en cours...")

    # On cherche le bon modèle avec l'exp_name
    search_pattern = os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep*.pt")
    final_model_files = glob.glob(search_pattern)

    if final_model_files:
        model_path = final_model_files[0] # Normalement il n'y en a qu'un seul grâce au nettoyage
        print(f"[WANDB] Fichier trouvé : {os.path.basename(model_path)}")

        # Création et upload de l'artefact
        artifact_name = f"model-{wandb.run.id}"
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print("[WANDB] Upload terminé avec succès.")
    else:
        print(f"[WANDB] ERREUR : Aucun modèle trouvé pour le pattern {search_pattern}")

    wandb.finish()
    print(f"[FIN] Terminé en {total_duration_hours:.2f} heures. Best Loss: {best_val_loss}")



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
    parser.add_argument('--train_data', type=str, default="../../../output/final_data/train", 
                        help="Dossier contenant les fichiers H5 de train")
    parser.add_argument('--val_data', type=str, default="../../../output/final_data/val", 
                        help="Dossier contenant les fichiers H5 de validation")
    # Hyperparamètres
    parser.add_argument('--epochs', type=int, default=50, help="Nombre max d'époques")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate initial")
    parser.add_argument('--patience', type=int, default=10, help="Nb époques sans amélioration avant arrêt")

    # Arguments Système
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement")

    args = parser.parse_args()

    # Configuration PyTorch pour éviter la fragmentation mémoire CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    run(args)


if __name__ == "__main__":
    main()
