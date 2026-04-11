import os, sys
import argparse
import re
import time
import math
from tqdm import tqdm
import glob
import shutil

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser, build_model

import warnings



# On dit à Python tkt c'est pas grave pour ce warning précis
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

torch.set_float32_matmul_precision('high')  # Test d'optimisation


need_compile = set(['PatchTSTModel', 'DinoTraceTemporal', 'ViT_TimeFreq', 'ViT_Image'])



def generate_exp_name(args, valid_kwargs, wandb_id):
    pad_status = "UnivPad" if args.use_static_padding else "MaxPad"
    exp_parts = [args.model_name, pad_status]

    exp_parts.append(f"bs{args.batch_size_theoric}")
    exp_parts.append(f"lr{args.lr}")
    exp_parts.append(f"backbone_lr{args.backbone_lr}")

    amp_status = "AMP_F" if args.not_use_amp else "AMP_T"
    exp_parts.append(amp_status)

    # Simplifie le nommage
    abbrv = {
        'in_channels': 'in', 'num_classes': 'cls', 'ch1': 'c1', 'ch2': 'c2', 'ch3': 'c3', 
        'kernel_size': 'k', 'window_size1D': 'w1D', 'n_fft': 'fft', 'context_length': 'ctx',
        'patch_length': 'pt', 'd_model': 'dm', 'num_heads': 'hd', 'use_cross_att': 'cross',
        'win_length': 'wlen', 'hop_length': 'hlen', 'mlp_ratio': 'mr', 'patch_size': 'ps',
        'num_layers': 'nlayers'
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
    return "-".join(exp_parts)


def load_pos_weight(pos_weight_path, num_classes, device):
    """
    Loads the pre-computed pos_weight tensor and moves it to the target device.
    The tensor is expected to have been produced by compute_pos_weight.py and saved
    via torch.save(). Its shape must be (num_classes,).
    If the file does not exist, or the shape is wrong, a warning is printed and
    None is returned — the caller will then fall back to unweighted BCE loss.

    Args:
        pos_weight_path : Absolute or relative path to the .pt file.
        num_classes     : Expected number of classes (used for shape validation).
        device          : The device on which training runs.

    Returns:
        torch.Tensor of shape (num_classes,) on `device`, or None on failure.
    """
    if not pos_weight_path:
        print("[POS_WEIGHT] No path provided — using unweighted BCEWithLogitsLoss.")
        return None

    if not os.path.exists(pos_weight_path):
        print(f"[POS_WEIGHT] WARNING: File not found at '{pos_weight_path}'. "
              "Falling back to unweighted loss.")
        return None

    pw = torch.load(pos_weight_path, map_location=device)

    if pw.shape != (num_classes,):
        print(f"[POS_WEIGHT] WARNING: Shape mismatch — expected ({num_classes},), "
              f"got {tuple(pw.shape)}. Falling back to unweighted loss.")
        return None

    print(f"[POS_WEIGHT] Loaded from '{pos_weight_path}'")
    print(f"    Shape : {pw.shape}")
    print(f"    Min   : {pw.min():.4f}  |  Max : {pw.max():.4f}  |  Mean : {pw.mean():.4f}")

    pw = pw.sqrt()
    print(f"[POS_WEIGHT] After sqrt -> Min: {pw.min():.4f} | Max: {pw.max():.4f}")

    return pw.to(device)


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
    batch_size_theoric=64, 
    batch_size_accumulat=64,
):
    """
    Exécute une époque d'entraînement avec accumulation de gradients (Gradient Accumulation).

    Cette stratégie permet de simuler un batch_size important (batch_size_theoric) en 
    effectuant plusieurs passes avant/arrière sur des micro-batchs plus petits 
    (batch_size_accumulat) qui tiennent en VRAM. Les poids ne sont mis à jour qu'une 
    fois le batch théorique complété.

    Args:
        model (nn.Module): Le réseau de neurones à entraîner.
        dataloader (DataLoader): Le chargeur de données (renvoie tracings, targets, batch_mask).
        optimizer (torch.optim.Optimizer): L'optimiseur (ex: AdamW).
        criterion (nn.Module): La fonction de perte (ex: BCEWithLogitsLoss).
        scaler (torch.amp.GradScaler | None): Scaler pour la précision mixte (AMP).
        device (torch.device): Le périphérique de calcul (CUDA).
        epoch (int): Index de l'époque actuelle.
        total_epochs (int): Nombre total d'époques.
        use_amp (bool): Activation de l'Automatic Mixed Precision.
        batch_size_theoric (int): Taille de batch effective souhaitée pour l'optimisation.
        batch_size_accumulat (int): Taille de batch physique réellement envoyée au GPU.

    Returns:
        float: La perte moyenne de l'époque.
    """
    model.train()

    # Calcul du nombre d'itérations nécessaires pour atteindre le batch théorique
    accumulation_steps = max(1, batch_size_theoric // batch_size_accumulat)

    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]", dynamic_ncols=False)
    total_loss = 0
    count = 0

    # Initialisation propre des gradients
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loop):
        tracings, targets, batch_mask = batch
        tracings = tracings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

        # 1. Forward Pass avec AMP
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            predictions = model(tracings, batch_mask=batch_mask)

            # Important : On divise la loss par accumulation_steps car backward() 
            # additionne les gradients. Cette division garantit que le gradient 
            # final correspond à la moyenne sur batch_size_theoric
            loss = criterion(predictions, targets) / accumulation_steps

        # 2. Backward Pass (Accumulation automatique dans les buffers de gradients)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 3. Step d'optimisation (Uniquement quand on a accumulé assez de gradients)
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Reset des gradients pour le prochain cycle d'accumulation
            optimizer.zero_grad(set_to_none=True)

        # 4. Logging (On affiche la loss réelle, donc multipliée par accumulation_steps)
        loss_val = loss.item() * accumulation_steps

        if math.isfinite(loss_val):
            total_loss += loss_val
            count += 1
            loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")
        else:
            # Diagnostic en cas de divergence (NaN/Inf)
            try:
                crash_table = wandb.Table(
                    columns=["Epoch", "Batch", "Loss", "Targets Sample"],
                    data=[[epoch, count, loss_val, str(targets[0].tolist())]]
                )
                wandb.log({"errors/train_crash_report": crash_table})
            except Exception:
                pass

    return total_loss / count if count > 0 else float('inf')



def validate(model, dataloader, device, use_amp, dtype, epoch, pr_auc_metric):
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
    pr_auc_metric.reset()

    with torch.no_grad():
        for batch in loop:
            tracings, targets, batch_mask = batch
            tracings = tracings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=dtype):
                logits = model(tracings, batch_mask=batch_mask)

                if not torch.isfinite(logits).all():
                    print(f"\n    [WARNING] NaNs ou Infs détectés dans les prédictions (Ep {epoch}).")
                    wandb.log({"errors/val_nan_warning": epoch})
                    # On renvoie 0.0 car c'est le pire PR-AUC possible
                    return 0.0 

                probs = torch.sigmoid(logits)

            pr_auc_metric.update(probs, targets.long())

    # Calcul final de la métrique
    val_pr_auc = pr_auc_metric.compute().item()

    # Vérifie que le calcul de la métrique est bon
    if math.isfinite(val_pr_auc):
        return val_pr_auc
    else:
        print(f"\n    [WARNING] Le PR-AUC calculé est invalide: {val_pr_auc}. Ignoré.")
        return 0.0



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

    # Si reprise WandB, on tente de récupérer le meilleur score connu
    if wandb.run.summary.get("best_val_pr_auc"):
        saved_best = wandb.run.summary["best_val_pr_auc"]
        if isinstance(saved_best, (int, float)) and math.isfinite(saved_best):
            best_val_pr_auc = saved_best
            print(f"[RESUME] Best score précédent récupéré de WandB: {best_val_pr_auc}")

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

    pr_auc_metric = MultilabelAveragePrecision(num_labels=args.num_classes, average='macro').to(device)

    # 8.Boucle des epochs
    print(f"\n[TRAIN] Démarrage : Epoch {start_epoch} -> {args.epochs}")
    stagnation_counter = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # A. Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp, amp_dtype, args.batch_size_theoric, args.batch_size_accumulat
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
        if epoch >= args.val_start_epoch:
            pr_auc = validate(model, val_loader, device, use_amp, amp_dtype, epoch, pr_auc_metric)
            metrics["val/pr_auc"] = pr_auc

            # C. Si la val a fonctionnée
            if math.isfinite(pr_auc):
                # C.1 Nouveau Record
                if pr_auc > best_val_pr_auc:
                    previous = best_val_pr_auc
                    best_val_pr_auc = pr_auc
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
                    wandb.run.summary["best_val_pr_auc"] = best_val_pr_auc
                    wandb.run.summary["best_epoch"] = epoch

                    print(f"    *** NEW RECORD *** {previous:.4f} -> {best_val_pr_auc:.4f} (Saved)")

                # C.2 Pas d'amélioration
                else:
                    stagnation_counter += 1
                    print(f"    [PATIENCE] {stagnation_counter}/{args.patience} (Best: {best_val_pr_auc:.4f})")

                    # Early Stopping
                    if stagnation_counter >= args.patience:
                        print(f"\n[STOP] Early Stopping déclenché (Patience {args.patience} atteinte).")
                        wandb.log(metrics) # Log final avant de quitter
                        break

            else:
                print(f"    [WARNING] Val pr_auc invalide ({pr_auc}). Ignoré.")

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
        model_path = final_model_files[0] 
        nom_fichier = os.path.basename(model_path)
        print(f"[WANDB] Fichier trouvé sur le cluster : {nom_fichier}")

        # 1. On copie le modèle DANS le dossier interne du run W&B
        wandb_internal_path = os.path.join(wandb.run.dir, nom_fichier)
        shutil.copy2(model_path, wandb_internal_path)

        # 2. Création et upload de l'artefact à partir de cette copie interne
        artifact_name = f"model-{model_name}-{wandb.run.id}"
        artifact = wandb.Artifact(artifact_name, type='model')

        # On ajoute le fichier interne, avec son nom propre
        artifact.add_file(wandb_internal_path, name=nom_fichier)
        wandb.log_artifact(artifact)

        print("[WANDB] Upload préparé avec succès pour la synchro locale.")
    else:
        print(f"[WANDB] ERREUR : Aucun modèle trouvé pour le pattern {search_pattern}")

    wandb.finish()
    print(f"[FIN] Terminé en {total_duration_hours:.2f} heures. Best PR-AUC: {best_val_pr_auc}")



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
    parser.add_argument('--epochs', type=int, default=80, help="Nombre max d'époques")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate initial")
    parser.add_argument('--backbone_lr', type=float, default=1e-6, help="Learning Rate initial for the backbone")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="pénaliter pour la régularisation du model")
    parser.add_argument('--patience', type=int, default=10, help="Nb époques sans amélioration avant arrêt")
    parser.add_argument('--val_start_epoch', type=int, default=15, help="epoch ou commencer la validation")

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

    args = parser.parse_args()

    # Configuration PyTorch pour éviter la fragmentation mémoire CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    run(args)


if __name__ == "__main__":
    main()
