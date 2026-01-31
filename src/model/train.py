import os
import json
import argparse
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

from torch.utils.data import DataLoader


from dataset import LargeH5Dataset 
from model import CNN 



# Force matplotlib à ne pas chercher d'écran
plt.switch_backend('agg')


def save_monitoring(history, checkpoint_dir):
    """ 
        Sauvegarde les métriques d'entraînement et génère un graphique de convergence.
        
        Args:
            history (dict): Dictionnaire contenant 'epoch', 'train_loss', 'val_loss'.
            checkpoint_dir (str): Dossier de sortie.
        """
    json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='#1f77b4')

    val_epochs = [e for e, v in zip(epochs, val_loss) if v > 0]
    val_values = [v for v in val_loss if v > 0]

    if val_values:
        plt.plot(val_epochs, val_values, marker='x', linestyle='--', label='Val Loss', color='#ff7f0e')
    
    plt.title("Convergence : Train vs Validation")
    plt.xlabel("Époques")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "convergence_plot.png"))
    plt.close()


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, total_epochs, use_amp):
    """
    Exécute une itération complète d'entraînement sur le dataset (une époque).

    Gère le passage avant (forward), le calcul de la loss, la rétropropagation (backward)
    et la mise à jour des poids. Intègre la gestion de la précision mixte (AMP).

    Args:
        model (nn.Module): Le modèle à entraîner.
        dataloader (DataLoader): Le chargeur de données d'entraînement.
        optimizer (Optimizer): L'optimiseur (ex: AdamW).
        criterion (nn.Module): La fonction de coût (ex: BCEWithLogitsLoss).
        scaler (torch.amp.GradScaler | None): Scaler pour la gestion des gradients en FP16 (si AMP activé).
        device (torch.device): CPU, CUDA ou MPS.
        epoch (int): Numéro de l'époque courante (pour l'affichage).
        total_epochs (int): Nombre total d'époques prévues.
        use_amp (bool): Si True, active le contexte d'autocast.

    Returns:
        float: La perte (loss) moyenne sur l'ensemble de l'époque.
    """
    model.train()
    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]")
    total_loss = 0
    count = 0

    for batch in loop:
        tracings, targets = batch
        tracings = tracings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            predictions = model(tracings)
            loss = criterion(predictions, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        count += 1
        loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device, use_amp):
    """
    Évalue le modèle sur le jeu de validation.

    Désactive le calcul des gradients (torch.no_grad) et passe le modèle en mode évaluation
    (fige le Dropout et la BatchNorm).

    Args:
        model (nn.Module): Le modèle à évaluer.
        dataloader (DataLoader): Le chargeur de données de validation.
        criterion (nn.Module): La fonction de coût.
        device (torch.device): CPU, CUDA ou MPS.
        use_amp (bool): Si True, utilise l'autocast pour l'inférence (gain de vitesse).

    Returns:
        float: La perte (loss) moyenne sur le jeu de validation.
    """
    model.eval()
    loop = tqdm(dataloader, desc="[VAL]")
    total_loss = 0
    count = 0

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
    Pilote l'intégralité du pipeline d'entraînement : configuration, chargement, boucle d'époques et sauvegarde.

    Cette fonction orchestre les étapes suivantes :
    1.  Configuration : Détecte le matériel (CPU/CUDA/MPS) et configure le Mixed Precision (AMP).
    2.  Données : Instancie les Datasets (Train/Val) et les DataLoaders optimisés.
    3.  Modèle : Initialise le modèle et charge les poids existants si une reprise est demandée (`resume_from`).
    4.  Boucle d'entraînement :
        - Exécute les époques d'entraînement et de validation.
        - Gère l'historique des pertes (Loss).
        - Applique l'Early Stopping si la validation stagne.
        - Sauvegarde le meilleur modèle (`best_model_*.pt`) et les checkpoints réguliers.

    Args:
        args (argparse.Namespace): Objet contenant les hyperparamètres et configurations :
            - args.class_map (str) : Chemin du JSON des classes cibles.
            - args.train_data (str) : Chemin du dataset d'entraînement.
            - args.val_data (str) : Chemin du dataset de validation.
            - args.checkpoint_dir (str) : Dossier de sortie.
            - args.resume_from (str, optional) : Chemin d'un checkpoint pour reprise.
            - args.batch_size (int) : Taille du batch.
            - args.lr (float) : Learning rate.
            - args.epochs (int) : Nombre max d'époques.
            - args.patience (int) : Seuil de patience pour l'Early Stopping.

    Returns:
        None: La fonction écrit les logs et les fichiers modèles sur le disque.
    """
    # 1. Chargement de la liste des classes
    print(f"[INIT] Chargement des classes depuis : {args.class_map}")
    with open(args.class_map, 'r') as f:
        loaded_classes = json.load(f)

    num_classes = len(loaded_classes)
    print(f"[INIT] {num_classes} classes détectées : {loaded_classes[:5]} ...")

    # 2. Setup Hardware
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        torch.backends.cudnn.benchmark = True 
        print("[INIT] Mode: NVIDIA CUDA (AMP On)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False 
        print("[INIT] Mode: APPLE METAL (MPS)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU")

    # 3. Datasets & Loaders
    # On passe la liste chargée dynamiquement
    print(f"[DATA] Chargement Train depuis : {args.train_data}")
    train_ds = LargeH5Dataset(input_dir=args.train_data, classes_list=loaded_classes)

    print(f"[DATA] Chargement Validation depuis : {args.val_data}")
    val_ds = LargeH5Dataset(input_dir=args.val_data, classes_list=loaded_classes)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  
    )

    # 4. Modèle
    print(f"[MODEL] Init du modèle...")
    model = CNN(num_classes=num_classes).to(device)
    torch.compile(model)

    # 5. Reprise de train
    start_epoch = 1
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"[RESUME] Chargement du checkpoint : {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            try:
                model.load_state_dict(new_state_dict)
            except RuntimeError as e:
                print(f"[WARNING] Mismatch poids : {e}")

            match = re.search(r'ep(\d+)', args.resume_from)
            if match:
                start_epoch = int(match.group(1)) + 1
                print(f"[RESUME] Reprise à l'époque {start_epoch}")
        else:
            print(f"[ERREUR] Checkpoint introuvable : {args.resume_from}")

    # 6. Optimisation
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # 7. Reprise d'entrainement
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    if args.resume_from and os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print("[RESUME] Historique chargé.")
        except:
            history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    else:
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        
    best_val_loss = float('inf')

    # Si l'historique existe, on récupère le meilleur val_loss connu pour ne pas le perdre
    val_losses_clean = [v for v in history.get('val_loss', []) if v > 0]
    if val_losses_clean:
        best_val_loss = min(val_losses_clean)
        print(f"[RESUME] Record Val Loss actuel : {best_val_loss:.4f}")

    print(f"\n[TRAIN] Démarrage Ep {start_epoch} -> {args.epochs}")

    stagnation_counter = 0

    # On commence la boucle à start_epoch
    for epoch in range(start_epoch, args.epochs+1):
        # A. Phase d'Apprentissage
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )

        # B. Phase de Validation (Conditionnelle : Seulement > 10)
        val_loss = None 

        if epoch > 10:
            val_loss = validate(model, val_loader, criterion, device, use_amp)
            print(f" -> Ep {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")
        else:
            print(f" -> Ep {epoch}: Train={train_loss:.4f} | Val=(Skipped)")

        # C. Logs
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else 0.0)
        save_monitoring(history, args.checkpoint_dir)

        # D. Checkpointing Intelligent
        if epoch > 10:
            # Si on s'améliore
            if (val_loss < best_val_loss):
                stagnation_counter = 0
                best_val_loss = val_loss

                # Supprime l'ancien modèle moins bon
                for f in glob.glob(os.path.join(args.checkpoint_dir, "best_model_*.pt")):
                    os.remove(f)

                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_model_ep{epoch}.pt"))
                print(f"    *** NEW RECORD! best_model.pt sauvegardé (Val: {val_loss:.4f}) ***")

            # Cas : On ne s'améliore pas
            else:
                stagnation_counter += 1
                print(f"    [PATIENCE] Pas d'amélioration ({stagnation_counter}/{args.patience})")

                # ARRÊT PRÉCOCE
                if stagnation_counter >= args.patience:
                    print(f"\n[STOP] Early Stopping déclenché ! Pas de progrès depuis {args.patience} époques.")
                    print(f"       Meilleur score final : {best_val_loss:.4f}")
                    break # On sort de la boucle 'for', fin de l'entraînement

        elif epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_ep{epoch}.pt"))
            print(f"    [BACKUP] model_ep{epoch}.pt sauvegardé.")

    print("[FIN] Entraînement terminé.")


def main():
    parser = argparse.ArgumentParser()

    # --- Arguments Données ---
    parser.add_argument('--train_data', type=str, default="../output/final_data/train", help="Dossier contenant les shards TRAIN")
    parser.add_argument('--val_data', type=str, default="../output/final_data/val", help="Dossier contenant les shards VAL")

    parser.add_argument('--class_map', type=str, default='../../ressources/final_class.json',
                        help="Chemin vers le fichier JSON contenant la liste ordonnée des classes")

    # --- Arguments Entraînement ---
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count()-2)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    run(args)



if __name__ == "__main__":
    main()
