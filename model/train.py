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
import sys

from torch.utils.data import DataLoader

# Gestion des imports relatifs
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(project_root))

from src.dataset import LargeH5Dataset 
from src.model import Cnn 



# Force matplotlib à ne pas chercher d'écran
plt.switch_backend('agg')


def save_monitoring(history, checkpoint_dir):
    """ Sauvegarde l'historique en JSON et trace les courbes. """
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
        shuffle=True, 
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
    model = Cnn(num_classes=num_classes).to(device)

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

    # 7. Historique
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    if args.resume_from and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        
    best_val_loss = float('inf')
    valid_losses = [v for v in history.get('val_loss', []) if v > 0]
    if valid_losses:
        best_val_loss = min(valid_losses)

    # 8. Boucle d'entraînement
    print(f"\n[TRAIN] Démarrage... (Batch: {args.batch_size})")
    stagnation_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )

        val_loss = validate(model, val_loader, criterion, device, use_amp)
        print(f" -> Ep {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        save_monitoring(history, args.checkpoint_dir)

        # Sauvegarde
        if val_loss < best_val_loss:
            stagnation_counter = 0
            best_val_loss = val_loss
            
            for f in glob.glob(os.path.join(args.checkpoint_dir, "best_model_*.pt")):
                try: os.remove(f)
                except: pass
            
            save_path = os.path.join(args.checkpoint_dir, f"best_model_ep{epoch}_loss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    *** NEW RECORD! Sauvegardé : {os.path.basename(save_path)} ***")
        else:
            stagnation_counter += 1
            print(f"    [PATIENCE] {stagnation_counter}/{args.patience}")

        if stagnation_counter >= args.patience:
            print("[STOP] Early Stopping.")
            break

        if epoch % 5 == 0:
             torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "last_checkpoint.pt"))

    print("[FIN] Entraînement terminé.")


def main():
    parser = argparse.ArgumentParser()

    # --- Arguments Données ---
    parser.add_argument('--train_data', type=str, required=True, help="Dossier contenant les shards TRAIN")
    parser.add_argument('--val_data', type=str, required=True, help="Dossier contenant les shards VAL")
    
    parser.add_argument('--class_map', type=str, default='../resources/final_class.json',
                        help="Chemin vers le fichier JSON contenant la liste ordonnée des classes")

    # --- Arguments Entraînement ---
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    run(args)



if __name__ == "__main__":
    main()
