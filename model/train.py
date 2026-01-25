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

from model import Cnn
from dataset import *



# Force matplotlib à ne pas chercher d'écran (utile sur serveur)
plt.switch_backend('agg')


def save_monitoring(history, checkpoint_dir):
    """
    Sauvegarde l'historique en JSON et trace les courbes de Loss.
    Permet de vérifier visuellement s'il y a de l'Overfitting (si Val remonte).
    """
    # 1. Sauvegarde JSON
    json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 2. Tracé du Graphique
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='#1f77b4') # Bleu

    # On filtre les valeurs > 0 pour ne pas tracer les 0.0
    val_epochs = [e for e, v in zip(epochs, val_loss) if v > 0]
    val_values = [v for v in val_loss if v > 0]

    if val_values:
        plt.plot(val_epochs, val_values, marker='x', linestyle='--', label='Val Loss', color='#ff7f0e') # Orange
    
    plt.title("Convergence : Train vs Validation")
    plt.xlabel("Époques")
    plt.ylabel("Loss (Masked LogMSE)")
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
        # --- A. Chargement GPU ---
        time_serries = batch['traicing'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)

        # --- B. Forward & Backward ---
        optimizer.zero_grad(set_to_none=True) # Reset gradients

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            prediction = model(
                images=imgs, 
                image_masks=img_masks, 
                input_ids=input_ids, 
                text_mask=text_mask, 
                x_cont=x_cont, 
                x_cat=x_cat
            )
            loss = criterion(prediction, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        # --- C. Stats ---
        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device, use_amp):
    model.eval()

    loop = tqdm(dataloader, desc="[VAL]")
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loop:
            imgs = batch['images'].to(device, non_blocking=True)
            img_masks = batch['image_masks'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_mask'].to(device, non_blocking=True)
            x_cont = batch['x_cont'].to(device, non_blocking=True)
            x_cat = batch['x_cat'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
                p_vente, p_loc = model(imgs, img_masks, input_ids, text_mask, x_cont, x_cat)
                loss = criterion(p_vente, p_loc, targets, masks)

            total_loss += loss.item()
            count += 1
            loop.set_postfix(val_loss=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0



def main():
    parser = argparse.ArgumentParser()
    # Dossiers séparés physiquement
    parser.add_argument('--train_data', type=str, default='../output/train')
    parser.add_argument('--val_data', type=str, default='../output/val')

    # Paramètres d'entraînement
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)

    # --- Argument pour reprendre l'entraînement ---
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement (ex: checkpoints/model_ep15.pt)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="Nombre de fois ou on autorise la stagnation")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Détection Hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
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
    print(f"[DATA] Chargement Train : {args.train_csv}")
    train_ds = LargeH5Dataset(
        df_or_folder=args.train_data, img_dir=args.img_dir, tokenizer=tokenizer,
        scaler=scaler_obj, medians=medians, modes=modes, cat_mappings=cat_mappings,
        cont_cols=cont_cols, cat_cols=cat_cols
    )

    print(f"[DATA] Chargement Validation : {args.val_csv}")
    val_ds = LargeH5Dataset(
        df_or_folder=args.val_data, img_dir=args.img_dir
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count()-1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count()-1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  
    )

    # 4. Initialisation du Modèle
    print(f"[MODEL] Init SOTA... (Tab Continuous: {len(cont_cols)}, Tab Categorical: {len(cat_dims)})")
    model = Cnn(
        num_continuous=len(cont_cols),
        cat_cardinalities=cat_dims,
        img_model_name='convnext_large.fb_in1k', 
        text_model_name='almanach/camembert-base',
        fusion_dim=512,
        depth=4,
        freeze_encoders=True 
    ).to(device)

    start_epoch = 1

    # Reprise depuis chekpoint
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"[RESUME] Chargement du checkpoint : {args.resume_from}")
            # Chargement des poids
            checkpoint = torch.load(args.resume_from, map_location=device)

            # Nettoyage des clés si le modèle avait été compilé (_orig_mod.)
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)

            # Tentative de déduction de l'époque de départ via le nom du fichier
            # Ex: model_ep15.pt -> start_epoch = 16
            match = re.search(r'ep(\d+)', args.resume_from)
            if match:
                last_epoch = int(match.group(1))
                start_epoch = last_epoch + 1
                print(f"[RESUME] Reprise à l'époque {start_epoch}")
            else:
                print("[RESUME] Impossible de lire l'époque dans le nom du fichier. Reprise à l'époque 1 par défaut.")
        else:
            print(f"[ERREUR] Le fichier checkpoint {args.resume_from} n'existe pas. Démarrage à zéro.")


    # Compilation PyTorch 2.0
    if torch.cuda.is_available():
        try: 
            model = torch.compile(model)
            print("[OPTIM] Torch.compile activé.")
        except: 
            pass

    # Optimiseur & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # 6. Boucle Principale
    # Si on reprend, on charge l'historique existant si possible, sinon on repart à zéro
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

        val_loss = None 

        if epoch > 10:
            # B. Phase de Validation
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
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_model_ep{epoch}.pt"))
                print(f"    *** NEW RECORD! best_model.pt sauvegardé (Val: {val_loss:.4f}) ***")

                for f in glob.glob(os.path.join(args.checkpoint_dir, "best_model_*.pt")):
                    os.remove(f)

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


if __name__ == "__main__":
    main()
