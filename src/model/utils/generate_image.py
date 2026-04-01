import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import torchvision.utils as vutils



# --- Palette de 12 couleurs très distinctes (Format RGB) ---
distinct_colors = [
    (255, 0, 0),      # 1. Rouge
    (0, 0, 255),      # 2. Bleu profond
    (0, 150, 0),      # 3. Vert foncé
    (255, 120, 0),    # 4. Orange
    (150, 0, 150),    # 5. Violet
    (0, 200, 200),    # 6. Cyan (Turquoise)
    (139, 69, 19),    # 7. Marron (SaddleBrown)
    (255, 0, 255),    # 8. Magenta
    (100, 200, 0),    # 9. Vert Lime
    (0, 100, 100),    # 10. Sarcelle (Teal)
    (255, 105, 180),  # 11. Rose (Hot Pink)
    (0, 0, 128)       # 12. Bleu Marine
]


def create_image_12leads_together(tracings, h=512, w=512, segment_size=1000, scale_y=2.8, rgb=True):
    """
    Génération pour ECG avec support optionnel des couleurs.
    
    Args:
        tracings: Tenseur des signaux bruts.
        rgb (bool): Si True, chaque lead a une couleur unique (3 canaux). 
                    Si False, tout est noir (1 canal étendu).
    """
    B, C, T = tracings.shape
    segments = tracings.unfold(2, segment_size, segment_size)
    B, C, S, L = segments.shape

    offsets = np.linspace(h * 0.05, h * 0.95, 12).astype(np.int32)
    x_coords = np.linspace(0, w - 1, L).astype(np.int32)

    # Allocation de la mémoire selon le mode
    num_channels = 3 if rgb else 1
    output_np = np.full((B, S, h, w, num_channels), 255, dtype=np.uint8)

    segments_np = segments.numpy()

    for b in range(B):
        for n in range(S):
            img_slice = output_np[b, n]
            for i in range(12):
                sig = segments_np[b, i, n, :]
                y_coords = (offsets[i] - sig * scale_y).astype(np.int32)
                np.clip(y_coords, 0, h - 1, out=y_coords)

                pts = np.column_stack((x_coords, y_coords)).reshape((-1, 1, 2))

                # Sélection de la couleur selon le mode
                color = distinct_colors[i] if rgb else 0

                # Épaisseur 2 et Anti-Aliasing
                cv2.polylines(img_slice, [pts], False, color, 2, cv2.LINE_AA)

    # Conversion en tenseur Torch [B, S, C, H, W]
    final_tensor = torch.from_numpy(output_np).permute(0, 1, 4, 2, 3).contiguous()

    # Si mode N&B, on simule les 3 canaux virtuellement
    if not rgb:
        final_tensor = final_tensor.expand(-1, -1, 3, -1, -1)

    return final_tensor # Retourne du uint8


def create_image_12leads_perchan(tracings, h=512, w=512, segment_size=4000):
    """
    Genere un tenseur d'images optimise ou chaque derivation possede son propre canal.

    Le trace est calcule avec une precision sous-pixel pour eviter les cassures temporelles.

    Args:
        tracings (torch.Tensor): Tenseur des signaux de forme initiale.
        h (int, optional): Hauteur de chaque canal visuel en pixels.
        w (int, optional): Largeur de chaque canal visuel en pixels.
        segment_size (int, optional): Nombre de points temporels par fenetre.

    Returns:
        torch.Tensor: Tenseur optimise contenant douze canaux separes par fenetre.
    """
    # 1. Mise en forme des signaux
    batch_size, channels, time_steps = tracings.shape

    # Padding temporel si le signal ne tombe pas juste
    pad_len = (segment_size - (time_steps % segment_size)) % segment_size
    if pad_len > 0:
        # Pad la dernière dimension (Time)
        tracings = F.pad(tracings, (0, pad_len), mode='constant', value=0.0)

    # Fenêtrage (unfold) sur l'axe du temps (dim=2) -> Shape: (B, 12, N, S)
    segments = tracings.unfold(2, segment_size, segment_size)

    segments_np = segments.transpose(1, 2).numpy()

    batch_size, num_segments, channels, seq_len = segments_np.shape

    # 2. Filtrage moy de 5% forcer à etre pair avec | 1
    #kernel_size = int(seq_len * 0.05) | 1
    # Le filtre s'applique toujours sur la dernière dimension (seq_len)
    #baseline = scipy.ndimage.uniform_filter1d(segments_np, size=kernel_size, axis=-1)
    sig_clean = segments_np #- baseline

    # 3. Auto-Scale dynamique
    # On calcule le max sur les segments (axe 1) et le temps (axe 3) 
    # pour avoir 1 scale par canal pour tout le patient. -> Shape: (B, 1, 12, 1)
    max_amp = np.max(np.abs(sig_clean), axis=(1, 3), keepdims=True)
    scale_y_dynamic = (h * 0.4) / (max_amp + 1e-6)

    # 4. Pré-calcul vectorisé des coordonnées
    shift = 4
    mult = 16

    # Calcul des Y : Broadcasting direct sans aucune boucle
    y_coords_float = (h / 2.0) - (sig_clean * scale_y_dynamic)
    np.clip(y_coords_float, 0, h - 1, out=y_coords_float) 
    y_coords = (y_coords_float * mult).astype(np.int32) # Shape: (B, N, 12, S)

    # --- Vectorisation "Canvas Empilé" ---
    total_images = batch_size * num_segments

    # On fusionne Batch et Segments: (Total_images, 12, S)
    y_coords_flat = y_coords.reshape(total_images, 12, seq_len)

    # Décalage vertical mathématique pour empiler les 12 dérivations
    offsets = (np.arange(12) * h * mult).reshape(1, 12, 1)
    y_coords_offset = y_coords_flat + offsets

    # Préparation des X (identiques partout)
    x_coords = np.linspace(0, (w - 1) * mult, seq_len).astype(np.int32)

    # Structure attendue par OpenCV : (Total_images, 12 courbes, seq_len, 1 point, 2 coords)
    pts_all = np.empty((total_images, 12, seq_len, 1, 2), dtype=np.int32)
    pts_all[..., 0, 0] = x_coords
    pts_all[..., 0, 1] = y_coords_offset

    # 5. Allocation et Dessin
    output_images_np = np.zeros((batch_size, num_segments, 12, h, w), dtype=np.uint8)

    # On crée une "fenêtre virtuelle" qui fait 12 fois la hauteur
    canvas_view = output_images_np.reshape(total_images, 12 * h, w)

    for j in range(total_images):
        cv2.polylines(canvas_view[j], list(pts_all[j]), isClosed=False, color=255, 
                      thickness=1, lineType=cv2.LINE_8, shift=shift)

    return torch.from_numpy(output_images_np)



if __name__ == "__main__":
    file_path = "../../../../output/normalize_data/georgia.hdf5"
    output_path = "../../../../output/img/"
    h = 512
    w = 512

    os.makedirs(output_path, exist_ok=True)

    with h5py.File(file_path, 'r') as f:
        tracings = torch.from_numpy(f['tracings'][:10])

    """
    images_tensor = create_image_12leads_perchan(tracings, h=518, w=518, segment_size=4000)

    print(f"Format du tenseur genere : {images_tensor.shape}")

    flat_images = images_tensor.view(-1, 12, 518, 518)

    for i in range(min(5, flat_images.size(0))):
        grid_input = flat_images[i].unsqueeze(1).float() / 255.0

        vutils.save_image(grid_input, f"../../../../output/img/check_dino_12channels_{i}.png", nrow=4, normalize=False)

    """
    images_tensor = create_image_12leads_together(tracings, h, w)

    # Fusion des dimensions du lot et des segments pour l'exportation
    # Regroupement sequentiel des images generees
    flat_images = images_tensor.view(-1, 3, h, w)

    # Enregistrement d'un echantillon pour le controle visuel
    for i in range(min(10, flat_images.size(0))):
        img_float = flat_images[i].to(torch.float32) / 255.0
        # Enregistrement direct au format image standard
        vutils.save_image(img_float, f"{output_path}/check_dino_1channels_{i}.png")

    print(f"Verification terminee : echantillons sauvegardes depuis le tenseur de taille {images_tensor.shape}")

    print("Generation terminee.")
