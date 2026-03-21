import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import torchvision.utils as vutils
import scipy.ndimage
import scipy.signal



def create_image_12leads_together(tracings, h=518, w=518, segment_size=4000, scale_y=13.0):
    """
    Genere un tenseur visuel ou les douze derivations sont tracees sur une image unique.

    Les signaux sont decoupes en fenetres temporelles. Pour chaque fenetre, 
    les derivations sont filtrees puis empilees verticalement sur un fond blanc 
    a l'aide de lignes de reference horizontales.

    Args:
        tracings (torch.Tensor): Tenseur des signaux bruts.
        h (int, optional): Hauteur de l'image de sortie en pixels.
        w (int, optional): Largeur de l'image de sortie en pixels.
        segment_size (int, optional): Nombre de points de mesure par image.
        scale_y (float, optional): Facteur d'amplification verticale du trace.

    Returns:
        torch.Tensor: Tenseur des images generees au format couleurs classiques.
    """
    tracings = torch.transpose(tracings, 1, 2)
    batch_size, channels, time_steps = tracings.shape

    step = segment_size
    segments = tracings.unfold(2, segment_size, step)
    batch_size, channels, num_segments, seq_len = segments.shape

    output_images = torch.empty((batch_size, num_segments, 3, h, w), dtype=torch.float32)
    offsets = np.linspace(h * 0.05, h * 0.95, 12).astype(int)

    for b in range(batch_size):
        for n in range(num_segments):
            img = np.full((h, w, 3), 255, dtype=np.uint8)

            for i in range(12):
                # Extraction du signal local
                sig_raw = segments[b, i, n, :].numpy()

                # Etape de nettoyage du signal
                # Estimation de la ligne de base par filtre median
                # La taille du noyau doit rester un nombre impair, fixee a cinq pour cent de la longueur
                kernel_size = int(seq_len * 0.05)
                if kernel_size % 2 == 0: 
                    kernel_size += 1 

                # Isolement de l'oscillation lente
                baseline = scipy.signal.medfilt(sig_raw, kernel_size)

                # Soustraction pour aplanir le trace
                sig_clean = sig_raw - baseline

                # Trace de la ligne de reference en gris
                #cv2.line(img, (0, offsets[i]), (w, offsets[i]), (230, 230, 230), 1)

                x_coords = np.linspace(0, w - 1, seq_len).astype(int)

                # Calcul des coordonnees avec le signal aplani
                y_coords = (offsets[i] - sig_clean * scale_y).astype(int)
                y_coords = np.clip(y_coords, 0, h - 1)

                pts = np.vstack((x_coords, y_coords)).T.reshape((-1, 1, 2))

                # Lissage du trait
                cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            output_images[b, n] = img_tensor

    return output_images


def create_image_12leads_perchan(tracings, h=518, w=518, segment_size=4000):
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
    kernel_size = int(seq_len * 0.05) | 1
    # Le filtre s'applique toujours sur la dernière dimension (seq_len)
    baseline = scipy.ndimage.uniform_filter1d(segments_np, size=kernel_size, axis=-1)
    sig_clean = segments_np - baseline

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
    file_path = "../../../output/normalize_data/georgia.hdf5"

    with h5py.File(file_path, 'r') as f:
        tracings = torch.from_numpy(f['tracings'][:10])

    images_tensor = create_image_12leads_perchan(tracings, h=518, w=518, segment_size=4000)

    print(f"Format du tenseur genere : {images_tensor.shape}")

    flat_images = images_tensor.view(-1, 12, 518, 518)
    os.makedirs("../../../output/img/", exist_ok=True)

    for i in range(min(5, flat_images.size(0))):
        grid_input = flat_images[i].unsqueeze(1).float() / 255.0

        vutils.save_image(grid_input, f"../../../output/img/check_dino_12channels_{i}.png", nrow=4, normalize=False)

    """
    images_tensor = create_image_12leads_together(tracings, h=518, w=518)

    # Fusion des dimensions du lot et des segments pour l'exportation
    # Regroupement sequentiel des images generees
    flat_images = images_tensor.view(-1, 3, 518, 518)

    # Enregistrement d'un echantillon pour le controle visuel
    for i in range(min(10, flat_images.size(0))):
        # Enregistrement direct au format image standard
        vutils.save_image(flat_images[i], f"check_dino_input_{i}.png")

    print(f"Verification terminee : echantillons sauvegardes depuis le tenseur de taille {images_tensor.shape}")
    """

    print("Generation terminee.")
