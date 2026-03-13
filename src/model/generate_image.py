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
    # Ajustement des dimensions initiales
    tracings = torch.transpose(tracings, 1, 2)
    batch_size, channels, time_steps = tracings.shape

    # Ajout de zeros pour completer la derniere fenetre si necessaire normalement impossible
    pad_len = (segment_size - (time_steps % segment_size)) % segment_size
    if pad_len > 0:
        tracings = F.pad(tracings, (0, pad_len), mode='constant', value=0.0)

    segments = tracings.unfold(2, segment_size, segment_size)
    segments_np = segments.numpy()
    batch_size, channels, num_segments, seq_len = segments_np.shape 

    # Filtrage median vectorise de la ligne de base
    kernel_size = int(seq_len * 0.05) | 1 
    baseline = scipy.ndimage.median_filter(segments_np, size=(1, 1, 1, kernel_size))
    sig_clean = segments_np - baseline

    # Mise a l'echelle dynamique automatique
    max_amp = np.max(np.abs(sig_clean), axis=(2, 3), keepdims=True)
    scale_y_dynamic = (h * 0.4) / (max_amp + 1e-6)

    # Anticipation des coordonnees avec une precision sous-pixel
    shift = 4
    mult = 16

    # Les abscisses restent identiques pour tous les signaux
    x_coords = np.linspace(0, (w - 1) * mult, seq_len).astype(np.int32)

    # Calcul des ordonnees
    y_coords_float = (h / 2.0) - (sig_clean * scale_y_dynamic)

    # Limitation des valeurs directement en memoire pour la performance
    np.clip(y_coords_float, 0, h - 1, out=y_coords_float) 
    y_coords = (y_coords_float * mult).astype(np.int32)

    # Reorganisation spatiale pour accelerer la lecture en boucle
    y_coords = np.transpose(y_coords, (0, 2, 1, 3))

    # Allocation memoire unique et prealable
    output_images_np = np.zeros((batch_size, num_segments, 12, h, w), dtype=np.float32)

    # Preparation du conteneur de points pour la bibliotheque de dessin
    pts = np.empty((seq_len, 1, 2), dtype=np.int32)
    pts[:, 0, 0] = x_coords

    # Preparation de la toile vierge
    img_channel = np.empty((h, w), dtype=np.float32)

    # Iterations de rasterisation optimisees
    for b in range(batch_size):
        for n in range(num_segments):
            for i in range(12):
                # Seules les ordonnees necessitent une mise a jour
                pts[:, 0, 1] = y_coords[b, n, i]

                # Remise a zero de la toile sans nouvelle allocation
                img_channel.fill(0)

                # Trace direct avec des aretes nettes sans anticrenelage pour un format binaire
                cv2.polylines(img_channel, [pts], isClosed=False, color=1.0, 
                              thickness=1, lineType=cv2.LINE_8, shift=shift)

                # Insertion du resultat dans le tenseur final
                output_images_np[b, n, i] = img_channel

    # Conversion differee vers le format tenseur final
    return torch.from_numpy(output_images_np)



if __name__ == "__main__":
    file_path = "../../../output/normalize_data/georgia.hdf5"

    with h5py.File(file_path, 'r') as f:
        tracings = torch.from_numpy(f['tracings'][:10]).permute(0, 2, 1)

    images_tensor = create_image_12leads_perchan(tracings, h=518, w=518, segment_size=4000)

    print(f"Format du tenseur genere : {images_tensor.shape}")

    flat_images = images_tensor.view(-1, 12, 518, 518)
    os.makedirs("../../../output/img/", exist_ok=True)

    for i in range(min(5, flat_images.size(0))):
        grid_input = flat_images[i].unsqueeze(1) 
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
