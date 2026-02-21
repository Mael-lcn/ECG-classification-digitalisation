import numpy as np
import torch
import cv2
import h5py
import torchvision.utils as vutils
import scipy.signal



def create_image_12leads_optimized_cleaned(tracings, h=518, w=518, segment_size=4096, scale_y=16.0):
    B, C, T = tracings.shape

    step = segment_size
    segments = tracings.unfold(2, segment_size, step)
    B, C, N, S = segments.shape

    output_images = torch.empty((B, N, 3, h, w), dtype=torch.float32)
    offsets = np.linspace(h * 0.05, h * 0.95, 12).astype(int)

    for b in range(B):
        for n in range(N):
            img = np.full((h, w, 3), 255, dtype=np.uint8)

            for i in range(12):
                # 1. Récupération du signal brut
                sig_raw = segments[b, i, n, :].numpy()

                # --- ÉTAPE DE NETTOYAGE ---
                # On estime la ligne de base avec un filtre médian large.
                # kernel_size doit être impair. On prend environ 5% de la longueur du signal.
                kernel_size = int(S * 0.05)
                if kernel_size % 2 == 0: kernel_size += 1 # S'assurer qu'il est impair

                # baseline est la "vague" lente
                baseline = scipy.signal.medfilt(sig_raw, kernel_size)

                # On soustrait la vague pour avoir un signal plat (sig_clean)
                sig_clean = sig_raw - baseline

                # Ligne 0 gris clair
                cv2.line(img, (0, offsets[i]), (w, offsets[i]), (230, 230, 230), 1)

                x_coords = np.linspace(0, w - 1, S).astype(int)
                
                # 2. On utilise le signal nottoyé pour le tracé
                y_coords = (offsets[i] - sig_clean * scale_y).astype(int)
                y_coords = np.clip(y_coords, 0, h - 1)

                pts = np.vstack((x_coords, y_coords)).T.reshape((-1, 1, 2))
                # thickness=2 et LINE_AA pour un rendu parfait pour Dino
                cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            output_images[b, n] = img_tensor

    return output_images



if __name__ == "__main__":
    file_path = "../../../exams_part0.hdf5"

    with h5py.File(file_path, 'r') as f:
        tracings  = torch.from_numpy(f['tracings'][:10]).permute(0, 2, 1)

    images_tensor = create_image_12leads_optimized_cleaned(tracings, h=518, w=518)

    # 2. On "aplatit" pour l'export (on prend tous les segments de tous les batchs)
    # On passe de (B, N, 3, 518, 518) à (B*N, 3, 518, 518)
    flat_images = images_tensor.view(-1, 3, 518, 518)

    # 3. Sauvegarde des 10 premières images du tenseur pour vérification
    for i in range(min(10, flat_images.size(0))):
        # On utilise torchvision pour sauvegarder le tenseur directement en PNG
        vutils.save_image(flat_images[i], f"check_dino_input_{i}.png")

    print(f"Vérification terminée : 10 images sauvegardées depuis le tenseur de taille {images_tensor.shape}")
