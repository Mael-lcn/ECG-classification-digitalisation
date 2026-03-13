import torch

from TurboDataset import TurboDataset
from generate_image import create_image_12leads_perchan



class DinoImageDataset(TurboDataset):
    """
    Dataset hérité pour la génération à la volée d'images ECG pour DINOv2.
    
    Hérite de la mécanique I/O ultra-optimisée de TurboDataset (lecture 1D contiguë)
    et applique la rastérisation OpenCV en multi-processing CPU juste avant 
    l'envoi au GPU.
    """
    def __init__(self, data_path, segment_size=4000, h=518, w=518, **kwargs):
        # On initialise toute la machinerie I/O du parent
        super().__init__(data_path, **kwargs)
        self.segment_size = segment_size
        self.h = h
        self.w = w

    def __iter__(self):
        # On itère sur le générateur parent qui nous crache des batchs 1D parfaits
        for batch_x, batch_y, batch_mask in super().__iter__():
            # 1. Génération des images
            batch_images = create_image_12leads_perchan(
                batch_x, 
                h=self.h, 
                w=self.w, 
                segment_size=self.segment_size
            )
            # Forme de batch_images : [B, I, 12, H, W]
            B, I, C, H, W = batch_images.shape

            # 2. Création du mask temporelle
            # Le parent donne un masque 1D par step temporel : [B, T, 12]
            # On calcule la vraie longueur du signal en sommant les "True" (1.0) sur un canal
            signal_lengths = batch_mask[:, :, 0].sum(dim=1) # Shape: [B]

            # On calcule le nombre de fenêtres réelles pour chaque patient
            # Ex: Si signal = 5000 et segment = 4000 -> 2 fenêtres valides
            num_windows = torch.ceil(signal_lengths / self.segment_size).long()

            # Création du masque [B, I] attendu par le nn.TransformerEncoder
            # True = Image de padding (à ignorer), False = Image valide
            indices = torch.arange(I).expand(B, I)
            mask_images = indices >= num_windows.unsqueeze(1)
            
            # On renvoie le batch formaté pour DinoTraceTemporal
            yield batch_images, batch_y, mask_images
