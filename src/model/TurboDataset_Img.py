import os, sys
import torch

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from TurboDataset import TurboDataset
#from generate_image import create_image_12leads_perchan



class TurboDataset_Img(TurboDataset):
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
        for batch_x, batch_y, batch_mask in super().__iter__():
            # 1. Generation des images en uint8
            batch_images = create_image_12leads_perchan(
                batch_x, 
                h=self.h, 
                w=self.w, 
                segment_size=self.segment_size
            )

            # 2. Logique du masque temporelle
            signal_lengths = batch_mask[:, :, 0].sum(dim=1)
            num_windows = torch.ceil(signal_lengths / self.segment_size).long()
            indices = torch.arange(batch_images.shape[1]).expand(batch_images.shape[0], -1)
            mask_images = indices >= num_windows.unsqueeze(1)

            yield batch_images, batch_y, mask_images
