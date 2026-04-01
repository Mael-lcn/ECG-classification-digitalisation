import torch

from TurboDataset import TurboDataset


class TurboDataset_Img(TurboDataset):
    """
    Dataset hérité pour la génération à la volée d'images ECG pour DINOv3.
    
    Hérite de la mécanique I/O ultra-optimisée de TurboDataset (lecture 1D contiguë)
    et applique la rastérisation OpenCV en multi-processing CPU juste avant 
    l'envoi au GPU.
    """
    def __init__(self, data_path, generate_img, segment_size=1000, h=512, w=512, **kwargs):
        # On initialise toute la machinerie I/O du parent
        super().__init__(data_path, **kwargs)
        self.segment_size = segment_size
        self.h = h
        self.w = w
        self.generate_img = generate_img

    def __iter__(self):
        for batch_x, batch_y, batch_mask in super().__iter__():
            # 1. Generation des images en uint8
            batch_images = self.generate_img(
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
