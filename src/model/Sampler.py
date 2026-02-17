from torch.utils.data import Sampler
import numpy as np



class MegaBatchSortishSampler(Sampler):
    """
    Sampler personnalisé optimisant le padding dynamique pour des datasets fragmentés (HDF5).

    Ce sampler implémente une stratégie de "Smart Batching" conçue pour
    minimiser le padding inutile dans les batchs, tout en respectant la contiguïté des fichiers 
    pour optimiser les I/O.

    Le processus se déroule en trois étapes hiérarchiques :
    1. Ordre des fichiers : Les fichiers sources sont traités un par un (ordre aléatoire si shuffle=True).
    2. Mega-Batch : Au sein d'un fichier, on prélève un grand pool d'indices (Mega-Batch).
    3. Tri Local : Ce pool est trié par longueur de séquence décroissante, puis découpé en
    mini-batchs homogènes.

    Cette approche réduit considérablement la consommation mémoire (VRAM) lors de l'utilisation 
    de `pad_sequence`, car les séquences courtes ne sont pas mélangées avec les très longues.

    Args:
        dataset (Dataset): Instance du dataset (doit exposer `cumulative_sizes` et `all_lengths`).
        batch_size (int): Taille cible du mini-batch final pour le GPU.
        mega_batch_factor (int, optional): Facteur déterminant la taille du buffer de tri.
            Taille du buffer = batch_size * mega_batch_factor.
            Un facteur élevé améliore l'efficacité du padding mais réduit la diversité 
            aléatoire locale du batch. Défaut: 20.
        shuffle (bool, optional): Si True, mélange l'ordre des fichiers et les indices au sein 
            de chaque fichier avant le tri. Défaut: True.
    """
    def __init__(self, dataset, batch_size, mega_batch_factor=20, shuffle=True):
        """
        Initialise le sampler et pré-calcule les bornes des fichiers.

        Args:
            dataset (Dataset): Le dataset source contenant les métadonnées de longueur.
            batch_size (int): La taille des batchs finaux renvoyés par le DataLoader.
            mega_batch_factor (int): Multiplicateur définissant la granularité du tri.
            shuffle (bool): Active le mélange aléatoire (Shuffle global des fichiers + Shuffle local des indices).
        """
        # Initialisation des variables
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # On définit la taille du "Méga-Lot" (ex: 50 * 64 = 3200 indices)
        self.mega_batch_size = self.batch_size * mega_batch_factor

        # Pré-calcul des bornes de fichiers
        self.boundaries = [0] + self.dataset.cumulative_sizes


    def __iter__(self):
        """
        Générateur principal itérant sur les indices du dataset.

        Ce générateur ne renvoie pas un indice unique, mais une liste d'indices constituant un batch complet.

        Logique d'itération :
        1. Pour chaque fichier :
            a. Récupère tous les indices locaux.
            b. Mélange ces indices (si shuffle=True).
            c. Crée des "Mega-Chunks".
            d. Trie chaque chunk par longueur de séquence (descendant).
            e. Découpe le chunk trié en mini-batchs de taille `batch_size`.
            f. Yield le mini-batch.

        Yields:
            list[int]: Une liste d'indices d'échantillons correspondant à un mini-batch.
        """
        num_files = len(self.dataset.h5_paths)
        file_order = np.arange(num_files)
        if self.shuffle:
            np.random.shuffle(file_order)

        # On traite chaque fichier un par un
        for f_idx in file_order:
            # Bornes globales de ce fichier
            start_global = self.boundaries[f_idx]
            end_global = self.boundaries[f_idx + 1]

            # Génération des indices globaux pour ce fichier
            indices_in_file = np.arange(start_global, end_global)

            # 3. Mélange local (Intra-fichier)
            if self.shuffle:
                np.random.shuffle(indices_in_file)

            # On découpe les indices du fichier en mega batch
            for i in range(0, len(indices_in_file), self.mega_batch_size):
                mega_chunk = indices_in_file[i : i + self.mega_batch_size]

                # Tri par longueur descendant pour minimiser le padding dans le batch
                lengths = self.dataset.all_lengths[mega_chunk]
                argsort = np.argsort(lengths)[::-1]
                sorted_mega_chunk = mega_chunk[argsort]

                # Découpage final en Mini-Batch
                for j in range(0, len(sorted_mega_chunk), self.batch_size):
                    batch = sorted_mega_chunk[j : j + self.batch_size]
                    yield batch.tolist()


    def __len__(self):
        """
        Renvoie le nombre total de batches dans une époque.
        """
        return (self.dataset.total_length + self.batch_size-1) // self.batch_size
