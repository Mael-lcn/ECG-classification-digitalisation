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
        Générateur principal itérant sur les indices du dataset de manière optimisée pour HDF5.

        STRATÉGIE "CONTIGUOUS SHUFFLE" (I/O Friendly) :
        Contrairement à un tirage purement aléatoire qui détruit les performances de lecture 
        du disque avec le format HDF5, cette méthode lit les données par blocs contigus.
        Puisque le jeu de données a été mélangé (shuffled) hors-ligne avant la création 
        des fichiers HDF5, chaque bloc contigu est déjà un échantillon statistiquement 
        représentatif (mix de classes).

        Logique d'itération :
        1. Ordre des fichiers : Mélange de l'ordre de lecture des fichiers HDF5.
        2. Blocs Séquentiels : Découpage des indices du fichier en gros blocs ("Mega-Chunks") SÉQUENTIELS.
        3. Mélange des Blocs : Mélange aléatoire de l'ordre d'apparition de ces Mega-Chunks.
        4. Tri Anti-Padding : Au sein d'un Mega-Chunk, tri des signaux par longueur décroissante en RAM.
        5. Mini-Batchs : Découpage du Mega-Chunk trié en mini-batchs de la taille demandée (batch_size).

        Yields:
            list[int]: Une liste d'indices d'échantillons correspondant à un mini-batch prêt pour le GPU.
        """
        num_files = len(self.dataset.h5_paths)
        file_order = np.arange(num_files)

        # 1. Mélange de l'ordre d'ouverture des fichiers
        if self.shuffle:
            np.random.shuffle(file_order)

        # On traite chaque fichier un par un
        for f_idx in file_order:
            # Bornes globales de ce fichier
            start_global = self.boundaries[f_idx]
            end_global = self.boundaries[f_idx + 1]

            # Génération des indices globaux pour ce fichier (strictement séquentiels)
            indices_in_file = np.arange(start_global, end_global)

            # 2. Découpage en Mega-Chunks SÉQUENTIELS (Ex: 0-3199, 3200-6399...)
            mega_chunks = []
            for i in range(0, len(indices_in_file), self.mega_batch_size):
                mega_chunks.append(indices_in_file[i : i + self.mega_batch_size])

            # 3. Mélange de l'ordre des Mega-Chunks (Stochasticité à l'échelle du bloc)
            if self.shuffle:
                np.random.shuffle(mega_chunks)

            # 4. Traitement en RAM de chaque Mega-Chunk
            for mega_chunk in mega_chunks:
                # Récupération des longueurs réelles pour optimiser le padding
                lengths = self.dataset.all_lengths[mega_chunk]

                # Tri par longueur décroissante (les plus longs en premier)
                argsort = np.argsort(lengths)[::-1]
                sorted_mega_chunk = mega_chunk[argsort]

                # 5. Découpage final en Mini-Batchs pour le DataLoader
                for j in range(0, len(sorted_mega_chunk), self.batch_size):
                    batch = sorted_mega_chunk[j : j + self.batch_size]
                    yield batch.tolist()


    def __len__(self):
        """
        Renvoie le nombre total de batches dans une époque.
        """
        return (self.dataset.total_length + self.batch_size-1) // self.batch_size
