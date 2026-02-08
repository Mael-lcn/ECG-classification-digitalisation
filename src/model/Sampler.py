from torch.utils.data import Sampler
import numpy as np



class MegaBatchSortishSampler(Sampler):
    """
    Sampler personnalisé pour optimiser le padding des séquences de longueurs variables.

    Stratégie "Global Shuffle, Local Sort":
    1. Mélange globalement tous les indices du dataset (Random).
    2. Prélève un 'Mega-Batch' (ex: 50 * batch_size).
    3. Trie ce Mega-Batch par longueur de séquence.
    4. Découpe ce Mega-Batch trié en mini-batches finaux.
    """
    def __init__(self, dataset, batch_size, mega_batch_factor=20, shuffle=True):
        """
        La carte des longueurs (lengths) : Il doit savoir que l'indice 50 dure 3000 points et l'indice 51 dure 4500 points.
        C'est pour cela qu'on lui passe le Dataset.

        Les paramètres de structure :
        batch_size (64) : Pour savoir comment découper les groupes.
        mega_batch_factor (50) : Pour savoir quelle taille de l'ensemble d'indices il doit mélanger avant de trier.

        Args:
            dataset (Dataset): Ton dataset (doit avoir un moyen d'accéder à la longueur) plutot bas en train et long en eval.
            batch_size (int): La taille finale du batch pour le GPU.
            mega_batch_factor (int): Combien de batches on pré-charge pour le tri (défaut 50).
            shuffle (bool): True pour l'entrainement (mélange global), False pour la val.
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
        Générateur principal appelé par le DataLoader à chaque époque.
        
        Logique à implémenter :
        1. Créer une liste d'indices de 0 à N.
        2. Si shuffle=True, mélanger cette liste aléatoirement.
        3. Boucler sur cette liste par pas de (batch_size * mega_batch_factor).
        4. Pour chaque 'Mega-Chunk' :
           a. Récupérer les longueurs de chaque indice via le dataset.
           b. Trier les indices du Mega-Chunk par longueur (descendant).
           c. Découper ce Mega-Chunk trié en petits morceaux de 'batch_size'.
           d. YIELD (renvoyer) chaque petit morceau (liste d'indices).
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
