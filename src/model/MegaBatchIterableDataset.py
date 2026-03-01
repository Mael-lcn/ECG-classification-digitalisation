import h5py, sys, glob, os
from pathlib import Path
import pandas as pd
import numpy as np
import math

import torch
from torch.utils.data import IterableDataset, get_worker_info


# On ajoute le dossier parent de 'src' au chemin de recherche
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from src.dataset.normalization import TARGET_FREQ



MAX_TEMPS = 144
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


class MegaBatchIterableDataset(IterableDataset):
    """
    Dataset itérable optimisé pour la lecture de fichiers HDF5 massifs avec des signaux de taille variable.

    Cette classe implémente une stratégie de lecture par blocs (Mega-Chunks) pour maximiser
    les performances d'I/O (lecture séquentielle contiguë sur le disque). Elle réduit également
    dynamiquement le padding au sein des mini-batchs en triant les signaux par leur longueur
    utile réelle en RAM, avant de les envoyer au GPU.

    Args:
        h5_paths (list[str]): Liste des chemins vers les fichiers de données .hdf5.
        csv_paths (list[str]): Liste des chemins vers les fichiers de métadonnées .csv correspondants.
        classes_list (list[str]): Liste ordonnée des classes pour la création des vecteurs de labels.
        batch_size (int, optional): Taille du mini-batch final envoyé au modèle. Défaut: 64.
        mega_batch_factor (int, optional): Multiplicateur pour définir la taille du bloc lu en une fois 
            sur le disque (Mega-Chunk = batch_size * mega_batch_factor). Défaut: 16.
        shuffle (bool, optional): Si True, mélange l'ordre des fichiers et l'ordre des Mega-Chunks. Défaut: True.
    """

    def __init__(self, data_path, classes_list, batch_size=64, mega_batch_factor=16, shuffle=True, use_static_padding=False):
        super().__init__()

        self.data_path = data_path
        self.classes = classes_list
        self.num_classes = len(classes_list)
        self.batch_size = batch_size
        self.mega_batch_size = batch_size * mega_batch_factor
        self.shuffle = shuffle
        self.use_static_padding = use_static_padding

        all_h5 = sorted(glob.glob(os.path.join(data_path, '*.hdf5')))
        self.h5_paths = []
        self.csv_paths = []

        for h5_p in all_h5:
            base_name = os.path.splitext(h5_p)[0]
            expected_csv = base_name + '.csv'
            if os.path.exists(expected_csv):
                self.h5_paths.append(h5_p)
                self.csv_paths.append(expected_csv)

        if not self.h5_paths:
            raise FileNotFoundError(f"Aucun fichier HDF5/CSV trouvé dans {self.data_path }")

        assert len(self.h5_paths) == len(self.csv_paths), "Erreur pas autant de csv que de h5"

    def __iter__(self):
        """
        Générateur principal produisant les mini-batchs.
        Gère automatiquement la répartition des fichiers si plusieurs workers (num_workers > 0) sont utilisés.
        """
        worker_info = get_worker_info()
        num_files = len(self.h5_paths)

        # Répartition des fichiers entre les workers pour le multi-processing
        if worker_info is None:
            # Mode mono-processus (num_workers=0) : ce worker traite tous les fichiers
            file_indices = list(range(num_files))
        else:
            # Mode multi-processus : on divise les fichiers équitablement entre les workers
            per_worker = int(math.ceil(num_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, num_files)
            file_indices = list(range(start_idx, end_idx))

        # Mélange de l'ordre de lecture des fichiers si demandé
        if self.shuffle:
            np.random.shuffle(file_indices)

        for f_idx in file_indices:
            h5_path = self.h5_paths[f_idx]
            csv_path = self.csv_paths[f_idx]

            # 1. Chargement et préparation des métadonnées (CSV) en RAM
            df = pd.read_csv(csv_path)

            # Alignement des colonnes de classes et conversion en matrice numpy
            df_labels = df.reindex(columns=self.classes, fill_value=0).fillna(0.0)
            labels_matrix = df_labels.astype(np.float32).values

            # Extraction des informations de positionnement temporelles
            starts = df['start_offset'].values
            lengths = df['length'].values
            total_samples_in_file = len(df)

            # 2. Découpage logique du fichier en blocs contigus (Mega-Chunks)
            mega_chunk_starts = list(range(0, total_samples_in_file, self.mega_batch_size))
            if self.shuffle:
                np.random.shuffle(mega_chunk_starts)

            # 3. Ouverture du fichier HDF5 et itération sur les Mega-Chunks
            with h5py.File(h5_path, 'r') as h5_file:
                for mc_start in mega_chunk_starts:
                    mc_end = min(mc_start + self.mega_batch_size, total_samples_in_file)
 
                    # Lecture disque optimisée : un seul appel I/O pour charger tout le bloc en RAM
                    # La matrice brute contient encore le padding statique du HDF5 à ce stade
                    raw_tracings = h5_file['tracings'][mc_start:mc_end]

                    # Récupération des métadonnées locales à ce bloc spécifique
                    mc_labels = labels_matrix[mc_start:mc_end]
                    mc_starts = starts[mc_start:mc_end]
                    mc_lengths = lengths[mc_start:mc_end]

                    # 4. Tri local en RAM pour minimiser le padding futur
                    # On trie les indices selon la longueur utile réelle, par ordre décroissant
                    argsort = np.argsort(mc_lengths)[::-1]

                    # 5. Découpage final et assemblage des mini-batchs pour le GPU
                    for j in range(0, len(argsort), self.batch_size):
                        batch_indices = argsort[j : j + self.batch_size]
                        current_batch_size = len(batch_indices)

                        if self.use_static_padding:
                            # Mode Statique : Longueur universelle fixe
                            target_t = MAX_SIGNAL_LENGTH
                        else:
                            # Optimisation du padding : le premier signal est le plus long du mini-batch
                            # Cela définit la taille temporelle exacte nécessaire, sans padding mort
                            target_t = mc_lengths[batch_indices[0]]

                        # Pré-allocation de tenseurs à la taille parfaitement ajustée
                        batch_signals = torch.zeros((current_batch_size, 12, target_t), dtype=torch.float32)
                        batch_labels = torch.zeros((current_batch_size, self.num_classes), dtype=torch.float32)
                        batch_lengths = torch.zeros(current_batch_size, dtype=torch.long)
 
                        # Remplissage du tenseur en filtrant le padding d'origine du HDF5
                        for i, idx in enumerate(batch_indices):
                            s_start = mc_starts[idx]
                            s_len = mc_lengths[idx]
                            # Sécurité : on ne lit pas plus que target_t
                            read_len = min(s_len, target_t)

                            # Extraction de la portion utile et insertion dans le tenseur PyTorch
                            utile_signal = raw_tracings[idx, :, s_start : s_start + read_len]
                            batch_signals[i, :, :read_len] = torch.from_numpy(utile_signal)

                            # Assignation du label
                            batch_labels[i] = torch.from_numpy(mc_labels[idx])
                            batch_lengths[i] = read_len

                        yield batch_signals, batch_labels, batch_lengths
