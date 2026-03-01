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
        Générateur SOTA : Lecture contiguë segmentée, Tri virtuel et Zero-Copy.
        Optimisé pour le débit GPU et la sobriété RAM.
        """
        worker_info = get_worker_info()
        num_files = len(self.h5_paths)

        # 1. Distribution équitable des fichiers par Worker
        if worker_info is None:
            file_indices = list(range(num_files))
        else:
            per_worker = int(math.ceil(num_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_indices = list(range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_files)))

        if self.shuffle:
            np.random.shuffle(file_indices)

        for f_idx in file_indices:
            h5_path = self.h5_paths[f_idx]
            csv_path = self.csv_paths[f_idx]

            # Chargement des métadonnées (Léger en RAM)
            df = pd.read_csv(csv_path)
            labels_matrix = df.reindex(columns=self.classes, fill_value=0).astype(np.float32).values
            starts = df['start_offset'].values
            lengths = df['length'].values
            total_samples = len(df)

            # 2. Mega-Pool Virtuelle (Tri global pour optimiser le padding GPU)
            # On définit un pool d'indices sans charger les signaux
            mega_pool_starts = list(range(0, total_samples, self.mega_batch_size))
            if self.shuffle:
                np.random.shuffle(mega_pool_starts)

            with h5py.File(h5_path, 'r') as h5_file:
                for mp_start in mega_pool_starts:
                    mp_end = min(mp_start + self.mega_batch_size, total_samples)

                    # Tri par longueur décroissante au sein du pool (Virtuel)
                    pool_lengths = lengths[mp_start:mp_end]
                    mc_argsort = np.argsort(pool_lengths)[::-1]

                    # 3. Lecture Physique par Tranches
                    # On définit Y = 4 : on traite 4 batchs à la fois pour lisser l'I/O
                    y_factor = 4 
                    ram_chunk_size = self.batch_size * y_factor

                    for k in range(0, len(mc_argsort), ram_chunk_size):
                        indices_in_pool = mc_argsort[k : k + ram_chunk_size]
                        real_indices = mp_start + indices_in_pool

                        # On trouve la plage minimale/maximale pour une lecture contiguë
                        idx_min, idx_max = np.min(real_indices), np.max(real_indices)

                        # Lecture d'un bloc physique unique (Vitesse maximale du disque)
                        raw_block_np = h5_file['tracings'][idx_min : idx_max + 1]

                        # Zero-copy : On enveloppe la mémoire NumPy dans PyTorch
                        raw_block_pt = torch.from_numpy(raw_block_np)

                        # --- 4. ASSEMBLAGE ET YIELD À LA VOLÉE ---
                        for j in range(0, len(indices_in_pool), self.batch_size):
                            batch_idx_in_chunk = slice(j, j + self.batch_size)
                            cur_indices = real_indices[batch_idx_in_chunk]
                            cur_bs = len(cur_indices)

                            # Padding dynamique basé sur le signal le plus long du mini-batch
                            target_t = MAX_SIGNAL_LENGTH if self.use_static_padding else lengths[cur_indices[0]]

                            # Pré-allocation
                            batch_signals = torch.zeros((cur_bs, 12, target_t), dtype=torch.float32)
                            batch_labels = torch.from_numpy(labels_matrix[cur_indices])
                            batch_lens = torch.from_numpy(lengths[cur_indices]).long()

                            # Remplissage par slicing mémoire
                            for i in range(cur_bs):
                                idx = cur_indices[i]
                                s_start = starts[idx]
                                s_len = lengths[idx]
                                read_len = min(s_len, target_t)

                                # On calcule la position relative dans le bloc lu
                                rel_idx = idx - idx_min
                                batch_signals[i, :, :read_len] = raw_block_pt[rel_idx, :, s_start : s_start + read_len]

                            # Envoi direct au GPU via le DataLoader
                            yield batch_signals, batch_labels, batch_lens

                        # On force la libération du bloc avant de charger le suivant
                        del raw_block_np, raw_block_pt
