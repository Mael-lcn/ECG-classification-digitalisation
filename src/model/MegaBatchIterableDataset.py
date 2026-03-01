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
        worker_info = get_worker_info()
        num_files = len(self.h5_paths)

        # --- 1) Distribution workers ---
        if worker_info is None:
            file_indices = list(range(num_files))
        else:
            worker_id = worker_info.id
            per_worker = int(math.ceil(num_files / float(worker_info.num_workers)))
            file_indices = list(range(
                worker_id * per_worker,
                min((worker_id + 1) * per_worker, num_files)
            ))

        if self.shuffle:
            np.random.shuffle(file_indices)

        # 🔹 Pool fixe y=4
        pool_size = 4 * self.batch_size

        for f_idx in file_indices:
            h5_path = self.h5_paths[f_idx]
            csv_path = self.csv_paths[f_idx]

            # --- 2) Metadata ---
            df = pd.read_csv(csv_path)
            labels_matrix = df.reindex(columns=self.classes, fill_value=0).astype(np.float32).values
            starts = df['start_offset'].values
            lengths = df['length'].values
            total_samples = len(df)

            # Pools contigus (lecture séquentielle)
            pool_starts = list(range(0, total_samples, pool_size))
            if self.shuffle:
                np.random.shuffle(pool_starts)

            with h5py.File(h5_path, 'r') as h5_file:
                ds_tracings = h5_file['tracings']

                for pool_start in pool_starts:
                    pool_end = min(pool_start + pool_size, total_samples)

                    # 🔥 Lecture CONTIGUË du pool (séquentielle → rapide)
                    raw_pool_np = ds_tracings[pool_start:pool_end]

                    # Option float16 disque
                    if raw_pool_np.dtype == np.float16:
                        raw_pool_pt = torch.from_numpy(raw_pool_np).to(torch.float32)
                    else:
                        raw_pool_pt = torch.from_numpy(raw_pool_np)

                    pool_lengths = lengths[pool_start:pool_end]

                    # Tri décroissant pour padding optimisé
                    argsort_pool = np.argsort(pool_lengths)[::-1]

                    # --- Mini-batches dans le pool ---
                    for j in range(0, len(argsort_pool), self.batch_size):
                        batch_rel_idx = argsort_pool[j:j+self.batch_size]
                        real_indices = pool_start + batch_rel_idx

                        cur_bs = len(batch_rel_idx)

                        if self.use_static_padding:
                            target_t = MAX_SIGNAL_LENGTH
                        else:
                            target_t = int(pool_lengths[batch_rel_idx[0]])

                        # 🔹 Allocation batch (CPU)
                        batch_signals = torch.zeros(
                            (cur_bs, 12, target_t),
                            dtype=torch.float32
                        )

                        batch_labels = torch.from_numpy(labels_matrix[real_indices])
                        batch_lens = torch.from_numpy(lengths[real_indices]).long()

                        # Slice direct depuis raw_pool_pt (déjà en RAM)
                        for i in range(cur_bs):
                            rel_idx = batch_rel_idx[i]
                            g_idx = real_indices[i]

                            s_start = starts[g_idx]
                            s_len = lengths[g_idx]
                            read_len = min(s_len, target_t)

                            batch_signals[i, :, :read_len] = \
                                raw_pool_pt[rel_idx, :, s_start:s_start+read_len]

                        yield batch_signals, batch_labels, batch_lens

                    # nettoyage pool
                    del raw_pool_np, raw_pool_pt
