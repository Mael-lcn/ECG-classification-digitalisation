import h5py
import glob
import os
import torch
from torch.utils.data import Dataset
import bisect
import pandas as pd
import numpy as np



class LargeH5Dataset(Dataset):
    """
    Dataset PyTorch conçu pour gérer de très gros volumes de données ECG fragmentés 
    en plusieurs fichiers HDF5 et CSV.

    Stratégie :
    - Les données ne sont pas chargées en RAM à l'initialisation.
    - Seule la structure (taille de chaque fichier) est scannée au début.
    - Les fichiers sont ouverts/fermés dynamiquement pendant l'entraînement (`__getitem__`).
    - Gère l'alignement des colonnes de labels pour garantir que l'ordre des classes 
      correspond exactement à la sortie du modèle.

    Attributes:
        classes (list): Liste des noms de classes cibles (ordre fixe).
        h5_paths (list): Liste des chemins vers les fichiers .hdf5 triés.
        csv_paths (list): Liste des chemins vers les fichiers .csv triés.
        cumulative_sizes (list): Liste des index cumulés pour mapper un index global vers un fichier spécifique.
        total_length (int): Nombre total d'échantillons dans tout le dataset.
    """
    def __init__(self, input_dir, classes_list):
        """
        Initialise le dataset en scannant le répertoire.

        Args:
            input_dir (str): Chemin du dossier contenant les paires de fichiers .hdf5 et .csv.
            classes_list (list): La liste ordonnée des classes (FINAL_CLASSES). 
                                 Crucial pour que le vecteur One-Hot/Multi-hot soit cohérent.
        """
        self.classes = classes_list
        self.num_classes = len(classes_list)

        # 1. Scan des fichiers
        self.h5_paths = glob.glob(os.path.join(input_dir, '*.hdf5'))
        self.csv_paths = glob.glob(os.path.join(input_dir, '*.csv'))

        # Trie pour garantir que h5_paths[i] correspond à csv_paths[i]
        self.h5_paths.sort()
        self.csv_paths.sort()

        assert len(self.h5_paths) == len(self.csv_paths), "Mismatch fichiers H5/CSV"

        # Construction de la carte cumulative
        self.cumulative_sizes = []
        cumul = 0
        print(f"Scan des fichiers pour {self.num_classes} classes...")
        for p in self.h5_paths:
            with h5py.File(p, 'r') as f:
                cumul += f['exam_id'].shape[0]
            self.cumulative_sizes.append(cumul)
        self.total_length = cumul
        print(f"Dataset prêt : {self.total_length} échantillons.")

        # Variables de cache 
        self.file_handle = None
        self.current_file_idx = None
        self.current_labels_map = {} 


    def __len__(self):
        """Retourne la taille totale du dataset (tous fichiers confondus)."""
        return self.total_length


    def __getitem__(self, idx):
        """
        Récupère un échantillon (Tracé + Label) à partir d'un index global.

        Args:
            idx (int): Index global de l'échantillon (entre 0 et total_length - 1).

        Returns:
            tuple: (tracing_tensor, label_tensor)
                   - tracing_tensor : Tensor (Channels=12, Time)
                   - label_tensor : Tensor (Num_Classes,)
        """
        # 1. Trouver le fichier
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # 2. Index local
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        # 3. Switch de fichier
        if self.current_file_idx != file_idx:
            if self.file_handle is not None:
                self.file_handle.close()

            self.file_handle = h5py.File(self.h5_paths[file_idx], 'r')

            df = pd.read_csv(self.csv_paths[file_idx])

            # A. On force l'extraction des colonnes dans l'ordre exact de FINAL_CLASSES
            # si une colonne manque, on met 0
            df_labels = df.reindex(columns=self.classes, fill_value=0)

            # B. On convertit tout le dataframe en une matrice Numpy de Float
            # Shape : (N_samples, 27)
            labels_matrix = df_labels.astype(np.float32).values

            # C. On récupère les IDs en string
            ids = df['exam_id'].astype(str).values

            # D. On crée le dictionnaire : ID -> Vecteur Numpy (taille 27)
            self.current_labels_map = dict(zip(ids, labels_matrix))

            self.current_file_idx = file_idx

        # 4. Lecture des données Brutes
        tracing = self.file_handle['tracings'][local_idx]
        raw_id = self.file_handle['exam_id'][local_idx]

        # 5. Décodage ID
        if isinstance(raw_id, bytes):
            exam_id_str = raw_id.decode('utf-8')
        else:
            exam_id_str = str(raw_id)

        # 7. Conversion Tensor
        # Tracing: (Temps, 12) -> (12, Temps)
        tracing_tensor = torch.from_numpy(tracing).float().transpose(0, 1)

        # Label: Numpy -> Tensor
        label_tensor = torch.from_numpy(self.current_labels_map[exam_id_str]).float()

        return tracing_tensor, label_tensor


    def __del__(self):
        """
        Destructeur de la classe.
        S'assure que le fichier HDF5 est bien fermé quand l'objet Dataset est détruit.
        """
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass
