import sys
from pathlib import Path

# On ajoute le dossier parent de 'src' au chemin de recherche
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

import h5py
import glob
import os
import torch
import bisect
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from src.dataset.normalization import TARGET_FREQ



MAX_TEMPS = 144
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


class LargeH5Dataset(Dataset):
    """
    Dataset PyTorch optimisé pour la lecture séquentielle de données volumineuses.

    Ce dataset gère des données fragmentées en multiples paires de fichiers HDF5 (tracés) 
    et CSV (labels).

    Stratégie de gestion mémoire :
        1. Initialisation légère : Seule la structure (taille des fichiers) est chargée.
        2. Lazy Loading : Les fichiers sont ouverts uniquement lorsque nécessaire.
        3. Cache Séquentiel : En lecture séquentielle (shuffle=False), le fichier courant 
           et son CSV de labels restent ouverts en mémoire, évitant les réouvertures constantes.

    Attributes:
        classes (list): Liste des noms de classes cibles, définit l'ordre du vecteur one-hot.
        h5_paths (list): Chemins triés des fichiers .hdf5.
        csv_paths (list): Chemins triés des fichiers .csv.
        cumulative_sizes (list): Index cumulatifs permettant de mapper un index global (0..N) 
                                 vers un fichier spécifique.
        total_length (int): Nombre total d'échantillons disponibles.
    """

    def __init__(self, input_dir, classes_list, use_static_padding=False):
        """
        Initialise le dataset en scannant le répertoire cible.

        Args:
            input_dir (str): Chemin du dossier contenant les fichiers .hdf5 et .csv.
            classes_list (list): Liste ordonnée des classes (ex: ['AFIB', 'SR', ...]). 
                                 Essentiel pour garantir la cohérence des labels.
        """
        self.classes = classes_list
        self.num_classes = len(classes_list)

        self.use_static_padding = use_static_padding

        # Ces variables servent à garder le fichier courant ouvert
        self.file_handle = None          # Pointeur vers le fichier H5 ouvert
        self.current_file_idx = -1       # Index du fichier actuellement ouvert
        self.current_labels_map = {}     # Cache des labels du fichier courant (ID -> Vector)
        self.cumulative_sizes = []       # Carte de navigation des index

        all_h5 = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))

        self.h5_paths = []
        self.csv_paths = []

        print(f"Vérification de l'intégrité des paires H5/CSV...")

        for h5_p in all_h5:
            # On déduit le nom du CSV attendu à partir du nom du H5
            base_name = os.path.splitext(h5_p)[0] # enleve .hdf5
            expected_csv = base_name + '.csv'

            if os.path.exists(expected_csv):
                self.h5_paths.append(h5_p)
                self.csv_paths.append(expected_csv)
            else:
                # Si le CSV manque, on ignore aussi le HDF5 pour éviter le décalage
                print(f"ATTENTION: CSV manquant pour {os.path.basename(h5_p)}. Fichier ignoré.")

        # Vérification de l'intégrité des paires de fichiers
        if len(self.h5_paths) != len(self.csv_paths):
            raise ValueError(f"Mismatch: {len(self.h5_paths)} fichiers H5 vs {len(self.csv_paths)} fichiers CSV.")

        # On ouvre brièvement chaque fichier pour connaître sa taille sans charger les données
        cumul = 0
        print(f"Initialisation du Dataset: Scan de {len(self.h5_paths)} fichiers...")

        # Calcul la taille cumulé
        for p in self.h5_paths:
            with h5py.File(p, 'r') as f:
                cumul += f['exam_id'].shape[0]

            self.cumulative_sizes.append(cumul)

        self.total_length = cumul

        self.all_lengths = np.concatenate([pd.read_csv(c, usecols=['length'])['length'].values for c in self.csv_paths])

        print(f"Dataset prêt : {self.total_length} échantillons au total.")


    def __len__(self):
        """Retourne la taille totale du dataset (tous fichiers confondus)."""
        return self.total_length


    def __getitem__(self, idx):
        """
        Récupère un échantillon et son label correspondant.

        Args:
            idx (int): Index global de l'échantillon demandé.

        Returns:
            tuple: (tracing_tensor, label_tensor)
                - tracing_tensor (torch.Tensor): Forme (12, Time), type Float.
                - label_tensor (torch.Tensor): Forme (Num_Classes,), type Float (Multi-hot/One-hot).
        """
        # A. Localisation de la donnée
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # Calcul de l'index local à l'intérieur de ce fichier spécifique
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        # B. Gestion des ressources
        # Si le fichier demandé n'est pas celui actuellement ouvert, on change de fichier
        if self.current_file_idx != file_idx:
            self.load_new_file_resources(file_idx)

        # C. Lecture des données brutes
        tracing_data = self.file_handle['tracings'][local_idx]
        raw_id = self.file_handle['exam_id'][local_idx]

        # D. Traitement de l'ID
        if hasattr(raw_id, 'decode'):
            exam_id_str = raw_id.decode('utf-8')
        else:
            # Sinon (c'est un int, un float, ou déjà une str), on convertit simplement en string
            exam_id_str = str(raw_id)

        csv_id_str = self.current_csv_idx[local_idx]

        # Test de désynchronisation des données
        if exam_id_str != csv_id_str:
            raise ValueError(f"CRITICAL DATA MISMATCH at idx {idx} (local {local_idx}): "
                             f"H5 ID={exam_id_str} vs CSV ID={csv_id_str}. "
                             f"Le CSV n'est pas trié dans le même ordre que le HDF5 !")

        # Récupère les indexs de starts et la taille des signaux
        start = self.current_starts[local_idx]
        length = self.current_lenghts[local_idx]

        # Si on garde le padding
        if self.use_static_padding:
            tracing_tensor = torch.from_numpy(tracing_data).float()
        else:
            # Récupère la partie T utile
            tracing_tensor = torch.from_numpy(tracing_data[:, start : start + length]).float()

        # E. Récupération du Label
        # On pioche dans le dictionnaire pré-chargé en mémoire
        label_vec = self.current_labels_map.get(exam_id_str)

        # F. Conversion en Tensors PyTorch
        label_tensor = torch.from_numpy(label_vec).float()

        if torch.isnan(label_tensor).any():
            raise ValueError(f"[LABEL CORRUPTED] NaN détecté dans les labels pour {exam_id_str} \n {label_tensor}")


        return tracing_tensor, label_tensor, length


    def load_new_file_resources(self, file_idx):
        """
        Méthode interne pour gérer la transition entre deux fichiers.
        Ferme les ressources précédentes et charge les nouvelles métadonnées en RAM.
        
        Args:
            file_idx (int): L'index du nouveau fichier à ouvrir dans `self.h5_paths`.
        """
        # 1. Nettoyage : Fermer l'ancien fichier s'il est ouvert
        if self.file_handle is not None:
            self.file_handle.close()

        # 2. Ouverture : Nouveau fichier HDF5
        self.file_handle = h5py.File(self.h5_paths[file_idx], 'r')

        # 3. Chargement du csv
        df = pd.read_csv(self.csv_paths[file_idx])

        # Récupère les positions effectives des tracés
        self.current_starts = df['start_offset'].values
        self.current_lenghts = df['length'].values

        # 4. Alignement des colonnes
        # On s'assure que les colonnes du CSV correspondent exactement à self.classes
        # `reindex` ajoute des 0 si une classe est manquante dans ce fichier CSV
        df_labels = df.reindex(columns=self.classes, fill_value=0)

        df_labels.fillna(0.0, inplace=True)

        labels_matrix = df_labels.astype(np.float32).values
        ids = df['exam_id'].astype(str).values

        self.current_csv_idx = ids

        # 5. Mise en Cache : Création du dictionnaire ID -> Label
        self.current_labels_map = dict(zip(ids, labels_matrix))

        # Mise à jour de l'état
        self.current_file_idx = file_idx


    def __del__(self):
        """
        Destructeur : Garantit la fermeture propre du fichier HDF5 
        lorsque l'objet Dataset est détruit ou que le script se termine.
        """
        # On vérifie que l'attribut existe et n'est pas None avant de fermer
        if hasattr(self, 'file_handle') and self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:
                pass    # Évite de lever une erreur lors de l'arrêt du programme






def ecg_collate_wrapper(batch, use_static_padding=False, fixed_length=MAX_SIGNAL_LENGTH):
    """
    Transforme une liste d'échantillons de tailles variables en un Batch PyTorch.

    Cette fonction harmonise les dimensions temporelles des signaux pour qu'ils puissent être empilés dans un seul tenseur.
    
    Elle propose deux stratégies via 'use_static_padding' :
    1. use_static_padding=False (Dynamique) : Idéal pour les RNN/Transformer/FCNN.
       Le batch est paddé à la longueur du signal le plus long de ce batch.
       Ex: Si le max du batch est 4500, tout le monde est paddé à 4500.
       
    2. use_static_padding=True (Universel) : Idéal pour les CNN classiques (avec couches Dense).
       Le batch est forcé à une taille fixe 'fixed_length'.
       - Trop court ? Padding (zéros à la fin).
       - Trop long ? Cropping (on coupe la fin).

    Args:
        batch (list): Liste de tuples [(signal, label), ...] fournie par le Dataset.
                      - signal : Tensor de forme (Channels=12, Time=Variable)
                      - label  : Tensor de forme (NumClasses,) ou scalaire.
        use_static_padding (bool): Si True, utilise le padding universel.
                           Si False pad dynamyquement.
        fixed_length (int): La taille cible temporelle (T) pour le mode universel.

    Returns:
        tuple: (padded_signals, stacked_labels)
               - padded_signals : Tensor (Batch, 12, T_Finale)
               - stacked_labels : Tensor (Batch, NumClasses)
    """
    signals, labels, length = zip(*batch)
    # Transpose (Channels, Time) -> (Time, Channels) car pad_sequence travaille sur la dimension 0 (Time)
    signals = [s.T for s in signals]

    # Option 1 : Padding dynamique
    if not use_static_padding:
        # pad_sequence trouve automatiquement le max du batch et comble les trous
        # batch_first=True -> Sortie : (Batch, Max_Time_du_Batch, 12)
        padded_batch = pad_sequence(signals, batch_first=True, padding_value=0.0)

    # Option 2 : Padding/Cropping fixe
    else:
        # On doit traiter chaque signal individuellement pour atteindre 'fixed_length'
        processed_signals = []

        for s in signals:
            current_len = s.shape[0] # Dimension Time (car on a transposé au début)

            if current_len == fixed_length:
                processed_signals.append(s)

            elif current_len < fixed_length:    # Le signal est trop court
                diff = fixed_length - current_len
                # F.pad sur (Time, 12) -> (Pad_Left, Pad_Right, Pad_Top, Pad_Bottom)
                padded = F.pad(s, (0, 0, 0, diff), "constant", 0.0)
                processed_signals.append(padded)

            else:   # Le signal est trop long
                # On ne garde que les 'fixed_length' premiers points temporels
                cropped = s[:fixed_length, :]
                processed_signals.append(cropped)

        # On empile la liste pour créer le tenseur (Batch, Fixed_Length, 12)
        padded_batch = torch.stack(processed_signals)

    # Remise en forme standard (Batch, Channels, Time)
    padded_signals = padded_batch.transpose(1, 2)

    # Empilage des labels
    stacked_labels = torch.stack(labels)

    return padded_signals, stacked_labels, length
