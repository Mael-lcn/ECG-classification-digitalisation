import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy import signal



"""
    IO PART
"""
def collect_files(input_dir):
    patch_dict = {}

    # Récupère les files à traiter
    for path_hd in glob.glob(os.path.join(input_dir, '*.hdf5')):
        filename = os.path.basename(path_hd)

        # On suppose que le CSV a le même nom au même endroit
        path_csv = path_hd.replace(".hdf5", ".csv")

        # On vérifie si le CSV existe pour éviter des crashs plus tard
        if os.path.exists(path_csv):
            # On stocke le couple (tuple) directement
            patch_dict[filename] = (path_hd, path_csv)
        else:
            print(f"Attention: CSV manquant pour {filename}")

    return patch_dict


def load(path):
    path_hd, path_csv = path

    # 1. Charger le CSV
    csv_data = pd.read_csv(path_csv)

    # 2. Charger le contenu du HDF5 dans un dictionnaire
    h5_content = {}

    with h5py.File(path_hd, 'r') as f:
        # On boucle sur toutes les clés disponibles à la racine du fichier
        for key in f.keys():
            # On charge les données en mémoire (RAM) immédiatement
            h5_content[key] = f[key][:]

    return h5_content, csv_data


def write_results(data_dict, name, output_dir):
    file_path = os.path.join(output_dir, name)

    with h5py.File(file_path, 'w') as f:
        # On parcourt le dictionnaire pour tout écrire
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)



"""
    Méthode de la partie 1
"""
def re_sampling(data, csv, fo=400):
    tracings = data['tracings'] # Shape (N, Value, Channels)
    all_ids = data['exam_id']   # Shape (N,)

    # Crée un mAPPING: ID -> Index (Rang)
    id_map = {
        (x.decode('utf-8') if isinstance(x, bytes) else str(x)): i 
        for i, x in enumerate(all_ids)
    }

    # On prend la première fréquence trouvée dans le CSV pour calculer la taille cible théorique
    ref_freq = csv['frequences'].iloc[0]
    old_len = tracings.shape[1]

    new_len = int(old_len * fo / ref_freq)
    
    # Shape : (N, NEW_LEN, Channels)
    new_tracings = np.zeros((tracings.shape[0], new_len, tracings.shape[2]), dtype=tracings.dtype)

    freq_to_id = csv.groupby('frequences')['id'].apply(list).to_dict()

    for fi, ids_list in freq_to_id.items():
        # Conversion des IDs (str) en Indices (int) via la map
        indexes = [id_map[x] for x in ids_list if x in id_map]

        if not indexes:
            raise ValueError(f"Erreur, aucun indexes pour les ids: {ids_list} de frequence {fi}")

        # Extraction du lot à traiter
        batch = tracings[indexes]

        # Paramètres de resampling
        gcd = np.gcd(int(fo), int(fi))
        up = int(fo // gcd)
        down = int(fi // gcd)

        # Calcul du resampling
        resampled_batch = signal.resample_poly(batch, up, down, axis=1)

        current_len = resampled_batch.shape[1]
        limit = min(new_len, current_len)
        
        new_tracings[indexes, :limit, :] = resampled_batch[:, :limit, :]

    return new_tracings




"""
    Méthode de la partie 2
"""
def z_norm(tracings):
    """
    Normalisation vectorisée sans aucune boucle Python.
    
    1. Calcule Starts/Ends pour tout le monde (N, C).
    2. Construit un masque 3D (N, T, C) via broadcasting.
    3. Utilise des NaNs pour ignorer les zones hors du signal.
    """
    _, n_time, _ = tracings.shape

    # Détection des bornes
    is_nonzero = np.abs(tracings) > 0

    # Matrices (N, C) des indices de début et de fin
    starts = np.argmax(is_nonzero, axis=1)

    # Pour la fin : inversion temporelle
    ends = n_time - np.argmax(is_nonzero[:, ::-1, :], axis=1)

    # Shape : (1, T, 1) pour être compatible avec (N, 1, C)
    time_indices = np.arange(n_time).reshape(1, n_time, 1)

    starts_bc = starts[:, None, :]
    ends_bc = ends[:, None, :]

    active_mask = (time_indices >= starts_bc) & (time_indices < ends_bc)

    # On remplace tout ce qui est hors du masque par NaN
    tracings[~active_mask] = np.nan

    # Calcul des stats en ignorant les NaNs (donc en ignorant le padding externe)
    # axis=1 : on écrase le temps
    means = np.nanmean(tracings, axis=1, keepdims=True)
    stds = np.nanstd(tracings, axis=1, keepdims=True)

    # Sécurité
    stds[np.isnan(stds)] = 1.0
    stds[stds == 0] = 1.0

    # Normalisation : (Val - Mean) / Std
    data_norm = (tracings - means) / stds

    # On remet les NaNs à 0
    return np.nan_to_num(data_norm, nan=0.0)
