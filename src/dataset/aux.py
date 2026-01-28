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
    """
    Parcourt un répertoire pour associer les fichiers HDF5 à leurs fichiers CSV correspondants.

    Cette fonction recherche tous les fichiers `.hdf5` dans le dossier spécifié.
    Pour chaque fichier trouvé, elle déduit le chemin du fichier `.csv` associé 
    (même nom, même dossier). Si le CSV existe, la paire est ajoutée au résultat.

    Args:
        input_dir (str): Le chemin du répertoire contenant les fichiers à traiter.

    Returns:
        dict: Un dictionnaire structuré comme suit :
            - Clé (str) : Le nom du fichier HDF5 (ex: 'sample.hdf5').
            - Valeur (tuple) : Un tuple contenant (chemin_absolu_hdf5, chemin_absolu_csv).
    """
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


def load(path, use_csv=True):
    """
    Charge les données depuis un fichier HDF5 et, optionnellement, depuis un CSV associé.

    Cette fonction ouvre le fichier HDF5 en lecture seule et extrait tous les datasets
    situés à la racine pour les stocker dans un dictionnaire Python.
    Elle gère également le chargement du fichier CSV associé via pandas si demandé.

    Args:
        path (tuple): Un tuple de deux chaînes de caractères (chemin_hdf5, chemin_csv).
        use_csv (bool, optional): Indique s'il faut charger le fichier CSV. 
                                  Par défaut à True.

    Returns:
        tuple: Un couple contenant :
            - h5_content (dict): Un dictionnaire où les clés sont les noms des datasets 
              HDF5 et les valeurs sont les données sous forme de tableaux NumPy.
            - csv_data (pd.DataFrame ou None): Le contenu du CSV sous forme de DataFrame,
              ou None si use_csv est False.
    """
    path_hd, path_csv = path
    csv_data = None

    # 1. Charger le CSV
    if use_csv:
        csv_data = pd.read_csv(path_csv)

    # 2. Charger le contenu du HDF5 dans un dictionnaire
    h5_content = {}

    with h5py.File(path_hd, 'r') as f:
        # On boucle sur toutes les clés disponibles à la racine du fichier
        for key in f.keys():
            # On charge les données en mémoire immédiatement
            h5_content[key] = f[key][:]

    return h5_content, csv_data


def write_results(data_dict, name, output_dir):
    """
    Sauvegarde le contenu d'un dictionnaire dans un fichier HDF5.

    Cette fonction crée un nouveau fichier HDF5 dans le répertoire spécifié.
    Chaque entrée du dictionnaire est enregistrée comme un 'dataset' distinct
    à la racine du fichier.

    Args:
        data_dict (dict): Dictionnaire contenant les données.
                          - Clés (str) : Noms des datasets HDF5.
                          - Valeurs (array-like) : Données (numpy arrays, listes) à stocker.
        name (str): Le nom du fichier de sortie (ex: 'resultat.hdf5').
        output_dir (str): Le chemin du répertoire de destination.
    """
    file_path = os.path.join(output_dir, name)

    with h5py.File(file_path, 'w') as f:
        # On parcourt le dictionnaire pour tout écrire
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)



"""
    Méthode de la partie 1
"""
def re_sampling(data, csv, fo=400):
    """
    Rééchantillonnage intelligent (Two-Pass strategy).

    Problème : On a des signaux de fréquences ET de durées variées.
    - Un signal 100Hz de 10s fait 1000 points. -> Devient 4000 points à 400Hz.
    - Un signal 500Hz de 10s fait 5000 points. -> Devient 4000 points à 400Hz.

    On ne peut pas deviner la taille du tableau final sans regarder la longueur 'utile'
    de chaque signal avant de commencer.

    Args:
        data (dict): Contient les données brutes. Doit avoir les clés :
            - 'tracings' (np.ndarray) : Tensor de forme (N_exams, T_samples, 12_channels).
            - 'exam_id' (list/array) : Identifiants correspondants aux tracés.
        csv (pd.DataFrame): Métadonnées contenant au moins les colonnes 'exam_id' et 'freq'.
        fo (int, optional): Fréquence d'échantillonnage cible en Hz. Par défaut 400.

    Returns:
        np.ndarray: Un nouveau tenseur de forme (N, T_new, 12) contenant les signaux
                    rééchantillonnés, alignés et padés si nécessaire.
    """
    tracings = data['tracings'] # Shape (N, Total_Buffer_Len, Channels)
    all_ids = data['exam_id']

    # Mapping ID -> Index
    id_map = {
        (x.decode('utf-8') if isinstance(x, bytes) else str(x)): i 
        for i, x in enumerate(all_ids)
    }

    # Regroupement par fréquence
    freq_to_id = csv.groupby('frequences')['exam_id'].apply(list).to_dict()

    # Passe 1 : calcul de la taille maximal d'un signal
    max_required_len = 0

    # On stocke les infos de découpe pour ne pas les recalculer à la passe 2
    groups_metadata = {} 

    for fi, ids_list in freq_to_id.items():
        indexes = [id_map[str(x)] for x in ids_list if str(x) in id_map]
        if not indexes: continue

        # On regarde ce lot brut
        batch = tracings[indexes] 

        # Détection de la longueur utile (On admet que les 0 de padding sont à la fin)
        # On cherche la dernière colonne qui n'est pas vide
        # (N, T, C) -> On écrase N (0) et C (2) pour ne garder que le profil Temporel (T,)
        time_profile = np.max(np.abs(batch), axis=(0, 2))

        active_indices = np.flatnonzero(time_profile > 1e-6)

        if active_indices.size > 0:
            # On prend le dernier index actif + 1
            real_len = active_indices[-1] + 1
        else:
            real_len = 0

        # Calcul de la taille projetée après resampling
        # Formule : N_out = N_in * (F_out / F_in)
        if real_len > 0:
            projected_len = int(np.ceil(real_len * fo / fi))
        else:
            projected_len = 0

        # On met à jour le record global
        if projected_len > max_required_len:
            max_required_len = projected_len

        # On sauvegarde les métadonnées pour aller vite ensuite
        groups_metadata[fi] = {
            'indexes': indexes,
            'real_len': real_len,
            'projected_len': projected_len
        }

    # Sécurité : Si tout est vide
    if max_required_len == 0:
        return np.zeros((tracings.shape[0], 1, tracings.shape[2]), dtype=np.float32)

    # On crée le tableau de taille max sur T
    new_tracings = np.zeros((tracings.shape[0], max_required_len, tracings.shape[2]), dtype=np.float32)


    # Passe 2 : resampling
    for fi, meta in groups_metadata.items():
        indexes = meta['indexes']
        real_len = meta['real_len']

        if real_len == 0: continue

        # 1. Extraction
        batch_trimmed = tracings[indexes, :real_len, :]

        # 2. Resampling
        gcd = np.gcd(int(fo), int(fi))
        up = int(fo // gcd)
        down = int(fi // gcd)

        resampled_batch = signal.resample_poly(batch_trimmed, up, down, axis=1)

        # 3. Placement
        # On colle le résultat dans le grand tableau
        # Comme new_tracings est rempli de 0, le padding se fait tout seul
        current_len = resampled_batch.shape[1]

        # Petit clip de sécurité (au cas où ceil/resample_poly diffèrent de 1 pixel)
        limit = min(max_required_len, current_len)

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
