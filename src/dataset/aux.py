import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy import signal



"""
    I/O PART
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
    Rééchantillonne les signaux ECG vers une fréquence cible (fo).

    Cette fonction utilise une stratégie de "Double Grouping" (Fréquence + Longueur Utile).
    Elle ne rééchantillonne que la partie active du signal pour éviter les artefacts
    de Gibbs (oscillations) causés par le padding de zéros.

    Args:
        data (dict): Dictionnaire HDF5 contenant :
            - 'tracings' (np.ndarray): Tenseur (N, T, C).
            - 'exam_id' (np.ndarray): Identifiants des examens.
        csv (pd.DataFrame): Métadonnées contenant 'exam_id' et 'frequences'.
        fo (int, optional): Fréquence cible. Par défaut 400 Hz.

    Returns:
        np.ndarray: Tenseur (N, T_new, C) rééchantillonné et aligné (padding à droite).
    """
    tracings = data['tracings']
    all_ids = data['exam_id']
    n_samples, n_time, n_channels = tracings.shape

    # 1. Détection de la Longueur Utile
    # On identifie la fin réelle du signal pour ignorer le padding existant.

    # Masque booléen (N, T) indiquant l'activité sur au moins un canal
    is_active = np.max(np.abs(tracings), axis=2) > 1e-6

    # Recherche vectorisée du dernier index actif (en inversant l'axe temps)
    flipped_active = is_active[:, ::-1]
    zeros_at_end = np.argmax(flipped_active, axis=1)

    real_lengths = n_time - zeros_at_end

    # Cas particulier : signal entièrement plat/vide
    has_any_signal = np.any(is_active, axis=1)
    real_lengths[~has_any_signal] = 0


    # 2. Construction des Mappings (Séparation Index / Taille)

    id_to_len = {}  # Map ID -> Taille (pour le CSV)
    id_to_idx = {}  # Map ID -> Index Ligne NumPy (pour l'extraction)

    # Limite pour éviter les index out-of-bounds si discordance ID/Data
    assert len(all_ids) == n_samples

    for i in range(n_samples):
        exam_id = all_ids[i]
        # Décodage robuste (bytes -> str)
        key = exam_id.decode('utf-8') if isinstance(exam_id, bytes) else str(exam_id)
    
        id_to_len[key] = real_lengths[i]
        id_to_idx[key] = i  # Stockage explicite de l'index de ligne (0, 1, 2...)


    # 3. Préparation des Métadonnées

    # Injection des longueurs calculées dans le DataFrame
    csv_ids = csv['exam_id'].astype(str)
    csv['temp_len'] = csv_ids.map(id_to_len)

    # Filtrage : on ne garde que les IDs présents dans le HDF5
    csv_valid = csv[csv['temp_len'].notna()].copy()
    csv_valid['temp_len'] = csv_valid['temp_len'].astype(int)


    # 4. Allocation Mémoire du Tenseur de Sortie

    # Calcul de la taille max nécessaire après resampling
    if not csv_valid.empty:
        csv_valid['needed_len'] = np.ceil(
            csv_valid['temp_len'] * fo / csv_valid['frequences']
        ).astype(int)
        max_required_len = csv_valid['needed_len'].max()
    else:
        max_required_len = 0

    if max_required_len == 0:
        return np.zeros((n_samples, 1, n_channels), dtype=np.float32)

    # Initialisation à zéro (le padding final est donc implicite)
    new_tracings = np.zeros((n_samples, int(max_required_len), n_channels), dtype=np.float32)

    # 5. Traitement Vectorisé par Groupes
    # On groupe par (Fréquence, Longueur) pour traiter par blocs homogènes

    groups = csv_valid.groupby(['frequences', 'temp_len'])

    for (fi, length_in), group_df in groups:
        if length_in == 0: continue

        batch_ids = group_df['exam_id'].astype(str).values
        
        # Récupération des indices de ligne via le mapping dédié
        indexes = [id_to_idx[bid] for bid in batch_ids if bid in id_to_idx]
        
        if not indexes: continue

        # Extraction chirurgicale : on ne prend que la zone utile [0:length_in]
        # Cela garantit qu'aucun zéro de padding n'est envoyé au filtre de resampling.
        batch_data = tracings[indexes, :length_in, :]

        # Calcul des ratios entiers pour le resampling polyphase
        gcd = np.gcd(int(fo), int(fi))
        up = int(fo // gcd)
        down = int(fi // gcd)

        # Application du filtre (axis=1 est l'axe temporel)
        resampled_batch = signal.resample_poly(batch_data, up, down, axis=1)

        # Stockage dans le tenseur final
        limit = min(resampled_batch.shape[1], max_required_len)
        new_tracings[indexes, :limit, :] = resampled_batch[:, :limit, :]

    return new_tracings



"""
    Méthode de la partie 2
"""
def z_norm(tracings):
    """
    Normalisation vectorisée.

    1. Calcule Starts/Ends pour tout le monde (N, C).
    2. Construit un masque 3D (N, T, C) via broadcasting.
    3. Utilise des NaNs pour ignorer les zones hors du signal.
    """
    _, n_time, _ = tracings.shape

    # Détection des bornes
    is_nonzero = np.abs(tracings) > 1e-6

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
