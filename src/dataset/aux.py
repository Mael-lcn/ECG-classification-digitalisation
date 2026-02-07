import os
import glob
import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio



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


def load(path, use_csv=True, to_gpu=True):
    """
    Charge les données HDF5/CSV.

    SPÉCIFIQUE : Seul le dataset nommé 'tracings' est envoyé sur le GPU.
    Tout le reste ('exam_id', etc.) reste sur le CPU.

    Args:
        path (tuple): (chemin_hdf5, chemin_csv).
        use_csv (bool): Charger le CSV.
        to_gpu (bool): Activer le transfert GPU pour 'tracings'.

    Returns:
        tuple: (h5_content, csv_data)
    """
    path_hd, path_csv = path
    csv_data = None

    # Charge le CSV
    if use_csv:
        csv_data = pd.read_csv(path_csv)

    # Charge tout le HDF5 en RAM
    h5_content = {}
    with h5py.File(path_hd, 'r') as f:
        for key in f.keys():
            h5_content[key] = f[key][:] # Charge en Numpy array

    if to_gpu:
        # Détection automatique du GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        data_cpu = h5_content['tracings']

        # A. Conversion en Tensor
        tensor = torch.from_numpy(data_cpu) # (N, T, C)

        # B. Permutation (N, T, C) -> (N, C, T)
        # Standard en PyTorch pour éviter les erreurs de dimension
        tensor = tensor.permute(0, 2, 1)

        # C. Contiguous
        # Réorganise la mémoire physiquement pour éviter RuntimeError: view size...
        tensor = tensor.contiguous()

        # Typage Float32
        if tensor.dtype == torch.float64:
            tensor = tensor.float()

        # E. Transfert GPU
        h5_content['tracings'] = tensor.to(device)
        print(f"Données 'tracings' converties (N, C, T) et transférées sur : {device}")

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
def get_active_boundaries(tracings, threshold=1e-6):
    """
    Détecte les indices de fin d'activité (Time) pour chaque patient.
    
    Optimisé pour le format (Batch, Channel, Time).
    On réduit la dimension Channel pour voir si le signal est actif à l'instant t.

    Args:
        tracings (torch.Tensor): Tenseur ECG.
            Format attendu : (N, C, T) -> (Batch, Channels, Time).
            Ex: (32, 12, 5000).
        threshold (float): Seuil de silence (zéro machine).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - start_idx (N,): Indices de début (généralement 0).
            - end_idx (N,): Indices de fin (dernier point non nul).
    """
    # 1. Sécurité Dimension
    if tracings.dim() == 2:
        # Cas (C, T) -> (1, C, T)
        tracings = tracings.unsqueeze(0)

    N, C, T = tracings.shape

    # 2. Masque d'Activité Temporelle (Réduction des Canaux)
    # On cherche le max absolu sur les canaux (dim 1) pour chaque instant t.
    # Résultat : (N, T)
    is_active_time = torch.amax(torch.abs(tracings), dim=1) > threshold

    # 3. Start Index (Premier '1' sur l'axe temps)
    start_idx = is_active_time.int().argmax(dim=1)

    # 4. End Index (Dernier '1' sur l'axe temps)
    # On inverse l'axe temps
    flipped_active = is_active_time.flip(dims=[1])
    zeros_from_end = flipped_active.int().argmax(dim=1)
    end_idx = T - zeros_from_end

    # 5. Gestion des Signaux Vides (Flatlines)
    # Si toute la ligne est False, argmax renvoie 0 -> end_idx = T (faux)
    # On vérifie s'il y a au moins un True
    has_any_signal = is_active_time.any(dim=1)

    start_idx = torch.where(has_any_signal, start_idx, torch.zeros_like(start_idx))
    end_idx = torch.where(has_any_signal, end_idx, torch.zeros_like(end_idx))

    return start_idx, end_idx


def re_sampling(data, csv, fo=400):
    """
    Rééchantillonne les signaux ECG (N, C, T) vers une fréquence cible (fo).
    
    Spécificité :
        - Travaille sur (Batch, Channel, Time).
        - Coupe le padding inutile à la fin avant resampling.
        - Utilise torchaudio (optimisé GPU).

    Args:
        data (dict): {'tracings': Tensor(N, C, T) sur GPU, 'exam_id': ...}
        csv (pd.DataFrame): Métadonnées ('exam_id', 'frequences').
        fo (int): Fréquence cible.

    Returns:
        torch.Tensor: (N, C, T_new) sur GPU.
    """
    tracings = data['tracings']
    all_ids = data['exam_id']

    N, C, T = tracings.shape
    device = tracings.device

    # 1. Détection de la longueur utile (Fin du signal)
    # end_idxs correspond à l'index temporel max où il y a du signal
    _, end_idxs = get_active_boundaries(tracings)

    # Transfert CPU pour mapping Pandas
    real_lengths = end_idxs.cpu().numpy()

    # 2. Création des Maps (ID -> Info)
    id_to_len = {}
    id_to_idx = {}

    for i in range(N):
        raw_id = all_ids[i]
        # Décodage robuste bytes/str
        s_id = raw_id.decode('utf-8') if isinstance(raw_id, bytes) else str(raw_id)
        
        id_to_len[s_id] = real_lengths[i]
        id_to_idx[s_id] = i

    # 3. Préparation Dataframe (CPU)
    # On travaille sur une copie pour ne pas toucher au CSV original
    df_proc = csv.copy()
    df_proc['exam_id'] = df_proc['exam_id'].astype(str)

    # Mapping des longueurs réelles détectées sur le GPU
    df_proc['current_len'] = df_proc['exam_id'].map(id_to_len)

    # Filtrage : on ne garde que les IDs présents dans le batch GPU
    df_valid = df_proc.dropna(subset=['current_len'])
    df_valid['current_len'] = df_valid['current_len'].astype(int)

    if df_valid.empty:
        # Retourne un tenseur vide si aucune correspondance
        return torch.zeros((N, C, 1), device=device)

    # 4. Calcul taille cible
    # T_new = ceil(T_old * fo / fs)
    df_valid['target_len'] = np.ceil(
        df_valid['current_len'] * fo / df_valid['frequences']
    ).astype(int)

    max_target_len = int(df_valid['target_len'].max())

    # Allocation Tenseur Sortie (N, C, T_new)
    # Initialisé à 0 (Padding implicite)
    new_tracings = torch.zeros((N, C, max_target_len), dtype=torch.float32, device=device)

    # 5. Traitement par Groupe (Fréquence + Longueur)
    # Cela permet de batcher le resampling
    groups = df_valid.groupby(['frequences', 'current_len'])

    for (fs_in, len_in), group in groups:
        if len_in == 0: continue

        # A. Récupération des indices batch
        batch_ids = group['exam_id'].values
        # On retrouve les index du tenseur GPU via la map
        indices = [id_to_idx[bid] for bid in batch_ids if bid in id_to_idx]

        if not indices: continue

        gpu_idx = torch.tensor(indices, device=device)

        # B. Slicing GPU (N, C, T) -> On coupe sur dim 2 (Time)
        # On prend tous les indices du groupe, tous les canaux, jusqu'à len_in
        sub_batch = tracings[gpu_idx, :, :len_in]

        # C. Resampling
        resampler = torchaudio.transforms.Resample(
            orig_freq=fs_in,
            new_freq=fo,
            resampling_method='sinc_interp_kaiser',
            lowpass_filter_width=6,
            rolloff=0.99,
            dtype=torch.float32
        ).to(device)

        resampled_sub = resampler(sub_batch)

        # D. Placement dans le tenseur final
        # On s'assure de ne pas dépasser la taille allouée (arrondi float)
        current_t_new = resampled_sub.shape[-1]
        limit = min(current_t_new, max_target_len)

        new_tracings[gpu_idx, :, :limit] = resampled_sub[:, :, :limit]

    return new_tracings




"""
    Méthode de la partie 2
"""
def z_norm(tracings, eps=1e-5):
    """
    Normalisation Z-Score (Mean=0, Std=1) par canal.
    Adapté au format (N, C, T).

    Calcule les stats uniquement sur la zone active (non-zéro) 
    pour éviter que le padding ne les écrases.

    Args:
        tracings (torch.Tensor): (N, C, T) sur GPU.
        eps (float): Stabilité numérique.

    Returns:
        torch.Tensor: (N, C, T) normalisé.
    """
    N, C, T = tracings.shape
    device = tracings.device

    # 1. Détection Active Zone (Par canal individuel)
    is_nonzero = torch.abs(tracings) > eps      # (N, C, T)

    # Start: Premier True sur l'axe T (dim 2)
    starts = torch.argmax(is_nonzero.int(), dim=2)  # (N, C)

    # End: Dernier True sur l'axe T (dim 2)
    flipped = is_nonzero.flip(dims=[2])
    ends_from_back = torch.argmax(flipped.int(), dim=2)
    ends = T - ends_from_back # (N, C)

    # 2. Création Masque 3D
    # Grid temporelle : (1, 1, T)
    grid = torch.arange(T, device=device).view(1, 1, T)

    # Bornes : (N, C, 1) pour comparer avec la grid (T)
    starts_bc = starts.unsqueeze(-1) 
    ends_bc = ends.unsqueeze(-1)

    # Masque float (1.0 = Signal, 0.0 = Padding)
    active_mask = ((grid >= starts_bc) & (grid < ends_bc)).float()

    # 3. Calcul Stats Pondérées (Sur dim 2 = Time)
    # Compte le nombre de points réels par canal
    count = active_mask.sum(dim=2, keepdim=True) # (N, C, 1)
    count = count.clamp(min=1.0) # Évite div/0

    # Moyenne
    sum_val = (tracings * active_mask).sum(dim=2, keepdim=True)
    means = sum_val / count

    # Variance / Std
    # On centre, on masque le padding (qui devient 0), on met au carré
    diff = (tracings - means) * active_mask
    sq_diff = diff.pow(2)
    vars_ = sq_diff.sum(dim=2, keepdim=True) / count
    stds = torch.sqrt(vars_)

    # Sécurité signal plat
    stds = torch.clamp(stds, min=eps)

    # 4. Normalisation
    # (X - mu) / sigma
    # On ré-applique le masque à la fin pour garantir que le padding reste un vrai 0 absolu
    data_norm = ((tracings - means) / stds) * active_mask

    return data_norm
