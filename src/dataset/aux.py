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


def load_metadata(path_h5, path_csv):
    """
    Charge les métadonnées et les identifiants d'examen sans charger les signaux.

    Cette approche 'lazy' permet de planifier le découpage en morceaux (chunks) 
    sans saturer la mémoire vive (VRAM) du système, même pour des fichiers de plusieurs Go.

    Args:
        path_h5 (str): Chemin vers le fichier source HDF5.
        path_csv (str): Chemin vers le fichier CSV contenant les labels/métadonnées.

    Returns:
        tuple: (exam_ids, csv_data)
            - exam_ids (np.ndarray): Tableau des identifiants (clés de synchronisation).
            - csv_data (pd.DataFrame): DataFrame complet des métadonnées.
    """
    # Lecture du CSV
    csv_data = pd.read_csv(path_csv)
    csv_data.drop(columns=['nn_predicted_age'], inplace=True, errors='ignore')

    # Ouverture du HDF5 pour extraire les IDs
    with h5py.File(path_h5, 'r') as f:
        # On lit uniquement le dataset 'exam_id'
        exam_ids = f['exam_id'][:]

    return exam_ids, csv_data


def load_chunk(path_hd, start, end, device):
    """
    Charge une portion spécifique du signal ECG directement depuis le disque.

    Utilise le slicing HDF5 pour ne lire que les octets nécessaires. Les données
    transitent brièvement par la RAM CPU avant d'être injectées sur le GPU.

    Args:
        path_hd (str): Chemin vers le fichier HDF5.
        start (int): Index de début de la tranche (inclus).
        end (int): Index de fin de la tranche (exclus).
        device (torch.device): Périphérique cible (ex: 'cuda:0').

    Returns:
        torch.Tensor: Tenseur sur GPU de forme (N_chunk, C, T) prêt pour le traitement.
    """
    with h5py.File(path_hd, 'r') as f:
        # Lecture uniquement de la fenetre [start:end] évite de charger tout le dataset
        chunk_np = f['tracings'][start:end].astype(np.float32)

    # 1. Conversion en tenseur PyTorch
    chunk_tensor = torch.from_numpy(chunk_np).permute(0, 2, 1).contiguous().to(device)

    return chunk_tensor


def write_results(data_dict, csv, name, output_dir):
    """
    Sauvegarde les signaux et métadonnées.

    Cette fonction exporte les résultats du traitement vers un fichier HDF5 (signaux)
    et un fichier CSV (métadonnées). Elle est optimisée pour des tenseurs PyTorch 
    déjà localisés sur le CPU.

    Args:
        data_dict (dict): Dictionnaire contenant les données à sauvegarder.
            Exemple : {'tracings': torch.Tensor (CPU), 'exam_id': np.array}.
        csv (pd.DataFrame): DataFrame des métadonnées alignées avec les signaux.
        name (str): Nom du fichier de sortie (ex: 'data_processed.hdf5').
        output_dir (str): Répertoire de destination.

    Returns:
        None
    """
    # Création du dossier si inexistant
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, name)
    print(f"[SAVE] Écriture disque (Raw) : {file_path}")

    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            # Conversion Tensor (CPU) -> Numpy
            if torch.is_tensor(value):
                value = value.numpy()

            # Gestion spécifique des IDs pour la compatibilité HDF5
            if key == 'exam_id' and isinstance(value, np.ndarray):
                # Encodage en chaînes de caractères de longueur variable (vlen)
                if value.dtype.kind in {'U', 'S', 'O'}:
                    value = value.astype(h5py.special_dtype(vlen=str))

            # Écriture
            try:
                f.create_dataset(key, data=value)

                shape_str = str(value.shape) if hasattr(value, 'shape') else str(len(value))
                print(f"   -> Dataset '{key}' écrit. Forme : {shape_str}")
            except Exception as e:
                print(f"   [ERREUR] Échec de l'écriture du dataset '{key}': {e}")

    # Sauvegarde des métadonnées CSV
    if csv is not None:
        # On s'assure que l'extension est correcte
        csv_name = name.rsplit('.', 1)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_name)
        try:
            csv.to_csv(csv_path, index=False)
            print(f"   -> CSV associé sauvegardé : {csv_name}")
        except Exception as e:
            print(f"   [ERREUR] Échec de l'écriture du fichier CSV : {e}")




"""
    Méthode de la partie 1
"""
def get_active_boundaries(tracings, threshold=1e-6):
    """
    Détecte les indices de début et fin d'activité.
    
    Args:
        tracings (torch.Tensor): (N, C, T)
        threshold (float): Seuil d'activité.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: start_idx, end_idx (N,)
    """
    if tracings.dim() == 2:
        tracings = tracings.unsqueeze(0)

    # Masque booléen
    is_active_time = tracings.abs().amax(dim=1) > threshold

    start_idx = is_active_time.int().argmax(dim=1)

    tmp_flip = torch.flip(is_active_time, dims=[1])
    last_active_from_end = tmp_flip.int().argmax(dim=1)
    del tmp_flip 

    T = tracings.shape[2]
    end_idx = T - last_active_from_end

    # Correction pour les signaux totalement plats
    has_signal = is_active_time.any(dim=1)
    start_idx = torch.where(has_signal, start_idx, torch.zeros_like(start_idx))
    end_idx = torch.where(has_signal, end_idx, torch.zeros_like(end_idx))

    del is_active_time, has_signal
    return start_idx, end_idx



def re_sampling(data, csv, fo=400):
    """
    Rééchantillonne les signaux ECG en les groupant par fréquence et longueur.

    Le traitement par blocs "naturels" permet d'utiliser des opérations vectorisées
    efficaces tout en contrôlant la consommation VRAM. La mémoire est libérée et le
    cache CUDA est vidé après chaque bloc de fréquence.

    Args:
        data (dict): Dictionnaire contenant 'tracings' (Tensor GPU) et 'exam_id'.
        csv (pd.DataFrame): Métadonnées contenant 'exam_id' et 'frequences'.
        fo (int): Fréquence d'échantillonnage cible (ex: 400Hz).

    Returns:
        torch.Tensor: Le nouveau tenseur rééchantillonné de forme (N, C, T_new).
    """
    tracings = data['tracings']
    device = tracings.device
    N, C, T = tracings.shape

    # Extraction des bornes d'activité
    _, end_idxs = get_active_boundaries(tracings)
    real_lengths = end_idxs.cpu().numpy()

    # Mapping rapide des IDs pour retrouver les index dans le tenseur original
    all_ids = [rid.decode('utf-8') if isinstance(rid, bytes) else str(rid) for rid in data['exam_id']]
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    # Calcul des longueurs cibles (target_len)
    df = csv.copy()
    df['exam_id'] = df['exam_id'].astype(str)
    df['current_len'] = df['exam_id'].map(id_to_idx).map(lambda x: real_lengths[x] if pd.notnull(x) else 0)

    max_target_len = int(np.ceil(df['current_len'] * fo / df['frequences']).max())

    # Allocation du tenseur de destination sur le GPU
    new_tracings = torch.zeros((N, C, max_target_len), dtype=torch.float32, device=device)

    resamplers = {}
    # On itère par groupes homogènes
    grouped = df.groupby(['frequences', 'current_len'])

    for (fs_in, len_in), group in grouped:
        if len_in <= 0: continue

        # Sélection des indices appartenant à ce groupe
        indices = [id_to_idx[bid] for bid in group['exam_id'] if bid in id_to_idx]
        idx_tensor = torch.tensor(indices, device=device)

        # sub_batch extrait une portion
        sub_batch = tracings[idx_tensor, :, :len_in]

        # Initialisation ou réutilisation du module Resample pour noyau partagé
        if fs_in not in resamplers:
            resamplers[fs_in] = torchaudio.transforms.Resample(
                orig_freq=int(fs_in),
                new_freq=int(fo),
                resampling_method='sinc_interp_kaiser',
                lowpass_filter_width=6,
                rolloff=0.99
            ).to(device)

        # Calcul du resampling
        resampled = resamplers[fs_in](sub_batch)

        # Libération de l'entrée du bloc immédiatement après calcul
        del sub_batch 

        # Transfert du résultat vers le tenseur final
        L_target = resampled.shape[-1]
        new_tracings[idx_tensor, :, :L_target] = resampled

        # Nettoyage du bloc traité
        del resampled, idx_tensor
        # Force la libération de la mémoire fragmentée sur le GPU
        # torch.cuda.empty_cache() 

    data['tracings'] = None 
    del tracings
    return new_tracings



"""
    Méthode de la partie 2
"""
def z_norm(chunk, eps=1e-5):
    """
    Normalisation Z-score (Mean=0, Std=1) effectuée directement sur le tenseur fourni.
    
    Cette fonction est optimisée pour la mémoire:
    1. Elle utilise des opérations "in-place" pour éviter les copies.
    2. Elle ne crée qu'un seul tenseur temporaire de la taille du chunk pour la variance.
    3. Elle convertit explicitement les compteurs en Float pour éviter les erreurs de cast.

    Args:
        chunk (torch.Tensor): Un tenseur PyTorch (Vue ou Copie) de forme (N, C, T).
                             Modifié directement en place.
        eps (float): Seuil de stabilité pour éviter la division par zéro et 
                     définir la zone d'activité.

    Returns:
        torch.Tensor: Le même tenseur 'chunk' normalisé.
    """
    _, _, T = chunk.shape
    device = chunk.device

    # Détection Start/Stop par canal
    is_active = (chunk.abs() > eps)

    # Premier index non-nul
    start_indices = is_active.int().argmax(dim=2).unsqueeze(-1) # (N, C, 1)

    # Dernier index non-nul
    flipped_active = is_active.flip(dims=[2])
    end_indices = (T - flipped_active.int().argmax(dim=2)).unsqueeze(-1) # (N, C, 1)
    del flipped_active

    # Création du masque temporel
    t_grid = torch.arange(T, device=device).view(1, 1, T)

    # window_mask est True uniquement entre start et end pour chaque canal
    window_mask = (t_grid >= start_indices) & (t_grid < end_indices)

    # Calculs statistiques
    count = window_mask.sum(dim=2, keepdim=True).float().clamp_(min=1.0)

    # Somme du signal
    mean = torch.sum(chunk * window_mask, dim=2, keepdim=True).div_(count)

    # Variance (x - mu)^2
    res = chunk.sub(mean).pow_(2).mul_(window_mask)
    std = res.sum(dim=2, keepdim=True).div_(count).sqrt_().clamp_(min=eps)
    del res

    # Application In-Place
    chunk.sub_(mean).div_(std).mul_(window_mask)

    return chunk
