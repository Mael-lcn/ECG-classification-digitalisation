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

    Seul le dataset nommé 'tracings' est envoyé sur le GPU.
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
        h5_content['tracings'] = f['tracings'][:].astype(np.float32)
        h5_content['exam_id'] = f['exam_id'][:]

    if to_gpu:
        # Détection automatique du GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        h5_content['tracings'] = torch.from_numpy(h5_content['tracings']).permute(0, 2, 1).contiguous().to(device)

    return h5_content, csv_data


def write_results(data_dict, csv, name, output_dir):
    """
    Sauvegarde le contenu d'un dictionnaire dans un fichier HDF5.
    Adapte automatiquement les Tenseurs GPU vers CPU/Numpy.

    Args:
        data_dict (dict): Données (Clé -> Tensor GPU ou List ou Numpy).
        name (str): Nom fichier (ex: 'clean_data.hdf5').
        output_dir (str): Dossier cible.
    """
    # Création du dossier si inexistant
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, name)
    print(f"[SAVE] Écriture dans : {file_path}")

    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            # adaptation GPU -> CPU
            if torch.is_tensor(value):
                value = value.cpu().numpy()

            # Écriture
            try:
                f.create_dataset(key, data=value)
                shape_str = str(value.shape) if hasattr(value, 'shape') else str(len(value))
                print(f"   -> Dataset '{key}' sauvegardé. Shape: {shape_str}")
            except Exception as e:
                print(f"   [ERREUR] Échec écriture '{key}': {e}")

    if csv is not None:
        # On remplace l'extension .hdf5 par .csv pour le nom
        csv_name = name.replace(".hdf5", ".csv")
        csv_path = os.path.join(output_dir, csv_name)
        try:
            csv.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"   [ERREUR] Échec écriture CSV: {e}")




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
    end_idx   = torch.where(has_signal, end_idx, torch.zeros_like(end_idx))

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

        # sub_batch extrait une portion. On force .float() pour le calcul
        sub_batch = tracings[idx_tensor, :, :len_in].float()

        # Initialisation ou réutilisation du module Resample
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
        torch.cuda.empty_cache() 

    data['tracings'] = None 
    del tracings
    return new_tracings



"""
    Méthode de la partie 2
"""
# TODO faux, car à l'interieur du signal on peut avoir des 0, il faut slice la sous sequence plutot !
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
    # On crée un masque booléen pour identifier le signal utile sur chaque canal indépendemment
    mask = (chunk.abs() > eps)

    # Calcul du nombre de points actifs par canal
    count = mask.sum(dim=2, keepdim=True).float().clamp_(min=1.0)

    # On calcule la somme pondérée par le masque
    mean = torch.sum(chunk * mask, dim=2, keepdim=True).div_(count)

    #  Calcul de la Variance
    res = chunk.sub(mean) 

    # Elévation au carré et application du masque in-place
    res.pow_(2).mul_(mask)

    std = res.sum(dim=2, keepdim=True).div_(count).sqrt_().clamp_(min=eps)

    # Libèration de la mémoire de travail
    del res 

    # Normalisation finale In-Place
    chunk.sub_(mean)
    chunk.div_(std)
    chunk.mul_(mask)

    # Nettoyage des petits tenseurs intermédiaires
    del mask, mean, std, count
    
    return chunk



def add_bilateral_padding(tracings, target_size):
    """
    Centre le signal actif dans une fenêtre de taille target_size.
    Version vectorisée, mémoire-friendly.

    Args:
        tracings (torch.Tensor): (N, C, T) ou (C, T).
        target_size (int): Taille cible.

    Returns:
        tuple:
            - torch.Tensor: (N, C, target_size) paddé avec signal centré.
            - torch.Tensor: (N,) longueurs originales des signaux actifs.
    """
    # Sécurité des dimensions
    if tracings.dim() == 2:
        tracings = tracings.unsqueeze(0)  # (1, C, T)

    N, C, T = tracings.shape
    device = tracings.device

    # Détection des bornes actives
    start_idxs, end_idxs = get_active_boundaries(tracings)  # (N,), (N,)
    lengths = end_idxs - start_idxs  # (N,)

    # Offset pour centrer
    offsets = ((target_size - lengths) // 2).clamp(min=0)  # (N,)

    # Préparer le tensor de sortie
    new_tracing = torch.zeros((N, C, target_size), dtype=tracings.dtype, device=device)

    # 4) Injection vectorisée
    # On filtre les signaux valides
    valid = lengths > 0
    if valid.any():
        idxs = torch.arange(N, device=device)[valid]
        for i in idxs:
            l = lengths[i].item()
            o = offsets[i].item()
            s = start_idxs[i].item()
            e = end_idxs[i].item()

            new_tracing[i, :, o:o+l] = tracings[i, :, s:e]

    return new_tracing, lengths
