import os
import argparse
import time
import glob
import sys
import numpy as np
import pandas as pd
import h5py
import multiprocessing
from tqdm import tqdm



def normalize_id(val):
    """
        Normalise une valeur d'identifiant (bytes, int ou str) en une chaîne de caractères propre.

        Cette méthode gère le décodage des bytes (fréquent avec h5py) et nettoie
        les espaces superflus pour garantir la cohérence des jointures (merge).

        Args:
            val (Union[bytes, int, str]): La valeur brute de l'identifiant extraite 
                du CSV ou du fichier HDF5.

        Returns:
            str: L'identifiant normalisé en format string UTF-8 sans espaces.
        """
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val).strip()


def scan_sources_and_map_indices(input_dir):
    """
        Réalise un inventaire global en croisant les métadonnées CSV et les index physiques HDF5.

        1. Lit tous les fichiers CSV du dossier pour obtenir les métadonnées (ex: patient_id).
        2. Ouvre chaque fichier HDF5 pour mapper chaque `exam_id` à son index numérique (row index).
        3. Effectue une jointure interne (Inner Join) pour ne conserver que les échantillons
        présents à la fois dans les métadonnées et dans les données brutes.

        Args:
            input_dir (str): Chemin du répertoire contenant les paires de fichiers .csv et .hdf5.

        Returns:
            pd.DataFrame: Un DataFrame "Inventaire" validé contenant les colonnes :
                - toutes les colonnes des CSV originaux (ex: patient_id, age, label...)
                - source_h5_path (str): Chemin absolu du fichier HDF5 source.
                - h5_idx_src (int): L'index de la ligne correspondant à l'exam_id dans le fichier source.

        Raises:
            ValueError: Si aucun fichier CSV n'est trouvé dans le dossier d'entrée.
        """
    print("--- 1. Scan des CSV (Métadonnées) ---")
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    df_list = []
    for f in tqdm(csv_files, desc="Lecture CSVs"):
        try:
            temp_df = pd.read_csv(f)
            # On normalise l'ID tout de suite pour la fusion future
            temp_df['exam_id'] = temp_df['exam_id'].apply(normalize_id)
            df_list.append(temp_df)
        except Exception as e:
            print(f"[Warning] CSV illisible {f}: {e}")

    if not df_list:
        raise ValueError("Aucun CSV trouvé !")
    
    full_csv = pd.concat(df_list, ignore_index=True)
    # On enlève source_h5_path du CSV, car on va le redéfinir proprement
    full_csv = full_csv.drop(columns=['source_h5_path'])

    print(f"-> {len(full_csv)} entrées trouvées dans les CSV.")

    print("\n--- 2. Scan des HDF5 (Indexation réelle) ---")
    h5_files = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))

    map_records = [] # Liste de dicts {exam_id, source_path, h5_idx}

    for h5_path in tqdm(h5_files, desc="Mapping HDF5 IDs"):
        try:
            with h5py.File(h5_path, 'r') as f:
                # Lecture rapide de tous les IDs
                raw_ids = f['exam_id'][:]

                # Création du mapping pour ce fichier
                for idx, raw_val in enumerate(raw_ids):
                    map_records.append({
                        'exam_id': normalize_id(raw_val),
                        'source_h5_path': h5_path, # Chemin absolu
                        'h5_idx_src': idx          # L'index integer pour h5py
                    })
        except Exception as e:
            print(f"[Error] Lecture HDF5 {h5_path}: {e}")

    df_mapping = pd.DataFrame(map_records)
    print(f"-> {len(df_mapping)} IDs trouvés physiquement dans les HDF5.")

    print("\n--- 3. Fusion et Validation (Inner Join) ---")
    # On ne garde que l'intersection parfaite.
    # Le CSV apporte les métadonnées (patient_id), le Mapping apporte l'adresse physique (path + index).
    merged_df = pd.merge(full_csv, df_mapping, on='exam_id', how='inner')

    print(f"-> {len(merged_df)} paires valides (CSV + HDF5) prêtes au traitement.")
    diff = len(full_csv) - len(merged_df)
    if diff > 0:
        print(f"[Info] {diff} lignes du CSV ont été ignorées car absentes des HDF5.")

    return merged_df


def write_shard_task(task_config):
    """
    Fonction Worker exécutée par un processus enfant pour générer un unique 'shard' (fichier HDF5 + CSV).

    1. Elle reçoit un sous-ensemble du DataFrame global (l'inventaire).
    2. Elle groupe les requêtes par fichier source HDF5 pour minimiser les ouvertures/fermetures de fichiers.
    3. Elle lit les données (tracings) et les IDs de manière vectorisée.
    4. Elle écrit le nouveau fichier HDF5 et le CSV associé.

    Args:
        task_config (dict): Un dictionnaire de configuration contenant :
            - 'output_path' (str): Le chemin complet du fichier HDF5 à créer.
            - 'inventory_df' (pd.DataFrame): Le sous-ensemble du DataFrame contenant 
              la liste des examens à écrire dans ce shard, incluant la colonne 'h5_idx_src'.

    Returns:
        str: Un message de statut indiquant le succès ("OK: ...") ou l'échec ("CRASH: ...").
             En cas de succès, inclut le nombre d'échantillons écrits.
    
    Raises:
        AssertionError: Si une incohérence est détectée entre la taille des données lues
        et la taille des métadonnées attendues avant l'écriture.
    """
    shard_path = task_config['output_path']
    # Le DataFrame reçu contient déjà 'source_h5_path' et 'h5_idx_src' corrects.
    inventory_df = task_config['inventory_df'].copy()

    # Réinitialiser l'index pour avoir un ordre 0..N propre pour ce shard
    inventory_df = inventory_df.reset_index(drop=True)
    inventory_df['shard_local_idx'] = inventory_df.index

    total_samples = len(inventory_df)
    if total_samples == 0:
        return "Shard Vide"

    # Conteneurs RAM
    tracings_data = None
    exam_ids_data = [None] * total_samples

    # On groupe par fichier source pour ne l'ouvrir qu'une fois
    grouped = inventory_df.groupby('source_h5_path')

    # Détection dynamique de la shape
    time_dim = 0
    n_channels = 0

    # Gestion des erreurs locales
    failed_indices = []

    for source_path, group in grouped:
        try:
            # Tri par index source croissant pour optimiser la lecture disque
            group_sorted = group.sort_values('h5_idx_src')

            src_indices = group_sorted['h5_idx_src'].values
            dest_indices = group_sorted['shard_local_idx'].values

            with h5py.File(source_path, 'r') as f_in:
                # Init buffer si premier passage
                if tracings_data is None:
                    dset_shape = f_in['tracings'].shape
                    time_dim = dset_shape[1]
                    n_channels = dset_shape[2] if len(dset_shape) > 2 else 12
                    tracings_data = np.zeros((total_samples, time_dim, n_channels), dtype='f4')

                # Lecture vectorisée
                raw_tracings = f_in['tracings'][src_indices]
                raw_ids = f_in['exam_id'][src_indices]

                # Écriture RAM
                tracings_data[dest_indices] = raw_tracings

                # Conversion ID
                for i, raw_id in enumerate(raw_ids):
                    target_idx = dest_indices[i]
                    exam_ids_data[target_idx] = normalize_id(raw_id)

        except Exception as e:
            # Si un fichier source plante, on note les indices locaux échoués
            failed_idx = group['shard_local_idx'].values
            failed_indices.extend(failed_idx)

    # --- Filtrage post-lecture (si échecs I/O) ---
    valid_mask = np.ones(total_samples, dtype=bool)
    if failed_indices:
        valid_mask[failed_indices] = False

    # On vérifie aussi qu'on n'a pas de None dans les IDs (double check)
    for i in range(total_samples):
        if exam_ids_data[i] is None:
            valid_mask[i] = False

    final_tracings = tracings_data[valid_mask] if tracings_data is not None else np.array([])
    final_ids_list = [exam_ids_data[i] for i in range(total_samples) if valid_mask[i]]
    final_df = inventory_df[valid_mask].copy()

    if len(final_df) == 0:
        return f"Shard {os.path.basename(shard_path)} : VIDE après filtrage erreurs."

    # --- Assertions de Sécurité ---
    # C'est ce qui garantit que le CSV correspond au HDF5
    assert len(final_df) == len(final_ids_list), "Mismatch taille CSV vs IDs"
    assert len(final_df) == final_tracings.shape[0], "Mismatch taille CSV vs Tracings"

    # --- Écriture ---
    try:
        with h5py.File(shard_path, 'w') as f_out:
            # Chunking activé : (1, time, 12) permet un accès rapide sample par sample
            chunk_shape = (1, time_dim, n_channels)

            f_out.create_dataset('tracings', 
                                 data=final_tracings, 
                                 chunks=chunk_shape, # Optimisation Training
                                 compression="gzip", 
                                 compression_opts=4)

            dt_str = h5py.string_dtype(encoding='utf-8')
            f_out.create_dataset('exam_id', 
                                 data=np.array(final_ids_list, dtype=object), 
                                 dtype=dt_str)

        # --- Préparation CSV Final ---
        csv_path = shard_path.replace('.hdf5', '.csv')

        # 1. Mise à jour du chemin source : pointe maintenant vers le nouveau shard
        final_df['source_h5_path'] = os.path.basename(shard_path)

        # 2.  Nettoyage colonnes techniques
        if 'split' in final_df.columns:
            final_df = final_df.drop(columns=['split'])
        cols_drop = ['shard_local_idx', 'h5_idx_src']
        final_df = final_df.drop(columns=[c for c in cols_drop if c in final_df.columns])

        final_df.to_csv(csv_path, index=False)

        return f"OK: {os.path.basename(shard_path)} ({len(final_df)} samples)"

    except Exception as e:
        return f"CRASH: {os.path.basename(shard_path)} - {e}"



def run(args):
    """
    Orchestre la pipeline complete de préparation des données.

    1. Scan & Map : Appelle `scan_sources_and_map_indices` pour indexer les données.
    2. Split Patient-Aware : Divise les données en Train/Val/Test en s'assurant
       qu'un même patient ne se retrouve pas dans plusieurs splits (fuite de données).
    3. Sharding : Découpe les données de chaque split en blocs de taille fixe (`shard_size`).
    4. Multiprocessing : Lance un pool de workers pour écrire les fichiers physiques en parallèle.

    Args:
        args (argparse.Namespace): Les arguments de la ligne de commande parsés, contenant :
            - args.input (str): Dossier source.
            - args.output (str): Dossier de destination.
            - args.shard_size (int): Taille des fichiers de sortie.
            - args.train_prct (float): Ratio pour le train set.
            - args.val_prct (float): Ratio pour le validation set.
            - args.workers (int): Nombre de cœurs CPU à utiliser.

    Returns:
        None: La fonction imprime le progrès et le rapport final dans la sortie standard.
    """
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    # 1. SCAN & MAPPING (Parent Process)
    # C'est ici qu'on fait le mapping exam_id -> h5_idx_src une fois
    try:
        full_df = scan_sources_and_map_indices(args.input)
    except Exception as e:
        print(f"Arrêt critique : {e}")
        return

    # 2. SPLIT PAR PATIENT
    print("\n--- Split Train/Val/Test (Patient Aware) ---")
    patients = full_df['patient_id'].unique()
    np.random.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * args.train_prct)
    n_val = int(n_total * args.val_prct)

    pats_train = set(patients[:n_train])
    pats_val = set(patients[n_train:n_train+n_val])

    full_df['split'] = full_df['patient_id'].apply(
        lambda x: 'train' if x in pats_train else ('val' if x in pats_val else 'test')
    )
    
    print("Répartition des échantillons :")
    print(full_df['split'].value_counts())

    # 3. PRÉPARATION DES TÂCHES
    print(f"\n--- Génération des Tâches (Shard size: {args.shard_size}) ---")

    # Mélange global final (Shuffling)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    tasks = []

    for split in ['train', 'val', 'test']:
        split_df = full_df[full_df['split'] == split]
        if split_df.empty: continue

        # Création dossier
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)

        # Découpage en shards
        n_shards = int(np.ceil(len(split_df) / args.shard_size))
        chunks = np.array_split(split_df, n_shards)

        for i, chunk_df in enumerate(chunks):
            # chunk_df contient déjà 'h5_idx_src', le worker n'a qu'à lire bêtement
            shard_name = f"{split}_shard_{i:03d}.hdf5"
            tasks.append({
                'output_path': os.path.join(out_dir, shard_name),
                'inventory_df': chunk_df
            })

    print(f"{len(tasks)} tâches prêtes.")

    # 4. EXÉCUTION
    print(f"\n--- Démarrage Multiprocessing ({args.workers} workers) ---")
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(write_shard_task, tasks), total=len(tasks)))

    # 5. Rapport
    success = [r for r in results if r.startswith("OK")]
    fails = [r for r in results if not r.startswith("OK")]
    
    print(f"\nTraitement terminé en {int((time.time()-start_time)//60)} min.")
    print(f"Succès : {len(success)} shards.")
    if fails:
        print(f"Erreurs : {len(fails)}")
        for f in fails[:5]: print(f)



def main():
    parser = argparse.ArgumentParser(
        description="Outil de Shuffling et Sharding Offline pour Datasets HDF5 Massifs."
    )

    parser.add_argument('-i', '--input', type=str, default='../../../data/15_prct/',
                        help='Dossier source contenant les paires .hdf5 et .csv')

    parser.add_argument('-o', '--output', type=str, default='../output/final_data/',
                        help='Dossier destination où seront créés les nouveaux fichiers')

    parser.add_argument('-s', '--shard_size', type=int, default=10000,
                        help='Nombre cible d\'échantillons par fichier de sortie (Défaut: 10k)')

    parser.add_argument('--train_prct', type=float, default=0.80, help='Ratio Train (0-1)')
    parser.add_argument('--val_prct', type=float, default=0.10, help='Ratio Validation (0-1)')
    
    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count()-1,
                        help='Nombre de processus parallèles (Défaut: CPU-1)')

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
