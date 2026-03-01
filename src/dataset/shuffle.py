import os, sys
import argparse
import time
import glob
import numpy as np
import pandas as pd, gc
import h5py
import multiprocessing
from tqdm import tqdm



# Garantit que np.random.shuffle donnera toujours le même résultat
np.random.seed(42)


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
                - trace_file (str): Chemin absolu du fichier HDF5 source.
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
    # On enlève trace_file du CSV, car on va le redéfinir proprement
    full_csv = full_csv.drop(columns=['trace_file'])

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
                        'trace_file': h5_path, # Chemin absolu
                        'h5_idx_src': idx          # L'index integer pour h5py
                    })
        except Exception as e:
            print(f"[Error] Lecture HDF5 {h5_path}: {e}")

    df_mapping = pd.DataFrame(map_records)
    print(f"-> {len(df_mapping)} IDs trouvés physiquement dans les HDF5.")

    print("\n--- 3. Fusion et Validation (Inner Join) ---")
    # On ne garde que l'intersection parfaite
    # Le CSV apporte les métadonnées (patient_id), le Mapping apporte l'adresse physique (path + index)
    merged_df = pd.merge(full_csv, df_mapping, on='exam_id', how='inner')

    print(f"-> {len(merged_df)} paires valides (CSV + HDF5) prêtes au traitement.")
    diff = len(full_csv) - len(merged_df)
    if diff > 0:
        print(f"[Info] {diff} lignes du CSV ont été ignorées car absentes des HDF5.")

    return merged_df



def write_shard_task(task_config):
    """
    Worker ultra-optimisé pour la RAM (Streaming Direct-to-Disk).
    
    Format : PyTorch (B, C, T).
    Optimisations :
    - Pas de buffer NumPy géant en RAM (Écriture directe dans HDF5).
    - Tri et déduplication des index pour éviter les erreurs H5Py.
    - Garbage collection explicite entre chaque source.
    """
    shard_path = task_config['output_path']
    inventory_df = task_config['inventory_df'].copy().reset_index(drop=True)
    inventory_df['shard_local_idx'] = inventory_df.index
    total_samples = len(inventory_df)

    if total_samples == 0:
        return f"SKIP: {os.path.basename(shard_path)}"

    # --- 1. SCAN DYNAMIQUE (Passe 1) ---
    # On identifie le T_max pour pré-allouer le fichier sur disque
    local_max_time = 0
    unique_sources = inventory_df['trace_file'].unique()

    try:
        for src in unique_sources:
            with h5py.File(src, 'r') as f:
                t_dim = f['tracings'].shape[2] # Format (N, 12, T)
                if t_dim > local_max_time:
                    local_max_time = t_dim
    except Exception as e:
        return f"CRASH SCAN: {e}"

    # --- 2. PRÉ-ALLOCATION HDF5 ---
    n_channels = 12
    try:
        # On crée le fichier et le dataset immédiatement
        f_out = h5py.File(shard_path, 'w')

        # Dataset pré-rempli de zéros par HDF5 sur le disque (fillvalue=0)
        dset_tracings = f_out.create_dataset(
            'tracings', 
            shape=(total_samples, n_channels, local_max_time),
            dtype='f4',
            chunks=(1, n_channels, local_max_time),
            compression="gzip",
            compression_opts=4,
            fillvalue=0.0 
        )

        # Buffer temporaire pour les IDs
        exam_ids_data = [None] * total_samples

    except Exception as e:
        if 'f_out' in locals(): f_out.close()
        return f"CRASH INIT: {e}"

    # 3. STREAMING DES DONNÉES
    grouped = inventory_df.groupby('trace_file')
    
    for source_path, group in grouped:
        try:
            # Indices sources et destinations
            req_src_idx = group['h5_idx_src'].values
            req_dest_idx = group['shard_local_idx'].values

            # H5Py exige des index triés et uniques pour les lectures vectorisées
            unique_src_idx, inverse_map = np.unique(req_src_idx, return_inverse=True)

            with h5py.File(source_path, 'r') as f_in:
                # Lecture du petit bloc
                # On force f4 (float32)
                raw_chunk = f_in['tracings'][unique_src_idx].astype('f4')
                raw_ids = f_in['exam_id'][unique_src_idx]

                # Reconstruction du batch
                batch_data = raw_chunk[inverse_map]
                batch_ids = raw_ids[inverse_map]

                t_src = batch_data.shape[2]

                # ÉCRITURE DIRECTE SUR DISQUE
                # Le padding est automatique : on n'écrit que sur :t_src
                # HDF5 garde le reste à 0 grâce au fillvalue.
                for i, d_idx in enumerate(req_dest_idx):
                    dset_tracings[d_idx, :, :t_src] = batch_data[i]
                    exam_ids_data[d_idx] = normalize_id(batch_ids[i])

                # Libération RAM immédiate
                del raw_chunk, raw_ids, batch_data, batch_ids
                gc.collect()

        except Exception as e:
            # On log l'erreur mais on continue pour les autres sources
            print(f"[ERR] Source {os.path.basename(source_path)} ignorée: {e}", file=sys.stderr)

    # --- 4. FINALISATION ---
    try:
        # Écriture des IDs à la fin
        # On filtre les éventuels None (si une source a crashé)
        valid_ids = [id_ if id_ is not None else "ERROR" for id_ in exam_ids_data]
        
        dt_str = h5py.string_dtype(encoding='utf-8')
        f_out.create_dataset('exam_id', data=np.array(valid_ids, dtype=object), dtype=dt_str)
        
        f_out.close() # Fermeture propre du fichier HDF5

        # Écriture du CSV compagnon
        csv_path = shard_path.replace('.hdf5', '.csv')
        # On ne garde que les lignes où l'ID a bien été écrit
        mask_valid = [id_ is not None for id_ in exam_ids_data]
        final_df = inventory_df[mask_valid].copy()
        
        final_df['trace_file'] = os.path.basename(shard_path)
        final_df = final_df.drop(columns=['shard_local_idx', 'h5_idx_src', 'split'], errors='ignore')
        final_df.to_csv(csv_path, index=False)

        return f"OK: {os.path.basename(shard_path)} (T={local_max_time})"

    except Exception as e:
        if 'f_out' in locals(): f_out.close()
        return f"CRASH FINAL: {e}"



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

    # 1. Scan et mapping
    # C'est ici qu'on fait le mapping exam_id -> h5_idx_src une fois
    try:
        full_df = scan_sources_and_map_indices(args.input)
    except Exception as e:
        print(f"Arrêt critique : {e}")
        return

    # Split par patient
    print("\n--- Split Train/Val/Test ---")
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

    # Préparation des taches
    print(f"\n--- Génération des Tâches (Shard size: {args.shard_size}) ---")

    # Mélange global final
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

    print(f"\n--- Démarrage Multiprocessing ({args.workers} workers) ---")
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(write_shard_task, tasks), total=len(tasks)))


    # Rapport
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

    parser.add_argument('-i', '--input', type=str, default='../../../output/normalize_data/',
                        help='Dossier source contenant les paires .hdf5 et .csv')

    parser.add_argument('-o', '--output', type=str, default='../../../output/final_data/',
                        help='Dossier destination où seront créés les nouveaux fichiers')

    parser.add_argument('-s', '--shard_size', type=int, default=8000,
                        help='Nombre cible d\'échantillons par fichier de sortie (Défaut: 8K)')

    parser.add_argument('--train_prct', type=float, default=0.80, help='Ratio Train (0-1)')
    parser.add_argument('--val_prct', type=float, default=0.10, help='Ratio Validation (0-1)')

    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count()-1,
                        help='Nombre de processus parallèles (Défaut: CPU-1)')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
