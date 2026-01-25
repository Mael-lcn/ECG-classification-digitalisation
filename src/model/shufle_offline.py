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

# Configuration des imports relatifs
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(project_root))


# Nombre d'échantillons à garder en RAM avant d'écrire sur le disque.
WRITE_BUFFER_SIZE = 5000


def write_shard_task(task_config):
    """
    Fonction exécutée par un processus Worker.
    Elle crée un fichier HDF5 (Shard) et va pêcher les données éparpillées 
    dans les fichiers sources pour les regrouper ici.

    Args:
        task_config (Dict): Dictionnaire contenant:
            - 'output_path': Chemin complet du fichier à créer.
            - 'inventory_df': DataFrame partiel (les lignes à écrire dans ce shard).

    Returns:
        str: Message de statut pour le logging.
    """
    shard_path = task_config['output_path']
    inventory_df = task_config['inventory_df'] 

    # Si le dataframe est vide, inutile de créer un fichier
    if inventory_df.empty:
        return f"Shard {os.path.basename(shard_path)} : Vide (Ignoré)"

    # On regroupe les demandes par fichier source.
    # Cela permet d'ouvrir 'source_A.h5', de tout lire ce qu'il faut dedans, 
    grouped_sources = inventory_df.groupby('source_h5_path')

    total_written = 0
    f_out = None

    try:
        # Ouverture du fichier de destination en écriture
        f_out = h5py.File(shard_path, 'w')

        # On ne connaît pas encore la dimension temporelle D.
        dset_tracings = None 
        dset_ids = None

        # Le type pour les chaînes de caractères (IDs) en HDF5
        dt_str = h5py.string_dtype(encoding='utf-8')

        buffer_tracings = []
        buffer_ids = []

        # --- BOUCLE PRINCIPALE DE LECTURE ---
        for source_path, sub_df in grouped_sources:
            # On récupère les indices HDF5 (ligne i) qu'on doit copier
            indices_to_grab = sorted(sub_df['h5_idx'].values)
            if not indices_to_grab: 
                continue

            try:
                with h5py.File(source_path, 'r') as f_in:
                    
                    # 1. détection de dimmension
                    if dset_tracings is None:
                        # On inspecte le premier fichier source valide
                        source_shape = f_in['tracings'].shape
                        D = source_shape[1]
                        
                        # Création des datasets extensibles
                        # chunks=(128, D, nb_leads) : Optimise la lecture future par batch de 128
                        dset_tracings = f_out.create_dataset('tracings', 
                                           shape=(0, D, 12), 
                                           maxshape=(None, D, 12), 
                                           dtype='f4', # Float32 pour Deep Learning
                                           chunks=(128, D, 12)) 

                        dset_ids = f_out.create_dataset('exam_id', 
                                           shape=(0,), 
                                           maxshape=(None,), 
                                           dtype=dt_str)

                    data_tracings = f_in['tracings'][indices_to_grab]
                    data_ids = f_in['exam_id'][indices_to_grab]

                    # Décodage des bytes (b'ID123') en string ('ID123') si nécessaire
                    if data_ids.dtype.kind == 'S': 
                        data_ids = [x.decode('utf-8') for x in data_ids]

                    # Ajout au buffer RAM
                    buffer_tracings.append(data_tracings)
                    buffer_ids.append(data_ids)
  
                    # 3. Flush
                    # On vide le buffer si il est plein pour libérer la RAM
                    current_buffer_len = sum(len(b) for b in buffer_tracings)
                    if dset_tracings is not None and current_buffer_len >= WRITE_BUFFER_SIZE:
                        flush_buffer(dset_tracings, dset_ids, buffer_tracings, buffer_ids)
                        # Reset des buffers
                        buffer_tracings, buffer_ids = [], []

            except Exception as e:
                # On log l'erreur mais on ne crash pas tout le process pour un fichier corrompu
                print(f"[Worker Error] Impossible de lire {source_path}: {e}")
                continue

        # On écrit ce qu'il reste dans les buffers à la fin du traitement
        if buffer_tracings and dset_tracings is not None:
            flush_buffer(dset_tracings, dset_ids, buffer_tracings, buffer_ids)
            total_written = dset_tracings.shape[0]

        # Sauvegarde du CSV compagnon (Méta-données alignées)
        csv_out_path = shard_path.replace('.hdf5', '.csv')
        inventory_df.to_csv(csv_out_path, index=False)

    except Exception as global_e:
        return f"CRITICAL ERROR sur {shard_path}: {global_e}"

    finally:
        # Sécurité : On s'assure que le fichier est fermé quoi qu'il arrive
        if f_out:
            f_out.close()
    
    return f"Shard {os.path.basename(shard_path)} : {total_written} samples générés."


def flush_buffer(dset_tr, dset_id, buf_tr, buf_id):
    """
    Fonction utilitaire interne pour écrire physiquement les buffers dans le fichier HDF5.
    Redimensionne le dataset et ajoute les données à la fin.
    """
    if not buf_tr: return

    # Fusion des listes en un seul gros tableau numpy
    arr_tr = np.concatenate(buf_tr, axis=0)
    arr_id = np.concatenate(buf_id, axis=0)

    n_new = arr_tr.shape[0]

    # Calcul des nouvelles dimensions
    current_size = dset_tr.shape[0]
    new_size = current_size + n_new

    # Extension du fichier
    dset_tr.resize(new_size, axis=0)
    dset_id.resize(new_size, axis=0)

    # Copie des données
    dset_tr[current_size:] = arr_tr
    dset_id[current_size:] = arr_id


def run(args):
    """
    Fonction principale d'orchestration.
    1. Scanne les fichiers.
    2. Répartit les patients (Train/Val/Test).
    3. Lance les workers pour créer les fichiers shards.
    """
    start_time = time.time()

    os.makedirs(args.output, exist_ok=True)

    print("\n--- Phase 1: Scan et Inventaire ---")
    h5_files = sorted(glob.glob(os.path.join(args.input, '*.hdf5')))

    if not h5_files:
        print("ERREUR: Aucun fichier .hdf5 trouvé dans le dossier input.")
        return

    all_dfs = []
    print("Chargement et alignement des métadonnées CSV...")

    for h_f in tqdm(h5_files, desc="Parsing Files"):
        # On remplace l'extension au lieu de supposer que les listes sont alignées
        c_f = h_f.replace('.hdf5', '.csv')

        if not os.path.exists(c_f):
            print(f"ATTENTION: CSV manquant pour {h_f} -> Ignoré.")
            continue

        # Lecture CSV
        df = pd.read_csv(c_f)

        # Ajout des métadonnées système indispensables
        df['source_h5_path'] = h_f
        df['h5_idx'] = range(len(df))

        all_dfs.append(df)

    if not all_dfs:
        print("ERREUR: Aucune paire valide HDF5/CSV trouvée.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Inventaire terminé. Total : {len(full_df)} échantillons.")

    # ---------------------------------------------------------
    # PHASE 2 : SPLIT PAR PATIENT (Anti-Data Leakage)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Split Train/Val/Test (Patient Aware) ---")
    patients = full_df['patient_id'].unique()

    # Mélange aléatoire des patients
    np.random.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * args.train_prct)
    n_val = int(n_total * args.val_prct)

    # Création des ensembles de patients
    pats_train = set(patients[:n_train])
    pats_val = set(patients[n_train:n_train+n_val])


    # Fonction de mapping
    def get_split(pid):
        if pid in pats_train: return 'train'
        if pid in pats_val: return 'val'
        return 'test'

    # Application du split
    full_df['split'] = full_df['patient_id'].map(get_split)

    print("Répartition des splits :")
    print(full_df['split'].value_counts())

    # ---------------------------------------------------------
    # PHASE 3 : CALCUL DES SHARDS
    # ---------------------------------------------------------
    print(f"\n--- Phase 3: Calcul des Shards (~{args.shard_size} samples/file) ---")

    # Mélange GLOBAL des lignes (Shuffle Offline)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    tasks = []

    for split in ['train', 'val', 'test']:
        split_df = full_df[full_df['split'] == split]
        if len(split_df) == 0: continue

        # Calcul du nombre de fichiers nécessaires
        num_shards = int(np.ceil(len(split_df) / args.shard_size))

        # Division du DataFrame en 'num_shards' morceaux
        chunks = np.array_split(split_df, num_shards)

        for i, chunk in enumerate(chunks):
            out_name = f"{split}_shard_{i:03d}.hdf5"
            tasks.append({
                'output_path': os.path.join(args.output, out_name),
                'inventory_df': chunk
            })

    print(f"Planification : Génération de {len(tasks)} fichiers Shards.")

    # ---------------------------------------------------------
    # PHASE 4 : EXECUTION PARALLELE
    # ---------------------------------------------------------
    print(f"\n--- Phase 4: Exécution avec {args.workers} Workers ---")

    ctx = multiprocessing.get_context('spawn')

    with ctx.Pool(args.workers) as pool:
        _ = list(tqdm(pool.imap_unordered(write_shard_task, tasks), 
                          total=len(tasks), 
                          desc="Processing Shards",
                          unit="file"))

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    print(f"\nsuccès ! Traitement terminé en {minutes} minutes.")



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

    # Validation des arguments
    if args.train_prct + args.val_prct >= 1.0:
        print("ERREUR CONFIG: La somme Train + Val doit être < 1.0 pour laisser de la place au Test.")
        return

    run(args)



if __name__ == '__main__':
    main()
