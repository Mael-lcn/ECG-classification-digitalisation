import os
import argparse
import time
import gc
import torch
import random
import threading
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

from aux import *



# --- CONFIGURATION GLOBALE ---
TARGET_FREQ = 400
MAX_TEMPS = 144
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


# Limite de sécurité VRAM (en Go). Le GPU fait environ 9.6, on réserve 1Go pour le système/overhead
VRAM_LIMIT_GB = 8.6 
current_vram_usage = 0.0
vram_lock = threading.Condition()


def estimate_vram_gb(path_h5):
    """
    Estime la mémoire nécessaire pour un file.
    """
    try:
        with h5py.File(path_h5, 'r') as f:
            shape = f['tracings'].shape
            # N * C * T * 4 bytes / 1024^3
            size_gb = (torch.tensor(shape).prod().item() * 4) / (1024**3)
            # Facteur 2 car resampling + z-norm créent des copies temporaires
            return max(size_gb * 2, 0.2)
    except:
        return 0.5


def unified_worker(task, output):
    """
    Orchestrateur de traitement ECG avec gestion de mémoire par segmentation (Chunks).

    Cette fonction traite des fichiers HDF5 massifs (5Go+) sur une VRAM limitée (9.6Go) 
    en suivant une stratégie de "chargement unique et traitement par morceaux". 
    Elle garantit l'alignement entre les signaux (HDF5) et les métadonnées (CSV) 
    en utilisant les 'exam_id' comme clés de synchronisation.

    Processus :
    1. Admission contrôlée par budget VRAM (Lock) pour éviter les crashs simultanés.
    2. Chargement HDF5 (GPU) et CSV (CPU).
    3. Découpage en 3 chunks via des 'vues' (narrow) pour éviter la duplication inutile.
    4. Réalignement dynamique du CSV pour chaque chunk à partir des exam_id.
    5. Traitement (Resampling + Z-Norm) et reconstruction finale sécurisée.
    6. Libération agressive de la mémoire (GC + Cache CUDA).

    Args:
        task (tuple): (mode, name, path) où path est (path_hdf5, path_csv).
        output (str): Répertoire de destination pour les résultats.
    """
    global current_vram_usage
    mode, name, path = task
    path_h5 = path[0]

    # On évalue le poids du fichier pour ne pas dépasser la capacité du GPU
    cost = estimate_vram_gb(path_h5)

    with vram_lock:
        while current_vram_usage + cost > VRAM_LIMIT_GB:
            vram_lock.wait()
        current_vram_usage += cost

    dataset, csv_full = None, None
    try:
        # dataset['tracings'] est envoyé sur GPU, 'exam_id' reste sur CPU (via load)
        dataset, csv_full = load(path, use_csv=True, to_gpu=True)
        tracings = dataset['tracings']
        exam_ids = dataset['exam_id']
        N = tracings.shape[0]

        # Préparation du CSV : Indexation par 'exam_id' pour un réalignement rapide
        csv_full['exam_id'] = csv_full['exam_id'].astype(str)
        csv_indexed = csv_full.set_index('exam_id')

        # Traitement par chunks
        # Définition des bornes pour diviser le batch en 3 parties égales
        indices = [0, N // 3, (2 * N) // 3, N]
        processed_chunks = []

        for i in range(3):
            start, end = indices[i], indices[i+1]
            if start == end: continue

            # .narrow() crée une vue
            chunk_tracings = tracings.narrow(0, start, end - start)
            chunk_ids_raw = exam_ids[start:end]

            # Conversion des IDs du segment pour l'extraction du CSV
            chunk_ids_str = [
                eid.decode('utf-8') if isinstance(eid, (bytes, bytearray)) else str(eid) 
                for eid in chunk_ids_raw
            ]

            # Extraction des métadonnées correspondant au chunk
            chunk_csv = csv_indexed.reindex(chunk_ids_str).reset_index()

            if mode == 'D1':
                temp_data = {'tracings': chunk_tracings, 'exam_id': chunk_ids_raw}
                # Le resampling crée un nouveau tenseur
                chunk_result = re_sampling(temp_data, chunk_csv, fo=TARGET_FREQ)
            else:
                chunk_result = chunk_tracings

            # Normalisation Z-score in-place
            z_norm(chunk_result)

            # Stockage du résultat intermédiaire du chunk
            processed_chunks.append(chunk_result)

            # Nettoyage des objets intermédiaires pour libérer de l'espace de travail
            del chunk_ids_str, chunk_csv
            torch.cuda.empty_cache()
            print(f"   -> [{name}] Bloc {i+1}/3 traité")

        dataset['tracings'] = None 
        del tracings
        torch.cuda.empty_cache()

        # Cherche l'axe T le plus long dans les reuslats
        all_t_sizes = [c.shape[2] for c in processed_chunks]
        max_t = max(all_t_sizes)

        # Pad les signaux pour etre homogène sur T
        for i in range(len(processed_chunks)):
            current_t = processed_chunks[i].shape[2]

            if current_t < max_t:
                # Calcul de la différence manquante
                diff = max_t - current_t
                N_chunk, C, _ = processed_chunks[i].shape

                # Création du "patch" de zéros sur le même device (GPU)
                padding_patch = torch.zeros((N_chunk, C, diff), 
                                        device=processed_chunks[i].device, 
                                        dtype=processed_chunks[i].dtype)

                # On allonge le chunk sur la dimension 2 (le temps)
                # On remplace l'élément dans la liste par sa version rallongée
                processed_chunks[i] = torch.cat([processed_chunks[i], padding_patch], dim=2)

                # Nettoyage immédiat du patch
                del padding_patch

        # Fusion finale
        dataset['tracings'] = torch.cat(processed_chunks, dim=0)
        del processed_chunks

        # On réordonne le CSV complet selon l'ordre exact des exam_id du HDF5
        final_h5_ids = [
            eid.decode('utf-8') if isinstance(eid, (bytes, bytearray)) else str(eid) 
            for eid in exam_ids
        ]
        final_csv = csv_indexed.reindex(final_h5_ids).reset_index()

        # Harmonisation des colonnes selon le mode (D1 vs Classification)
        if mode != 'D1':
            final_csv.rename(columns={"normal_ecg": "NSR"}, inplace=True)

        final_csv['length'] = 1

        #Save
        write_results(dataset, final_csv, name, output)

    except Exception as e:
        print(f"   [ERREUR FATALE] {name} : {str(e)}")
        raise e

    finally:
        #Nettoyage
        if dataset is not None: del dataset
        if csv_full is not None: del csv_full
        gc.collect()
        torch.cuda.empty_cache()

        # Libération de l'admission VRAM pour les threads en attente
        with vram_lock:
            current_vram_usage -= cost
            vram_lock.notify_all()



def run(args):
    """
    Orchestre le traitement global en mélangeant les datasets pour une efficacité maximale.
    """
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    # Collecte et étiquetage des tâches
    print("Indexation des fichiers...")
    all_tasks = []

    d1_files = collect_files(args.dataset1)
    for name, path in d1_files.items():
        all_tasks.append(('D1', name, path))

    d2_files = collect_files(args.dataset2)
    for name, path in d2_files.items():
        all_tasks.append(('D2', name, path))

    if not all_tasks:
        print("Aucun fichier trouvé.")
        return

    # 2. Mélange aléatoire
    # Permet de lisser la charge GPU en mixant gros fichiers (D1) et petits (D2)
    random.shuffle(all_tasks)

    print(f"Traitement de {len(all_tasks)} fichiers avec {args.workers} threads.")
    print(f"Gestion dynamique de la VRAM (Limite: {VRAM_LIMIT_GB} GB).")

    # Exécution parallèle
    worker_func = partial(unified_worker, output=args.output)

    with ThreadPool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, all_tasks), 
                  total=len(all_tasks), 
                  desc='Global Preprocessing'))

    elapsed = time.time() - start_time
    print(f"\nPipeline terminée en {int(elapsed // 60)}m {int(elapsed % 60)}s.")



def main():
    parser = argparse.ArgumentParser(description="Pipeline ECG Mixte avec Budget VRAM")
    parser.add_argument('-d1', '--dataset1', type=str, default='../output/dataset1/')
    parser.add_argument('-d2', '--dataset2', type=str, default='../../../data/15_prct/')
    parser.add_argument('-o', '--output', type=str, default='../output/normalize_data')
    parser.add_argument('-w', '--workers', type=int, default=8, help="Nb de threads (I/O)")

    args = parser.parse_args()

    # Config PyTorch pour éviter la fragmentation CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    run(args)


if __name__ == '__main__':
    main()
