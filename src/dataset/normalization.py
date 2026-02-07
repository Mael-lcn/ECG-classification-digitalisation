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


# Limite de sécurité VRAM (en Go). Le GPU fait environ 10Go, on réserve 2Go pour le système/overhead
VRAM_LIMIT_GB = 8.0 
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
            # Facteur 4 car resampling + z-norm créent des copies temporaires
            return max(size_gb * 2, 0.2)
    except:
        return 0.5


def unified_worker(task, output):
    """
    Worker polyvalent avec gestion dynamique de l'admission GPU.
    """
    global current_vram_usage
    mode, name, path = task
    path_h5 = path[0]

    # 1. Estimation du coût VRAM
    cost = estimate_vram_gb(path_h5)
    print("QUEUE", f"{name} | mode={mode} | estimated={cost:.2f} GB")


    # 2. Admission contrôlée
    with vram_lock:
        while current_vram_usage + cost > VRAM_LIMIT_GB:
            vram_lock.wait()    # Attend qu'un autre thread libère de la place
        current_vram_usage += cost

    print("START"f"{name} | allocated={cost:.2f} GB | "f"total={current_vram_usage:.2f}/{VRAM_LIMIT_GB} GB")

    dataset, csv = None, None
    try:
        # 3. Pipeline de traitement
        dataset, csv = load(path, use_csv=True, to_gpu=True)

        if mode == 'D1':
            dataset['tracings'] = re_sampling(dataset, csv, fo=TARGET_FREQ)
            csv['frequences'] = TARGET_FREQ
        else:
            # Dataset 2 : Normalisation simple
            if csv is not None and "normal_ecg" in csv.columns:
                csv = csv.rename(columns={"normal_ecg": "NSR"})

        dataset['tracings'] = z_norm(dataset['tracings'])
        dataset['tracings'], lengths = add_bilateral_padding(dataset['tracings'], MAX_SIGNAL_LENGTH)
        csv['length'] = lengths     # Créer une colonne avec la lg utile du signal pour la suite

        write_results(dataset, csv, name, output)

    except Exception as e:
        print(f"\n[Erreur] {name} ({mode}) : {e}")
    finally:
        # Libération de la mémoire et du budget
        if dataset is not None: del dataset
        if csv is not None: del csv
        gc.collect()
        torch.cuda.empty_cache()

        with vram_lock:
            current_vram_usage -= cost
            vram_lock.notify_all()  # Réveille les threads en attente


def run(args):
    """
    Orchestre le traitement global en mélangeant les datasets pour une efficacité maximale.
    """
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    # 1. Collecte et étiquetage des tâches
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

    # 3. Exécution parallèle
    # car le verrou VRAM empêchera le GPU d'exploser
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
