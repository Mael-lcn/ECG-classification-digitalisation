import os
import argparse
import time
from tqdm import tqdm
import shutil

from functools import partial
import multiprocessing

from aux import *



def worker1(couple, output):
    """
    Docstring for worker

    :param couple: 1er element nom du file, le second est le path 
    """
    name, path = couple

    dataset, csv = load(path)
    time_serries_norm = re_sampling(dataset, csv)
    dataset['tracings'] = time_serries_norm

    z_norm(dataset['tracings'], csv)

    write_results(dataset, name, output)


def worker2(couple, output):
    """
    Docstring for worker

    :param couple: 1er element nom du file, le second est le path 
    """
    name, path = couple

    dataset, csv = load(path)
    dataset_norm = z_norm(dataset, csv)

    write_results(dataset_norm, name, output)



def run(args):
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    patch_dict1 = collect_files(args.dataset1)

    # Vérifie que des données ont été récoltés
    if not patch_dict1:
        print("Erreur: aucune données trouvé dans dataset1")
        return

    patch_items1 = list(patch_dict1.items())
    pool_worker = partial(worker1, output=args.output)

    # ---- Partie 1 uniquement sur dataset 1 -----------
    # normalise le dataset 1 pour qu'il correspondent au dataset 2
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(pool_worker, patch_items1),
                                        total=len(patch_items1),
                                        desc='Preprocessing'):
            pass


    # ---- Partie 2 sur dataset 1 & 2 ! -----------
    # Normalisation des serie temporelle
    patch_dict = collect_files(args.dataset2)

    # Vérifie que des données ont été récoltés
    if not patch_dict:
        print("Erreur: aucune données trouvé pour dataset 2")
        return

    patch_items = list(patch_dict.items())
    pool_worker = partial(worker2, output=args.output)

    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(pool_worker, patch_items),
                                        total=len(patch_items),
                                        desc='Preprocessing'):
            pass


    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Completed in {minutes} minutes and {seconds} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', '--dataset1', type=str, default='../../data/physioNet/')
    parser.add_argument('-d2', '--dataset2', type=str, default='../../data/15_prct/')
    parser.add_argument('-o', '--output', type=str, default='../output/data')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
