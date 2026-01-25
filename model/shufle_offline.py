import os
import argparse
import time
from tqdm import tqdm
import shutil

from functools import partial
import multiprocessing

import sys

# On remonte juste à la racine du projet (le dossier parent '..')
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(project_root))

from src.dataset.aux import *



def worker(couple, output):
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


def run(args):
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    patch_dict1 = collect_files(args.dataset1)

    # Vérifie que des données ont été récoltés
    if not patch_dict1:
        print("Erreur: aucune données trouvé dans dataset1")
        return

    patch_items1 = list(patch_dict1.items())
    pool_worker = partial(worker, output=args.output)

    # ---- Partie 1 uniquement sur dataset 1 -----------
    # normalise le dataset 1 pour qu'il correspondent au dataset 2
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(pool_worker, patch_items1),
                                        total=len(patch_items1),
                                        desc='Preprocessing'):
            pass


    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Completed in {minutes} minutes and {seconds} seconds")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/data/')
    parser.add_argument('-t_prct', '--train_prct', type=float, default=0.75)
    parser.add_argument('-v_prct', '--val_prct', type=float, default=0.10)
    parser.add_argument('--test_prct', type=float, default=0.15)
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)
    args = parser.parse_args()

    if args.train_prct + args.val_prct + args.test_prct >= 1:
        print(f"Erreur on ne paux partionner comme suit: {args.train_prct} + {args.val_prct} + {args.test_prct}")
        return

    run(args)


if __name__ == '__main__':
    main()
