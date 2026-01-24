import os
import argparse
import time
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial
import multiprocessing

from scipy import signal



def load(path):
    path_hd, path_csv = path
    return h5py.File(path_hd, 'r'), pd.read(path_csv)



# Frequence < frequence cible
def re_sampling(data, csv, fo=400):
    """
    
    :param data: Description
    :param fo: frequence de sortie
    """
    freq_to_id = csv.groupby('frequences')['id'].apply(list).to_dict()

    for fi, id in freq_to_id.itmes():
        gcd = np.gcd(fo, fi)
        data['tracings'][id] = signal.resample_poly(data['tracings'][id], up=fo/gcd, down=fi/gcd, axis=1)



def write_results(data, name, output):
    file = os.path.joint(output, name)

    with h5py.File(file, 'r') as f:
        data.dump(file)


def worker(couple, output):
    """
    Docstring for worker

    :param couple: 1er element nom du file, le second est le path 
    """
    name, path = couple

    dataset, csv = load(path)
    dataset_norm = re_sampling(dataset, csv)

    write_results(dataset_norm, name, output)



def run(args):
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    patch_dict1 = {}

    # Récupère les files à traiter
    for path_hd in glob.glob(os.path.join(args.dataset1, '*.hdf5')):
        filename = os.path.basename(path_hd)
        # Récupère le chemin originel et le chemin vers le csv
        patch_dict1[filename].append(path_hd, path_hd.replace(".hdf5", ".csv"))

    # Vérifie que des données ont été récoltés
    if not patch_items1:
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


    if not args.mode:
        # ---- Partie 2 sur dataset 1 & 2 ! -----------
        # Normalisation des serie temporelle
        patch_dict2 = {}

        # Récupère les files à traiter
        for path_hd in glob.glob(os.path.join(args.input, '*.hdf5')):
            filename = os.path.basename(path_hd)
            # Récupère le chemin originel et le chemin vers le csv
            patch_dict2[filename].append(path_hd, path_hd.replace(".hdf5", ".csv"))

        # Vérifie que des données ont été récoltés
        if not patch_dict2:
            print("Erreur: aucune données trouvé pour dataset 2")
            return

        patch_items = patch_items1 + list(patch_dict2.items())
        pool_worker = partial(worker, output=args.output)

        # normalise le dataset 1 pour qu'il correspondent au dataset 2
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
    parser.add_argument('-o', '--output', type=str, default='../output/')
    parser.add_argument('-m', '--mode', action=argparse.BooleanOptionalAction)
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
