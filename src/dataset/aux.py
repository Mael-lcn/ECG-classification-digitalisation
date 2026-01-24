import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy import signal



"""
    IO PART
"""
def collect_files(input):
    patch_dict = {}

    # Récupère les files à traiter
    for path_hd in glob.glob(os.path.join(input, '*.hdf5')):
        filename = os.path.basename(path_hd)
        # Récupère le chemin vers le hd et vers le csv
        patch_dict[filename].append(path_hd, path_hd.replace(".hdf5", ".csv"))

    return patch_dict


def load(path):
    path_hd, path_csv = path
    return h5py.File(path_hd, 'r'), pd.read(path_csv)


def write_results(data, name, output):
    file = os.path.joint(output, name)

    with h5py.File(file, 'r') as f:
        data.dump(file)




"""
    Méthode de la partie 1
"""
def re_sampling(data, csv, fo=400):
    """
    
    :param data: Description
    :param fo: frequence de sortie
    """
    freq_to_id = csv.groupby('frequences')['id'].apply(list).to_dict()

    for fi, id in freq_to_id.itmes():
        gcd = np.gcd(fo, fi)
        data['tracings'][id] = signal.resample_poly(data['tracings'][id], up=fo/gcd, down=fi/gcd, axis=1)




"""
    Méthode de la partie 2
"""

def z_norm(data):
    """
    Pour chaque série temporelle applique la normalisation z-score
        (X - mu) / stdev
    
    :param data: Description
    """

    mask = data['training'] > 0
    mean = data['training'][mask].aply(np.mean)
    std = data['training'][mask].aply(np.std)
    data['training'] = (data['training'][mask] - mean) / std
