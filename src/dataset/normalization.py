import os
import argparse
import time
from tqdm import tqdm
import json

from functools import partial
import multiprocessing

from aux import *


def merge_labels(dataset, FINAL_CLASSES):
    """
    Merge certain labels in dataset['labels'] to match FINAL_CLASSES
    dataset['labels'] shape (N, num_labels)
    """
    
    # RBBB merge
    dataset['labels'][:, FINAL_CLASSES.index("RBBB")] = (
        dataset['labels'][:, FINAL_CLASSES.index("RBBB")] |
        dataset['labels'][:, FINAL_CLASSES.index("CRBBB")] |
        dataset['labels'][:, FINAL_CLASSES.index("IRBBB")]
    )

    # LBBB merge
    dataset['labels'][:, FINAL_CLASSES.index("LBBB")] = (
        dataset['labels'][:, FINAL_CLASSES.index("LBBB")] |
        dataset['labels'][:, FINAL_CLASSES.index("CLBBB")]
    )

    # Sinus bradycardia / Brady
    sb_idx = FINAL_CLASSES.index("SB")
    brady_idx = FINAL_CLASSES.index("Brady")
    sb = dataset['labels'][:, sb_idx]
    brady = dataset['labels'][:, brady_idx]
    # Sinus brady -> SB=1, Brady=0
    dataset['labels'][:, sb_idx] = sb
    dataset['labels'][:, brady_idx] = (~sb) & brady

    # 1dAVb merge
    dataset['labels'][:, FINAL_CLASSES.index("1dAVb")] = dataset['labels'][:, FINAL_CLASSES.index("IAVB")]

    # ST merge
    dataset['labels'][:, FINAL_CLASSES.index("ST")] = dataset['labels'][:, FINAL_CLASSES.index("STach")]

    return dataset


def worker1(couple, output, FINAL_CLASSES):
    """
    Orchestre la pipeline du traitement pour un hdf5 : 
    Chargement -> Rééchantillonnage (avec CSV) -> Normalisation -> Sauvegarde.

    Args:
        couple (tuple): Un tuple (filename, (path_h5, path_csv)).
                        - filename (str): Le nom du fichier de sortie (ex: 'patient_01.hdf5').
                        - (path_h5, path_csv) : Les chemins complets vers les fichiers sources.
        output (str): Le chemin du répertoire où sauvegarder le fichier HDF5 traité.
        FINAL_CLASSES (list): Liste des classes finales pour la fusion des labels.
    """
    name, path = couple

    dataset, csv = load(path)
    time_serries_norm = re_sampling(dataset, csv)
    dataset['tracings'] = time_serries_norm

    z_norm(dataset['tracings'])
    dataset = merge_labels(dataset) # fusionne les labels selon les regles definies

    write_results(dataset, name, output)


def worker2(couple, output, FINAL_CLASSES):
    """
    Orchestre une pipeline simplifié : Chargement -> Normalisation -> Sauvegarde.

    Ce worker ignore l'étape de rééchantillonnage et le fichier CSV.
    Il est utile si les données sont déjà à la bonne fréquence ou si le CSV est absent.

    Args:
        couple (tuple): Un tuple (filename, (path_h5, path_csv)).
        output (str): Le chemin du répertoire de sortie.
        FINAL_CLASSES (list): Liste des classes finales pour la fusion des labels.
    """
    name, path = couple

    dataset, _ = load(path, use_csv=False)
    dataset_norm = z_norm(dataset['tracings'])
    dataset_norm = merge_labels(dataset_norm)
    write_results(dataset_norm, name, output)



def run(args):
    """
    Fonction principale d'orchestration de la pipeline de prétraitement.

    Elle gère le flux de travail en deux étapes distinctes :
    1. Traitement du Dataset 1 : Resampling + Normalisation (via worker1).
    2. Traitement du Dataset 2 : Normalisation simple (via worker2).

    Elle utilise le multiprocessing pour paralléliser les tâches sur les fichiers.

    Args:
        args (argparse.Namespace): Les arguments de la ligne de commande contenant :
            - args.dataset1 : Chemin source Dataset 1.
            - args.dataset2 : Chemin source Dataset 2.
            - args.output : Dossier de sortie.
            - args.workers : Nombre de processus parallèles.
    """

    # Load FINAL_CLASSES from JSON
    with open(args.class_map) as f:
        FINAL_CLASSES = json.load(f)

    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    print("Récolte des données")

    patch_dict1 = collect_files(args.dataset1)

    # Vérifie que des données ont été récoltés
    if not patch_dict1:
        print("Erreur: aucune données trouvé dans dataset1")
        return

    patch_items1 = list(patch_dict1.items())
    pool_worker = partial(worker1, output=args.output, FINAL_CLASSES=FINAL_CLASSES)

    print("Début de la normalisation du dataset1 vers le model du dataset2")


    # ---- Partie 1 uniquement sur dataset 1 -----------
    # normalise le dataset 1 pour qu'il correspondent au dataset 2
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(pool_worker, patch_items1),
                                        total=len(patch_items1),
                                        desc='Preprocessing'):
            pass


    print("Normalisation du dataset2")


    # ---- Partie 2 uniquement sur dataset 2 -----------
    patch_dict = collect_files(args.dataset2)

    # Vérifie que des données ont été récoltés
    if not patch_dict:
        print("Erreur: aucune données trouvé pour dataset 2")
        return

    patch_items = list(patch_dict.items())
    pool_worker = partial(worker2, output=args.output, FINAL_CLASSES=FINAL_CLASSES)

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
    """
    Point d'entrée du script. 
    Parse les arguments de la ligne de commande et lance la fonction run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', '--dataset1', type=str, default='../output/dataset1/')
    parser.add_argument('-d2', '--dataset2', type=str, default='../../../data/15_prct/')
    parser.add_argument('-o', '--output', type=str, default='../output/normalize_data')
    # Added an argument for final class mapping file
    parser.add_argument('--class_map', default='../../ressources/final_class.json', help="JSON ordered class list")
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
