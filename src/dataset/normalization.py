import os
import argparse
import time
import gc
import torch
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

from aux import *



# --- CONFIGURATION GLOBALE ---
TARGET_FREQ = 400
MAX_TEMPS = 144
EPSILON = 1e-6
# Calcul de la taille du signal : 144s * 400Hz = 57600 points + marge de sécurité
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


def worker1(couple, output):
    """
    Worker pour le Dataset 1 : Pipeline complète avec rééchantillonnage.
    
    Étapes : 
    1. Chargement HDF5 + CSV.
    2. Rééchantillonnage (400Hz) en utilisant les infos du CSV.
    3. Normalisation (Z-Norm).
    4. Padding (Remplissage) pour atteindre la taille fixe.
    5. Sauvegarde sur disque.

    Args:
        couple (tuple): (nom_fichier, (chemin_h5, chemin_csv))
        output (str): Dossier de sortie.
    """
    name, path = couple

    # Initialisation à None pour que le bloc 'finally' ne plante pas 
    dataset = None
    csv = None
    try:
        # 1. Chargement
        dataset, csv = load(path)

        # 2. resampling
        dataset['tracings'] = re_sampling(dataset, csv, fo=TARGET_FREQ)
        # Mise à jour des métadonnées
        csv['frequences'] = TARGET_FREQ

        # 3. Normalisation & Padding
        dataset['tracings'] = z_norm(dataset['tracings'])
        dataset['tracings'] = add_bilateral_padding(dataset['tracings'], MAX_SIGNAL_LENGTH)

        # 4. Sauvegarde
        write_results(dataset, csv, name, output)

    except Exception as e:
        print(f"[Erreur Worker 1] Fichier {name} : {e}")

    finally:
        # gestion mémoire
        # On supprime les références pour aider le Garbage Collector
        if dataset is not None:
            del dataset
        if csv is not None:
            del csv

        # On force la libération de la RAM système immédiatement
        gc.collect() 



def worker2(couple, output):
    """
    Worker pour le Dataset 2 : Pipeline simplifiée (Pas de rééchantillonnage).

    Ce worker gère le cas où le CSV source est absent ou ignoré.
    Il renomme la colonne 'normal_ecg' en 'NSR' pour standardiser.

    Args:
        couple (tuple): (nom_fichier, (chemin_h5, chemin_csv))
        output (str): Dossier de sortie.
    """
    name, path = couple
    dataset = None
    csv = None
    try:
        # 1. Chargement
        dataset, csv = load(path)

        # Harmonisation des labels
        csv = csv.rename(columns={"normal_ecg": "NSR"})

        # 2. Traitement (Normalisation + Padding uniquement)
        dataset['tracings'] = z_norm(dataset['tracings'])
        dataset['tracings'] = add_bilateral_padding(dataset['tracings'], MAX_SIGNAL_LENGTH)

        # 3. Sauvegarde
        write_results(dataset, csv, name, output)

    except Exception as e:
        print(f"[Erreur Worker 2] Fichier {name} : {e}")

    finally:
        # gestion mémoire
        if dataset is not None:
            del dataset
        if csv is not None:
            del csv
        gc.collect()



def run(args):
    """
    Orchestrateur principal.
    Exécute le traitement du Dataset 1, nettoie le GPU, puis traite le Dataset 2.
    """
    start_time = time.time()

    # Création du dossier de sortie
    os.makedirs(args.output, exist_ok=True)


    # ------------------------------------------------------
    # PHASE 1 : TRAITEMENT DATASET 1
    # ------------------------------------------------------
    print("--- Phase 1 : Dataset 1 ---")
    print(f"Indexation depuis : {args.dataset1}")
    patch_dict1 = collect_files(args.dataset1)

    if not patch_dict1:
        print("Alerte : Aucune donnée trouvée dans dataset1. Passage à la suite.")
    else:
        patch_items1 = list(patch_dict1.items())

        # Configuration du worker avec le dossier de sortie
        pool_worker1 = partial(worker1, output=args.output)

        print(f"Traitement de {len(patch_items1)} fichiers avec {args.workers} threads...")

        # Lancement du ThreadPool
        # ThreadPool partage la mémoire et le contexte GPU -> Pas de surcharge VRAM
        with ThreadPool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(pool_worker1, patch_items1),
                      total=len(patch_items1),
                      desc='D1 Processing'))


    # INTERMÈDE : NETTOYAGE GPU
    if torch.cuda.is_available():
        print("Nettoyage intermédiaire du cache GPU...")
        torch.cuda.empty_cache()


    # ------------------------------------------------------
    # PHASE 2 : TRAITEMENT DATASET 2
    # ------------------------------------------------------
    print("\n--- Phase 2 : Dataset 2 ---")
    print(f"Indexation depuis : {args.dataset2}")
    patch_dict2 = collect_files(args.dataset2)

    if not patch_dict2:
        print("Alerte : Aucune donnée trouvée pour dataset 2.")
    else:
        patch_items2 = list(patch_dict2.items())

        pool_worker2 = partial(worker2, output=args.output)

        print(f"Traitement de {len(patch_items2)} fichiers avec {args.workers} threads...")

        with ThreadPool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(pool_worker2, patch_items2),
                      total=len(patch_items2),
                      desc='D2 Processing'))


    # FIN DU SCRIPT
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nPipeline terminée avec succès en {minutes}m {seconds}s.")



def main():
    """
    Configuration des arguments et lancement.
    """
    parser = argparse.ArgumentParser(description="Pipeline de normalisation ECG (Multithreaded)")
    
    parser.add_argument('-d1', '--dataset1', type=str, default='../output/dataset1/',
                        help='Chemin source Dataset 1')
    parser.add_argument('-d2', '--dataset2', type=str, default='../../../data/15_prct/',
                        help='Chemin source Dataset 2')
    parser.add_argument('-o', '--output', type=str, default='../output/normalize_data',
                        help='Dossier de destination')

    # Recommandation : 4 à 8 workers sur un GPU unique.
    # Plus de workers n'accélère pas forcément (contention GPU).
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Nombre de threads parallèles (Défaut: 4)')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
