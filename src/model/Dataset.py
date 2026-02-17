import h5py
import numpy as np
import os
import argparse
import glob
import multiprocessing
import random



def analyze_dataset_chunked(dset, chunk_size=2048):
    """
    Analyse un dataset HDF5 par morceaux pour extraire les statistiques sans charger la mémoire.

    Cette méthode itère sur le dataset par blocs pour éviter de saturer la RAM,
    même sur des fichiers très volumineux.

    Args:
        dset (h5py.Dataset): L'objet dataset HDF5 à analyser.
        chunk_size (int, optional): Taille du buffer de lecture en nombre d'échantillons. 
            Défaut à 2048.

    Returns:
        dict: Dictionnaire contenant les clés suivantes :
            - has_nan (bool): True si des NaN sont détectés.
            - has_inf (bool): True si des valeurs infinies sont détectées.
            - min_val (float): La valeur minimale trouvée.
            - max_val (float): La valeur maximale trouvée.
    """
    stats = {
        "has_nan": False,
        "has_inf": False,
        "min_val": float('inf'),
        "max_val": float('-inf')
    }
    
    total_samples = dset.shape[0]

    for i in range(0, total_samples, chunk_size):
        # Lecture du chunk
        chunk = dset[i : i + chunk_size]

        # On ignore les données non numériques (ex: strings)
        if not np.issubdtype(chunk.dtype, np.number):
            continue

        # Détection NaN / Inf
        if np.isnan(chunk).any():
            stats["has_nan"] = True
        
        if np.isinf(chunk).any():
            stats["has_inf"] = True

        # Calcul Min/Max (en ignorant les NaN pour ne pas fausser le min/max)
        try:
            # Note: Si le chunk est vide ou full-NaN, cela peut lever une erreur ou warning
            if chunk.size > 0:
                current_min = np.nanmin(chunk)
                current_max = np.nanmax(chunk)
                
                if current_min < stats["min_val"]:
                    stats["min_val"] = current_min
                
                if current_max > stats["max_val"]:
                    stats["max_val"] = current_max
        except ValueError:
            pass 

    return stats


def worker_inspect_file(file_path):
    """
    Fonction exécutée par un processus travailleur pour inspecter un fichier unique.

    Args:
        file_path (str): Chemin absolu vers le fichier .hdf5 à inspecter.

    Returns:
        dict: Un dictionnaire de rapport contenant :
            - file (str): Nom du fichier.
            - status (str): 'OK', 'CORRUPT', 'WARNING' ou 'ERROR'.
            - report (list[str]): Liste des messages d'erreurs ou d'avertissements.
    """
    filename = os.path.basename(file_path)
    report_lines = []
    status = "OK"
    
    try:
        with h5py.File(file_path, 'r') as f:
            datasets_found = 0
            
            def visit_node(name, obj):
                nonlocal status, datasets_found
                
                # On ne traite que les datasets numériques
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    datasets_found += 1
                    s = analyze_dataset_chunked(obj)

                    # 1. Vérification intégrité numérique
                    if s["has_nan"]:
                        status = "CORRUPT"
                        report_lines.append(f"[DATASET] {name} -> CONTIENT DES NAN")
                    
                    if s["has_inf"]:
                        status = "CORRUPT"
                        report_lines.append(f"[DATASET] {name} -> CONTIENT DES INFINIS")

                    # 2. Vérification débordement Float16 (AMP)
                    # La limite float16 est 65504. Au-delà, risque d'overflow en entraînement mixte.
                    limit_fp16 = 65500
                    if abs(s['max_val']) > limit_fp16 or abs(s['min_val']) > limit_fp16:
                        report_lines.append(f"[DATASET] {name} -> WARNING AMP: Valeurs hors limites Float16 (Min:{s['min_val']:.1f}, Max:{s['max_val']:.1f})")

            # Parcours récursif du fichier
            f.visititems(visit_node)
            
            if datasets_found == 0:
                status = "WARNING"
                report_lines.append("Aucun dataset numérique trouvé.")

    except Exception as e:
        status = "ERROR"
        report_lines.append(f"ERREUR CRITIQUE DE LECTURE: {str(e)}")

    return {
        "file": filename,
        "status": status,
        "report": report_lines
    }


def main():
    """
    Point d'entrée principal du script.
    
    Orchestre la recherche de fichiers, la sélection aléatoire et l'exécution parallèle.
    """
    # Configuration des arguments
    parser = argparse.ArgumentParser(description="Inspection aléatoire et parallèle de fichiers HDF5.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Répertoire racine contenant les fichiers .hdf5")
    
    # Par défaut, on utilise tous les cœurs disponibles
    parser.add_argument('--workers', type=int, default=os.cpu_count(), 
                        help=f"Nombre de fichiers à inspecter (et de cœurs à utiliser). Défaut: {os.cpu_count()}")
    
    args = parser.parse_args()

    # 1. Recherche des fichiers
    search_pattern = os.path.join(args.input_dir, "**/*.hdf5")
    all_files = glob.glob(search_pattern, recursive=True)

    # Fallback si pas de récursivité
    if not all_files:
        all_files = glob.glob(os.path.join(args.input_dir, "*.hdf5"))

    total_files = len(all_files)
    if total_files == 0:
        print(f"[ERREUR] Aucun fichier .hdf5 trouvé dans : {args.input_dir}")
        return

    # 2. Échantillonnage Aléatoire
    # On ne peut pas inspecter plus de fichiers qu'il n'y en a
    nb_to_inspect = min(args.workers, total_files)

    random.seed() # Initialisation du générateur aléatoire
    selected_files = random.sample(all_files, nb_to_inspect)

    print(f"{'-'*60}")
    print(f"RAPPORT D'INSPECTION HDF5")
    print(f"Dossier source : {args.input_dir}")
    print(f"Total fichiers disponibles : {total_files}")
    print(f"Fichiers sélectionnés (Random) : {nb_to_inspect}")
    print(f"Processus parallèles (Workers) : {nb_to_inspect}")
    print(f"{'-'*60}\n")

    # 3. Exécution Parallèle
    # On crée un pool exactement de la taille de l'échantillon pour que chaque fichier ait son worker
    with multiprocessing.Pool(processes=nb_to_inspect) as pool:
        # map bloque jusqu'à ce que tous les résultats soient prêts
        results = pool.map(worker_inspect_file, selected_files)

    # 4. Affichage des résultats
    files_corrupt = 0
    files_ok = 0
    
    for res in results:
        status = res["status"]
        filename = res["file"]
        report = res["report"]

        if status == "OK":
            files_ok += 1
            print(f"[OK] {filename}")
        else:
            files_corrupt += 1
            print(f"[{status}] {filename}")
            for line in report:
                print(f"    - {line}")

    # 5. Résumé final
    print(f"\n{'-'*60}")
    print(f"BILAN DE L'ECHANTILLONNAGE")
    print(f"{'-'*60}")
    
    if files_corrupt == 0:
        print(f"SUCCES : Les {nb_to_inspect} fichiers testés sont valides.")
    else:
        print(f"ECHEC : {files_corrupt} fichiers problématiques détectés sur {nb_to_inspect}.")
        print("Il est recommandé de nettoyer le dataset ou de vérifier la chaîne de production des données.")


if __name__ == "__main__":
    main()
