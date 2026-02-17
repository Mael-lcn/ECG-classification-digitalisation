import h5py
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm



def analyze_dataset_statistics(dset, chunk_size=1024):
    """
    Parcourt un dataset HDF5 par morceaux pour calculer ses statistiques globales
    sans charger tout le fichier en mémoire.

    Args:
        dset (h5py.Dataset): L'objet dataset HDF5 à analyser.
        chunk_size (int): Nombre d'échantillons à charger par itération.

    Returns:
        dict: Dictionnaire contenant {has_nan, has_inf, min_val, max_val, total_zeros}.
    """
    # Initialisation des statistiques
    stats = {
        "has_nan": False,
        "has_inf": False,
        "min_val": float('inf'),
        "max_val": float('-inf'),
        "total_zeros": 0
    }

    total_samples = dset.shape[0]

    # On utilise tqdm pour voir l'avancement car le scan peut être long
    # desc=f"   Scan {dset.name}" permet de voir quel dataset est traité
    pbar = tqdm(total=total_samples, desc=f"   Analyse {dset.name.split('/')[-1]}", unit="samples", leave=False)

    for i in range(0, total_samples, chunk_size):
        # Lecture optimisée : on ne charge que le morceau nécessaire en RAM
        chunk = dset[i : i + chunk_size]

        # Conversion en float pour les calculs stats si nécessaire, 
        # mais on garde le type original pour la détection NaN si possible.
        # Note: Si c'est des entiers, pas de NaN possible, mais on gère le cas général.
        if not np.issubdtype(chunk.dtype, np.number):
            pbar.update(chunk.shape[0])
            continue

        # 1. Détection NaN / Inf
        if np.isnan(chunk).any():
            stats["has_nan"] = True
        
        if np.isinf(chunk).any():
            stats["has_inf"] = True

        # 2. Calcul Min / Max Local
        # On utilise nanmin/nanmax pour ne pas propager les NaN dans le calcul du min/max
        # si jamais il y en a (bien qu'on les flag juste avant)
        try:
            current_min = np.nanmin(chunk)
            current_max = np.nanmax(chunk)

            if current_min < stats["min_val"]:
                stats["min_val"] = current_min

            if current_max > stats["max_val"]:
                stats["max_val"] = current_max
        except ValueError:
             # Cas où le chunk serait vide ou rempli de NaN uniquement
             pass

        # 3. Compte les zéros exacts (pour détecter des signaux vides/morts)
        stats["total_zeros"] += np.count_nonzero(chunk == 0)

        pbar.update(chunk.shape[0])

    pbar.close()
    return stats


def inspect_h5_thorough(file_path):
    """
    Ouvre un fichier HDF5 et lance une inspection approfondie de tous les datasets numériques.

    Args:
        file_path (str): Chemin vers le fichier .hdf5.
    """
    if not os.path.exists(file_path):
        print(f"[ERREUR] Fichier introuvable : {file_path}")
        return

    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"RAPPORT D'INSPECTION : {os.path.basename(file_path)}")
    print(f"Taille sur disque : {file_size:.2f} MB")
    print(f"{'='*80}")

    try:
        with h5py.File(file_path, 'r') as f:
            
            def visit_node(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\n[DATASET] {name}")
                    print(f"   Shape : {obj.shape}")
                    print(f"   Type  : {obj.dtype}")

                    # On analyse uniquement les datasets contenant des nombres (float/int)
                    # On exclut les chaînes de caractères (ex: exam_id)
                    if np.issubdtype(obj.dtype, np.number):
                        
                        # Lancement de l'analyse chunk par chunk
                        s = analyze_dataset_statistics(obj)

                        # --- RAPPORT D'ANALYSE ---
                        if s["has_nan"]:
                            print(f"   🛑 CRITIQUE : Contient des NaN ! (Corrompu)")
                        else:
                            print(f"   ✅ Intégrité : Pas de NaN détecté.")

                        if s["has_inf"]:
                            print(f"   🛑 CRITIQUE : Contient des valeurs Infinies (Inf) !")
                        else:
                            print(f"   ✅ Intégrité : Pas de valeurs Infinies.")

                        print(f"   📊 Statistiques globales :")
                        print(f"       -> Min : {s['min_val']:.4f}")
                        print(f"       -> Max : {s['max_val']:.4f}")
                        
                        # Alerte si les valeurs sont trop grandes pour du Float16 (AMP)
                        # Limite Float16 ~ 65500
                        if abs(s['max_val']) > 65000 or abs(s['min_val']) > 65000:
                            print(f"       ⚠️ WARNING AMP : Valeurs > 65000 (Risque d'Overflow en Float16)")

                        # Vérification de "signaux plats"
                        total_elements = np.prod(obj.shape)
                        zero_ratio = (s["total_zeros"] / total_elements) * 100
                        print(f"       -> Zéros : {zero_ratio:.2f}% du volume total")
                    
                    else:
                        print(f"   ℹ️  Données non-numériques (ignorées pour stats min/max)")

            f.visititems(visit_node)

    except Exception as e:
        print(f"\n[FATAL ERROR] Impossible de lire le fichier : {e}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspecteur HDF5 exhaustif (Détection NaN/Inf/MinMax)")
    parser.add_argument('file', type=str, help="Chemin vers le fichier .hdf5")
    args = parser.parse_args()
    
    inspect_h5_thorough(args.file)
