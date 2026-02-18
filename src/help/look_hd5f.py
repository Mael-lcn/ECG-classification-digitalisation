import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed



def get_dataset_stats(data):
    """
    Calcule les statistiques sur un tableau NumPy chargé en mémoire.
    Inclut la détection d'Underflow (valeurs trop petites pour FP16).
    
    Args:
        data (np.ndarray): Le tableau de données complet.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les métriques d'intégrité.
    """
    stats = {
        "has_nan": False,
        "has_inf": False,
        "has_underflow": False,
        "min_val": float('inf'),
        "max_val": float('-inf'),
        "total_zeros": 0,
        "size": data.size
    }

    if data.size == 0:
        return stats

    # 1. Détection vectorisée (NaN / Inf)
    if np.isnan(data).any():
        stats["has_nan"] = True

    if np.isinf(data).any():
        stats["has_inf"] = True

    # 2. Détection Underflow (Risque FP16)
    # On cherche des valeurs non-nulles mais inférieures au seuil de précision du float16 (~6e-5)
    if np.any((np.abs(data) > 0) & (np.abs(data) < 6.1e-5)):
        stats["has_underflow"] = True

    # 3. Calcul des bornes
    try:
        stats["min_val"] = float(np.nanmin(data))
        stats["max_val"] = float(np.nanmax(data))
    except ValueError:
        # Cas rare : dataset rempli uniquement de NaN
        pass

    # 4. Comptage des zéros
    stats["total_zeros"] = np.count_nonzero(data == 0)

    return stats


def process_single_file(file_path):
    """
    Worker Function : Charge uniquement la clé 'tracing'.
    """
    file_name = file_path.name
    report_lines = []

    # En-tête
    header = f"\n{'='*60}\nFICHIER : {file_name}\n{'='*60}"
    report_lines.append(header)

    try:
        with h5py.File(file_path, 'r') as f:

            # Récupération du dataset
            dset = f['tracing']

            # Vérification type
            if not np.issubdtype(dset.dtype, np.number):
                return f"{header}\n[SKIP] Le dataset 'tracing' n'est pas numérique.\n"

            # Charge tout le tenseur (B, C, T) en RAM
            data = dset[:]

            # Analyse
            s = get_dataset_stats(data)

            # Construction du rapport
            dset_report = [f"\n   [TARGET] 'tracing' | Shape: {dset.shape} | Type: {dset.dtype}"]

            # --- ALERTES ---
            if s["has_nan"]:
                dset_report.append("      >>> ALERTE : Contient des NaN (Corrompu)")

            if s["has_inf"]:
                dset_report.append("      >>> ALERTE : Contient des valeurs Infinies (Inf)")
            
            if s["has_underflow"]:
                dset_report.append("      >>> WARNING : Underflow détecté (valeurs positives < 6.1e-5)")

            # --- STATS ---
            if s["min_val"] != float('inf'):
                dset_report.append(f"      -> Min: {s['min_val']:.6f} | Max: {s['max_val']:.6f}")

                # Check Overflow Float16
                if abs(s['max_val']) > 65500 or abs(s['min_val']) > 65500:
                    dset_report.append("      -> WARNING : Valeurs > 65500 (Risque Overflow Float16)")

            else:
                dset_report.append("      -> Dataset vide ou invalide.")

            report_lines.extend(dset_report)

    except Exception as e:
        return f"{header}\n[ERREUR FATALE] Échec lecture : {e}\n"

    return "\n".join(report_lines)


def main():
    """
    Point d'entrée principal.
    """
    parser = argparse.ArgumentParser(description="Inspecteur HDF5 'tracing' (Full Load)")
    parser.add_argument('--input', type=str, default='../output/final_data/train', help="Dossier contenant les fichiers .hdf5")
    
    # Gestion du nombre de workers
    default_workers = max(1, os.cpu_count() - 1)
    parser.add_argument('--workers', type=int, default=default_workers, help=f"Nombre de processus (défaut: {default_workers})")

    args = parser.parse_args()
    input_dir = Path(args.input)

    # 1. Validation du dossier
    if not input_dir.exists():
        print(f"[ERREUR] Dossier introuvable : {input_dir}")
        sys.exit(1)

    # 2. Collecte des fichiers
    files = list(input_dir.glob("*.hdf5"))
    if not files:
        print("[INFO] Aucun fichier .hdf5 trouvé.")
        sys.exit(0)

    print(f"[INFO] Analyse de {len(files)} fichiers (Cible: 'tracing').")
    print(f"[INFO] Mode : Chargement intégral en RAM.")
    print(f"[INFO] Workers actifs : {args.workers}\n")

    # 3. Lancement du Pool de Processus
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(process_single_file, f): f for f in files}

        with tqdm(total=len(files), desc="Progression", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    report = future.result()
                    tqdm.write(report)
                except Exception as exc:
                    tqdm.write(f"\n[EXCEPTION MAIN] Erreur sur {file_path.name} : {exc}")
                finally:
                    pbar.update(1)

    print(f"\n{'='*80}")
    print("[TERMINE] Audit complet effectué.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
