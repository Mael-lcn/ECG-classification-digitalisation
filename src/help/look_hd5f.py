import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from datetime import datetime
from typing import Dict, Any



def get_dataset_stats(data):
    """
    Calcule les statistiques d'intégrité sur un tableau NumPy chargé en mémoire.
    """
    stats = {
        "has_nan": False,
        "has_inf": False,
        "has_underflow": False,
        "min_val": float('inf'),
        "max_val": float('-inf'),
        "size": data.size
    }

    if data.size == 0:
        return stats

    # 1. Détection vectorisée rapide des valeurs corrompues (NaN / Inf)
    if np.isnan(data).any():
        stats["has_nan"] = True

    if np.isinf(data).any():
        stats["has_inf"] = True

    # 2. Détection d'Underflow (Risque lors d'une conversion en FP16)
    if np.any((np.abs(data) > 0) & (np.abs(data) < 6.1e-5)):
        stats["has_underflow"] = True

    # 3. Calcul des bornes (Minimum et Maximum)
    try:
        stats["min_val"] = float(np.nanmin(data))
        stats["max_val"] = float(np.nanmax(data))
    except ValueError:
        pass

    return stats


def process_single_file(file_path):
    """
    Fonction exécutée par chaque worker pour analyser un fichier HDF5 unique.
    
    1. Vérifie la shape (N, C, T) sans charger la RAM.
    2. Si valide, charge la RAM et effectue l'analyse statistique.
    3. Retourne le rapport formaté.
    """
    file_name = file_path.name
    report_lines = []

    # Création de l'en-tête visuel pour délimiter les fichiers dans le log
    header = f"\n{'='*60}\nFICHIER : {file_name}\n{'='*60}"
    report_lines.append(header)

    try:
        # Ouverture du fichier HDF5 en mode lecture seule
        with h5py.File(file_path, 'r') as f:

            # Vérification de l'existence de la clé cible
            if 'tracings' not in f:
                return f"{header}\n[SKIP] La clé 'tracings' est absente du fichier.\n"

            dset = f['tracings']
            shape = dset.shape

            # =========================================================
            # VÉRIFICATION CRITIQUE DES DIMENSIONS (AVANT CHARGEMENT RAM)
            # =========================================================

            # A. Vérification qu'on a bien 3 dimensions (N, C, T)
            if len(shape) != 3:
                return f"{header}\n[CRITIQUE] Shape invalide : attendu 3D (N, C, T), obtenu {shape}.\n"
            
            N, C, T = shape

            # B. Aucune dimension ne doit être égale à 0
            if N == 0 or C == 0 or T == 0:
                return f"{header}\n[CRITIQUE] Dimension à 0 détectée : {shape}. Fichier ignoré.\n"

            # C. Le nombre de canaux (C / B) doit être strictement égal à 12
            if C != 12:
                return f"{header}\n[CRITIQUE] Canaux invalides : attendu C=12, obtenu C={C} dans la shape {shape}.\n"

            # D. La dimension temporelle (T) doit être strictement supérieure à 100
            if T <= 100:
                return f"{header}\n[CRITIQUE] Séquence temporelle trop courte : attendu T>100, obtenu T={T} dans la shape {shape}.\n"

            # =========================================================

            # Vérification que les données sont bien de type numérique
            if not np.issubdtype(dset.dtype, np.number):
                return f"{header}\n[SKIP] Le dataset 'tracings' n'est pas numérique (Type: {dset.dtype}).\n"

            # Chargement intégral du tenseur en mémoire RAM (seulement si tout est valide)
            data = dset[:]

            # Appel de la fonction d'analyse
            s = get_dataset_stats(data)

            # Construction du rapport détaillé pour ce dataset
            dset_report = [f"\n   [TARGET] 'tracings' | Shape: {dset.shape} | Type: {dset.dtype}"]

            # --- GESTION DES ALERTES (Corruptions et précisions) ---
            if s["has_nan"]:
                dset_report.append("      >>> ALERTE : Contient des NaN (Corrompu)")

            if s["has_inf"]:
                dset_report.append("      >>> ALERTE : Contient des valeurs Infinies (Inf)")
            
            if s["has_underflow"]:
                dset_report.append("      >>> WARNING : Underflow détecté (valeurs positives < 6.1e-5)")

            # --- AFFICHAGE DES STATISTIQUES GLOBALES ---
            if s["min_val"] != float('inf'):
                dset_report.append(f"      -> Min: {s['min_val']:.6f} | Max: {s['max_val']:.6f}")

                # Check de l'Overflow pour Float16 (limite max ~65504)
                if abs(s['max_val']) > 65500 or abs(s['min_val']) > 65500:
                    dset_report.append("      -> WARNING : Valeurs > 65500 (Risque Overflow Float16)")
            else:
                dset_report.append("      -> Dataset contenant uniquement des NaN.")

            # Ajout des lignes générées à la liste globale du rapport
            report_lines.extend(dset_report)

    except Exception as e:
        # En cas de crash (fichier HDF5 illisible, corruption bas niveau),
        # on capture le stack trace complet (traceback).
        error_details = traceback.format_exc()
        return f"{header}\n[ERREUR FATALE] Échec lors du traitement :\n{error_details}\n"

    # Concaténation des lignes avec des retours à la ligne pour le fichier final
    return "\n".join(report_lines)


def main():
    """
    Point d'entrée principal du script.
    """
    parser = argparse.ArgumentParser(description="Inspecteur HDF5 'tracings' (Full Load & Logging)")
    parser.add_argument('--input', type=str, default='../output/final_data/train', 
                        help="Dossier contenant les fichiers .hdf5 à analyser")
    parser.add_argument('--log', type=str, default='audit_report.log', 
                        help="Chemin du fichier texte de log en sortie")
    
    # Calcul par défaut du nombre de processus de travail
    default_workers = max(1, os.cpu_count() - 1)
    parser.add_argument('--workers', type=int, default=default_workers, 
                        help=f"Nombre de processus parallèles (défaut: {default_workers})")

    args = parser.parse_args()
    input_dir = Path(args.input)

    # 1. Validation de l'existence du dossier source
    if not input_dir.exists():
        print(f"[ERREUR] Dossier introuvable : {input_dir}")
        sys.exit(1)

    # 2. Collecte de tous les fichiers .hdf5 dans le dossier cible
    files = list(input_dir.glob("*.hdf5"))
    if not files:
        print("[INFO] Aucun fichier .hdf5 trouvé dans le dossier indiqué.")
        sys.exit(0)

    # 3. Préparation et ouverture du fichier de log principal
    with open(args.log, 'w', encoding='utf-8') as log_file:
        
        # Écriture de l'en-tête global dans le fichier de log
        log_file.write(f"=== AUDIT HDF5 DÉMARRÉ LE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(f"[INFO] Analyse de {len(files)} fichiers (Cible: 'tracings').\n")
        log_file.write(f"[INFO] Mode : RAM (si Shape N, 12, T>100 valide).\n")
        log_file.write(f"[INFO] Workers actifs : {args.workers}\n\n")

        # Seul message visible dans le terminal avant la barre de progression
        print(f"Analyse en cours... Les résultats sont écrits en direct dans : {args.log}")

        # 4. Lancement du Pool de Processus pour un traitement parallèle asynchrone
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Soumission de tous les fichiers aux workers
            future_to_file = {executor.submit(process_single_file, f): f for f in files}

            # Configuration de la barre de progression (seule sortie terminal active)
            with tqdm(total=len(files), desc="Progression", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        # Récupération du rapport formaté depuis le worker
                        report = future.result()
                        log_file.write(report + "\n")
                        log_file.flush() 
                    except Exception as exc:
                        # Capture d'une éventuelle exception remontée par le framework multiprocessing
                        error_details = traceback.format_exc()
                        log_file.write(f"\n{'='*60}\nFICHIER : {file_path.name}\n{'='*60}\n")
                        log_file.write(f"[EXCEPTION MAIN] Erreur critique du worker :\n{error_details}\n")
                        log_file.flush()
                    finally:
                        # Mise à jour de la barre de progression, succès ou échec
                        pbar.update(1)

        # 5. Écriture du pied de page indiquant la fin de l'audit
        fin_msg = f"\n{'='*80}\n[TERMINE] Audit complet effectué le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
        log_file.write(fin_msg)

    # Message final dans le terminal
    print("\nTerminé ! Tu peux consulter le fichier de log pour le sum up.")


if __name__ == "__main__":
    main()
