import argparse
import numpy as np
import pandas as pd
import h5py
import glob
import sys
import os
import time

# Permet de trouver les modules src si lancé depuis le dossier test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataset.aux import re_sampling, z_norm



def generate_strict_physio_batch(n_samples_per_freq=10, duration_sec=2.0):
    """
    Génère des données synthétiques synchronisées mais morphologiquement uniques.
    """
    target_fs_list = [1000, 257, 250, 500, 128] 
    max_fs = max(target_fs_list)
    buffer_len = int(duration_sec * max_fs) + 200
    
    total_samples = len(target_fs_list) * n_samples_per_freq
    tracings = np.zeros((total_samples, buffer_len, 12), dtype=np.float32)
    exam_ids = []
    meta_rows = []

    current_idx = 0
    print(f"\n[SYNTHETIC] Génération de {total_samples} examens de stress...")

    for fs in target_fs_list:
        n_points = int(duration_sec * fs)
        t = np.linspace(0, duration_sec, n_points).reshape(-1, 1)
        
        for i in range(n_samples_per_freq):
            e_id = f"SYNTH_Fs{fs}_{i}"
            b_id = e_id.encode('utf-8')

            # Paramètres physiques
            heart_rate_hz = np.random.uniform(0.5, 3.0) 
            amps = np.random.uniform(-40.0, 40.0, size=(1, 12))
            phases = np.random.uniform(0, 2*np.pi, size=(1, 12))
            offsets = np.random.uniform(-100, 100, size=(1, 12))

            # Synthèse
            clean_sig = amps * np.sin(2 * np.pi * heart_rate_hz * t + phases) + offsets
            noise = np.random.normal(0, 0.1 * np.abs(amps), (n_points, 12))
            
            tracings[current_idx, :n_points, :] = clean_sig + noise
            
            exam_ids.append(b_id)
            meta_rows.append({'exam_id': e_id, 'frequences': fs})
            current_idx += 1

    # Shuffle
    perm = np.random.permutation(total_samples)
    return {
        'tracings': tracings[perm], 
        'exam_id': [exam_ids[i] for i in perm]
    }, pd.DataFrame(meta_rows)



def load_real_data(input_dir, n_samples=50):
    """
    Charge aléatoirement n_samples par fichier depuis un dossier.
    Gère la correspondance HDF5 <-> CSV via 'exam_id'.

    Args:
        input_dir (str): Chemin du dossier contenant les .hdf5 et .csv.
        n_samples (int): Nombre d'échantillons à prélever DANS CHAQUE FICHIER.

    Returns:
        dict: {'tracings': np.array (N, T, 12), 'exam_id': list}
        pd.DataFrame: Les métadonnées correspondantes (N lignes).
    """
    print(f"\n[LOADING] Scan du dossier : {input_dir}")

    # 1. Repérage des fichiers
    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if not h5_files:
        raise FileNotFoundError("Aucun fichier .hdf5 trouvé.")

    # 2. Chargement et Indexation du CSV Global
    # On charge tous les CSV pour créer une "Map" globale : exam_id -> row
    print("   -> Chargement et indexation des métadonnées (CSV)...")
    if csv_files:
        try:
            df_global = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            
            # OPTIMISATION : On met l'exam_id en index pour une recherche ultra-rapide
            df_global = df_global.set_index('exam_id')

        except Exception as e:
            raise ValueError(f"Erreur lors du chargement des CSV : {e}")
    else:
        raise FileNotFoundError("Aucun fichier CSV trouvé pour la correspondance.")

    # 3. Extraction des données (HDF5 + CSV Row)
    collected_tracings = []
    collected_ids = []
    collected_meta_rows = []

    total_files = len(h5_files)
    print(f"   -> Traitement de {total_files} fichiers HDF5 ({n_samples} samples/file)...")

    for i, h5_path in enumerate(h5_files):
        fname = os.path.basename(h5_path)

        try:
            with h5py.File(h5_path, 'r') as f:
                # Vérification structure
                if 'tracings' not in f.keys() or 'exam_id' not in f.keys():
                    print(f"      [SKIP] {fname} : Clés manquantes.")
                    continue

                dset_tracings = f['tracings']
                dset_ids = f['exam_id']
                n_avail = dset_tracings.shape[0]

                n_to_pick = min(n_samples, n_avail)

                # Tirage aléatoire d'indices
                indices = np.sort(np.random.choice(n_avail, n_to_pick, replace=False))

                # Extraction
                for idx in indices:
                    # A. Récupération ID et Signal
                    raw_id = dset_ids[idx]
                    signal = dset_tracings[idx] # Shape (T, 12)

                    # B. Décodage de l'ID (Bytes -> String)
                    exam_id_str = raw_id.decode('utf-8')

                    # C. Récupération de la ligne CSV correspondante
                    if exam_id_str in df_global.index:
                        # On récupère la ligne sous forme de Series
                        meta_row = df_global.loc[exam_id_str]

                        # D. Stockage
                        collected_tracings.append(signal)
                        collected_ids.append(raw_id)

                        # On remet l'exam_id dans la ligne car il est passé en index
                        meta_dict = meta_row.to_dict()
                        meta_dict['exam_id'] = exam_id_str
                        collected_meta_rows.append(meta_dict)
                    else:
                        # Si l'ID n'est pas dans le CSV, on ignore ce sample pour garantir la cohérence
                        print(f"Warning: ID {exam_id_str} introuvable dans le CSV.")
                        pass

        except Exception as e:
            print(f"      [ERR] Erreur lecture {fname} : {e}")

    # 4. Assemblage Final
    if not collected_tracings:
        raise ValueError("Aucune donnée valide récupérée (vérifiez la correspondance IDs HDF5 <-> CSV).")

    # Création du DataFrame final
    df_result = pd.DataFrame(collected_meta_rows)
    
    # Création du Tenseur Numpy avec Padding (car durées variables possibles)
    # On trouve la longueur temporelle max
    max_len = max(t.shape[0] for t in collected_tracings)
    n_total = len(collected_tracings)

    # Init tenseur (N, T_max, 12)
    batch_tensor = np.zeros((n_total, max_len, 12), dtype=np.float32)

    # Remplissage
    for idx, trace in enumerate(collected_tracings):
        # On aligne à gauche (padding à droite)
        length = trace.shape[0]
        batch_tensor[idx, :length, :] = trace

    print(f"   -> Terminé. Total chargé : {n_total} examens. Shape: {batch_tensor.shape}")

    return {'tracings': batch_tensor, 'exam_id': collected_ids}, df_result



def run_checks(data, csv_meta, target_fo=400):
    """
    Exécute Resampling + Z-Norm et valide les résultats.
    """
    print("="*60)
    print("DÉBUT DU PIPELINE DE VALIDATION")
    print("="*60)
    
    start_time = time.time()
    
    # --- ETAPE 1 : RESAMPLING ---
    print(f"\n1. Exécution re_sampling (Cible: {target_fo}Hz)...")
    #try:
    resampled = re_sampling(data, csv_meta, fo=target_fo)
    print(f"   Succès. Output Shape: {resampled.shape}")


    # Check intégrité resampling
    errors_integrity = 0
    for i in range(min(5, resampled.shape[0])):
        sig = resampled[i]
        # Check simple : pas de valeurs infinies
        if not np.isfinite(sig).all():
            print(f"   Sample {i} contient des NaNs ou Infs après resampling !")
            errors_integrity += 1

        # Check Cross-Talk (Si Lead 0 == Lead 1, c'est louche, sauf ligne plate)
        if np.std(sig[:, 0]) > 0.1 and np.allclose(sig[:, 0], sig[:, 1]):
             print(f"   Alerte : Lead 0 et 1 identiques sur sample {i}")

    if errors_integrity == 0:
        print("   Intégrité basique OK.")


    # --- ETAPE 2 : Z-NORM ---
    print(f"\n2. Exécution z_norm...")
    try:
        normalized = z_norm(resampled)
        print(f"   Succès. Output Shape: {normalized.shape}")
    except Exception as e:
        print(f"   CRASH z_norm: {e}")
        return

    # --- ETAPE 3 : AUDIT STATISTIQUE ---
    print(f"\n3. Audit Statistique (Check Mean=0, Std=1, Padding=0)...")

    n_samples = normalized.shape[0]
    stats_ok_count = 0
    padding_ok_count = 0

    print(f"   {'IDX':<6} | {'MEAN (Max)':<12} | {'STD (Range)':<20} | {'PADDING'}")
    print("   " + "-"*55)

    # On audite tout le monde (ou max 20 pour affichage)
    samples_to_show = min(n_samples, 20)

    for i in range(n_samples):
        sig = normalized[i]
        # Découpe zone active
        active_idx = np.where(np.any(np.abs(sig) > 1e-6, axis=1))[0]

        if len(active_idx) == 0:
            # Signal vide/mort
            if i < samples_to_show:
                print(f"   {i:<6} | {'VIDE':<12} | {'VIDE':<20} | OK")
            continue

        stop = active_idx[-1] + 1
        active = sig[:stop, :]
        padding = sig[stop:, :]

        # Stats
        means = np.mean(active, axis=0)
        stds = np.std(active, axis=0)

        max_mean_dev = np.max(np.abs(means))
        min_std, max_std = np.min(stds), np.max(stds)

        # Check Padding
        pad_clean = True
        if padding.size > 0:
            pad_max = np.max(np.abs(padding))
            if pad_max > 1e-6:
                pad_clean = False

        if pad_clean: padding_ok_count += 1

        # Validité Statistique (Mean < 0.2, 0.5 < Std < 1.5 pour être large sur données réelles)
        is_stat_valid = (max_mean_dev < 0.2) and (0.5 < min_std) and (max_std < 1.5)
        if is_stat_valid: stats_ok_count += 1
        
        if i < samples_to_show:
            stat_str = "OK" if is_stat_valid else "BAD"
            pad_str = "OK" if pad_clean else "DIRTY"
            print(f"   {i:<6} | {max_mean_dev:<12.4f} | {min_std:.2f}-{max_std:.2f} ({stat_str})   | {pad_str}")

    print("   " + "-"*55)
    
    # Résumé
    print(f"\n[RAPPORT FINAL]")
    print(f"Samples traités      : {n_samples}")
    print(f"Statistiques Valides : {stats_ok_count} / {n_samples} ({(stats_ok_count/n_samples)*100:.1f}%)")
    print(f"Padding Propre       : {padding_ok_count} / {n_samples} ({(padding_ok_count/n_samples)*100:.1f}%)")
    
    if stats_ok_count < n_samples * 0.9:
        print("\nÉCHEC : Trop d'échantillons ont une normalisation incorrecte.")
    else:
        print("\nSUCCÈS : Le pipeline semble robuste.")

    print(f"Temps d'exécution : {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Validation du pipeline ECG (Synthétique ou Réel)")

    parser.add_argument('-i', '--input_dir', type=str, default="../output/normalize_data/",
                        help="Chemin vers le dossier contenant les .hdf5 et .csv (Données réelles).")

    parser.add_argument('-a', '--use_input', action=argparse.BooleanOptionalAction,
                        help="True pour utiliser les données réelles")

    parser.add_argument('-n', '--n_samples', type=int, default=500,
                        help="Nombre d'échantillons à tester (Réel ou Synthétique).")

    parser.add_argument('-f', '--freq_out', type=int, default=400,
                        help="Fréquence de rééchantillonnage cible.")

    args = parser.parse_args()

    if args.use_input:
        # --- CAS RÉEL ---
        if not os.path.exists(args.input_dir):
            print(f"ERREUR: Le dossier {args.input_dir} n'existe pas.")
            sys.exit(1)

        print(f"MODE: Validation sur Données RÉELLES")
        try:
            data_dict, csv_df = load_real_data(args.input_dir, n_samples=args.n_samples)
            run_checks(data_dict, csv_df, target_fo=args.freq_out)
        except Exception as e:
            print(f"ERREUR FATALE lors du chargement: {e}")

    else:
        # --- CAS SYNTHÉTIQUE ---
        print(f"MODE: Validation sur Données SYNTHÉTIQUES (Stress Test)")
        # On calcule combien par freq pour atteindre le total approx
        per_freq = max(1, args.n_samples // 5) 
        data_dict, csv_df = generate_strict_physio_batch(n_samples_per_freq=per_freq)
        run_checks(data_dict, csv_df, target_fo=args.freq_out)


if __name__ == "__main__":
    main()
