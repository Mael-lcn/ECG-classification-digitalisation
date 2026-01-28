import os
import glob
import argparse
import multiprocessing
import h5py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool



# CONSTANTES DE VALIDATION
TOLERANCE_MU = 0.05
SIGMA_MIN = 0.8
SIGMA_MAX = 1.2        


def plot_sample(sample_data, filename):
    """
    Génère une visualisation 12 pistes pour validation humaine.
    """
    if sample_data is None:
        return

    print(f"\n[VISUALISATION] Génération du graphique pour : {filename}")

    fig, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # sample_data shape attendue : (Length, 12)
    for i in range(12):
        ax = axes[i]
        signal = sample_data[:, i]

        ax.plot(signal, color='black', linewidth=0.8)
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=20, fontsize=12, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.axhline(0, color='red', linewidth=0.5, alpha=0.7) # Ligne de référence 0

        # Esthétique "Clean"
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False if i < 11 else True)
        ax.spines['left'].set_visible(False)

    plt.xlabel("Temps (samples)")
    plt.suptitle(f"Check Z-Norm\nFichier: {filename}\nμ={np.mean(sample_data):.3f}, σ={np.std(sample_data):.3f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def worker_check_file(args):
    """
    Fonction exécutée par chaque processeur (Worker).
    Ouvre un fichier, tire des samples au hasard et calcule les stats.
    """
    filepath, n_samples_to_check = args
    filename = os.path.basename(filepath)
    source = "D2 (Ref)" if "15_prct" in filepath else "D1 (Data)"

    # Structure de retour
    result = {
        "filename": filename,
        "source": source,
        "checked_count": 0,
        "avg_mu": 0.0,
        "avg_sigma": 0.0,
        "errors": [],
        "status": "OK",
        "last_sample": None,
        "fatal": None
    }

    try:
        with h5py.File(filepath, 'r') as f:
            if 'tracings' not in f.keys():
                result["status"] = "CRASH"
                result["fatal"] = "Clé 'tracings' introuvable"
                return result

            dataset = f['tracings']
            total_available = dataset.shape[0]

            # On prend le min entre ce qui est demandé et ce qui existe
            n_batch = min(n_samples_to_check, total_available)

            # Indices aléatoires
            indices = np.random.choice(total_available, n_batch, replace=False)
            indices.sort()

            mus = []
            sigmas = []

            for idx in indices:
                raw = dataset[idx]

                # --- Normalisation des dimensions ---
                # On veut (1, Time, Leads) pour les calculs de stats globaux
                sample = np.expand_dims(raw, axis=0)

                # Check intégrité
                if sample.shape[2] != 12:
                    result["errors"].append(f"Idx {idx}: Shape {sample.shape}")
                    continue

                # --- Maths ---
                val_mu = np.mean(sample)
                val_sigma = np.std(sample)

                # Check NaN
                if np.isnan(sample).any():
                    result["errors"].append(f"Idx {idx}: Contient NaN")

                # Check Z-Norm
                # TODO il faut masker le padding !!!!!!!!
                elif abs(val_mu) > TOLERANCE_MU:
                    result["errors"].append(f"Idx {idx}: μ={val_mu:.2f} (Hors tolérance)")

                elif not (SIGMA_MIN < val_sigma < SIGMA_MAX):
                    result["errors"].append(f"Idx {idx}: σ={val_sigma:.2f} (Hors tolérance)")

                mus.append(val_mu)
                sigmas.append(val_sigma)

                # On garde le dernier pour l'affichage
                if idx == indices[-1]:
                    result["last_sample"] = sample[0] # (Time, 12)

            # Synthèse
            if mus:
                result["avg_mu"] = np.mean(mus)
                result["avg_sigma"] = np.mean(sigmas)
                result["checked_count"] = len(mus)

            if result["errors"]:
                result["status"] = "FAIL"

    except Exception as e:
        result["status"] = "CRASH"
        result["fatal"] = str(e)

    return result


def run(args):
    """
    Orchestre la collecte des fichiers et le multiprocessing.
    """
    print(f"--- DÉMARRAGE DU CHECK Z-NORM ---")
    print(f"Dataset 1 (Cible) : {args.dataset1}")
    print(f"Dataset 2 (Ref)   : {args.dataset2}")
    print(f"Workers           : {args.workers}")
    print(f"Samples / Fichier : {args.n_samples}")
    print("="*100)

    # 1. Collecte des fichiers
    files_d1 = sorted(glob.glob(os.path.join(args.dataset1, '*.hdf5')))

    # Pour le dataset 2, on en prend juste 1 comme demandé pour référence
    files_d2 = sorted(glob.glob(os.path.join(args.dataset2, '*.hdf5')))[:1]

    all_files = files_d1 + files_d2
    
    if not all_files:
        print("ERREUR: Aucun fichier HDF5 trouvé.")
        return

    # Préparation des arguments pour le map (Tuples)
    tasks = [(f, args.n_samples) for f in all_files]

    # 2. Exécution Multiprocessing
    print(f"{'SOURCE':<10} | {'FICHIER':<35} | {'COUNT':<6} | {'μ (Moy)':<10} | {'σ (Moy)':<10} | {'STATUS'}")
    print("-" * 100)

    last_valid_data = None
    last_valid_name = ""
    total_fails = 0

    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(worker_check_file, tasks):
            # Mise en forme nom
            fname = res['filename']
            short_name = (fname[:32] + '..') if len(fname) > 34 else fname

            # Icônes
            icon = "[OK]"
            if res['status'] == "FAIL": icon = "[FAIL]"
            if res['status'] == "CRASH": icon = "[ERR]"

            print(f"{res['source']:<10} | {short_name:<35} | {res['checked_count']:<6} | {res['avg_mu']:<10.4f} | {res['avg_sigma']:<10.4f} | {icon}")

            # Gestion des erreurs détaillées
            if res['fatal']:
                print(f"   >>> ERREUR FATALE: {res['fatal']}")
            elif res['errors']:
                total_fails += 1
                # On affiche max 3 erreurs pour ne pas polluer
                for err in res['errors'][:3]:
                    print(f"   >>> {err}")
                if len(res['errors']) > 3:
                    print(f"   >>> ... et {len(res['errors']) - 3} autres.")

            # Sauvegarde pour le plot final
            if res['last_sample'] is not None:
                last_valid_data = res['last_sample']
                last_valid_name = res['filename']

    print("-" * 100)
    print(f"Terminé. Fichiers avec erreurs statistiques : {total_fails}")

    # 3. Plot final
    if args.no_plot:
        print("Affichage graphique désactivé via --no-plot")
    else:
        plot_sample(last_valid_data, last_valid_name)


def main():
    """
    Point d'entrée du script. 
    Parse les arguments de la ligne de commande et lance la fonction run.
    """
    parser = argparse.ArgumentParser(description="Script de vérification de la Z-Normalization (HDF5)")

    # Arguments de chemins
    parser.add_argument('-d1', '--dataset1', type=str, default='../../../data/physionnet/', 
                        help="Dossier principal contenant les fichiers HDF5 à vérifier.")
    
    parser.add_argument('-d2', '--dataset2', type=str, default='../../../data/15_prct/', 
                        help="Dossier de référence (dont on prendra 1 fichier témoin).")

    # Arguments de performance et config
    parser.add_argument('-n', '--n_samples', type=int, default=1000, 
                        help="Nombre d'échantillons à tester aléatoirement par fichier.")

    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count() - 1), 
                        help="Nombre de processus CPU parallèles.")

    parser.add_argument('--no-plot', action='store_true', 
                        help="Désactive l'affichage du graphique final (utile pour SSH).")

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
