import h5py
import numpy as np
import os
import argparse



def inspect_h5(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    print(f"\n{'='*60}")
    print(f"INSPECTION DE : {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        with h5py.File(file_path, 'r') as f:
            def scan_node(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\nDataset: {name}")
                    print(f"   Shape: {obj.shape}")
                    print(f"   Dtype: {obj.dtype}")

                    # On ne charge qu'un échantillon ou les métadonnées pour ne pas saturer la RAM
                    data_shape = obj.shape

                    # 1. Anomalie de Dimension
                    if len(data_shape) == 3 and name == 'tracings':
                        # Alerte si T est délirant (ex: > 100 000 points)
                        if data_shape[2] > 100000:
                            print(f"   ANOMALIE: Dimension temporelle suspecte (> 100k points) !")

                    # 2. Vérification du contenu (sur le premier échantillon pour aller vite)
                    sample = obj[0]
                    if isinstance(sample, np.ndarray) and np.issubdtype(sample.dtype, np.number):
                        if np.isnan(sample).any():
                            print(f"   ANOMALIE: Valeurs NaN détectées dans le premier échantillon.")
                        if np.isinf(sample).any():
                            print(f"   ANOMALIE: Valeurs Infinies détectées.")
                        
                        v_min, v_max = np.min(sample), np.max(sample)
                        print(f"   Range (sample 0): [{v_min:.4f} , {v_max:.4f}]")
                        
                        if v_min == v_max == 0:
                            print(f"   ATTENTION: Le premier échantillon est totalement vide (que des zéros).")

                elif isinstance(obj, h5py.Group):
                    print(f"\nGroupe: {name}")

            f.visititems(scan_node)

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Utilisation : python inspect_h5.py mon_fichier.hdf5
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="Chemin vers le fichier .hdf5")
    args = parser.parse_args()
    
    inspect_h5(args.file)
