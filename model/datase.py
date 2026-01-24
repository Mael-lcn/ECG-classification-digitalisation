import h5py
import glob
import os
import torch
from torch.utils.data import Dataset
import bisect



class LargeH5Dataset(Dataset):
    """
    Dataset PyTorch optimisé pour la lecture de multiples fichiers HDF5 volumineux.
    """
    def __init__(self, input_dir):
        """
            Initialisation et Scan des métadonnées
        """
        # Récupération de tous les fichiers .hdf5
        self.files_paths = glob.glob(os.path.join(input_dir, '*.hdf5'))

        self.files_paths.sort()

        # Structures pour la "Carte Globale" des données
        self.file_sizes = []        # Taille individuelle de chaque fichier
        self.cumulative_sizes = []  # Index de fin cumulés
        cumul = 0

        print("--- Initialisation du Dataset ---")
        print("Scan des fichiers en cours (lecture des headers uniquement)...")

        # On parcourt chaque fichier pour construire l'index
        for p in self.files_paths:
            with h5py.File(p, 'r') as f:
                n_samples = f['exam_id'].shape[0]

            self.file_sizes.append(n_samples)
            cumul += n_samples
            self.cumulative_sizes.append(cumul)

        self.total_length = cumul
        print(f"Terminé. Dataset total : {self.total_length} échantillons sur {len(self.files_paths)} fichiers.")

        self.file_handle = None       # L'objet fichier h5py ouvert
        self.current_file_idx = None  # L'index du fichier actuellement ouvert


    def __len__(self):
        """Retourne la taille totale virtuelle du dataset (somme de tous les fichiers)."""
        return self.total_length


    def __getitem__(self, idx):
        """
        Phase 3 : Chargement à la demande (appelé par le DataLoader)
        Transforme un index global (ex: 3500) en donnée réelle.
        """
        # 1. Localisation du fichier (Mapping Global -> Fichier)
        # dans quel intervalle se situe 'idx'. Beaucoup plus rapide qu'une boucle for.
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # 2. Calcul de l'index local (Mapping Global -> Local)
        # Si idx=1500 et que le fichier précédent finissait à 1000, 
        # on veut l'élément 500 du fichier actuel.
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        # 3. On vérifie si le fichier nécessaire est déjà ouvert.
        if self.current_file_idx != file_idx:
            # Si un autre fichier était ouvert, on le ferme proprement
            if self.file_handle is not None:
                self.file_handle.close()

            # On ouvre le nouveau fichier correspondant
            self.file_handle = h5py.File(self.files_paths[file_idx], 'r')
            self.current_file_idx = file_idx

        # 4. Lecture réelle sur le disque
        tracing = self.file_handle['tracings'][local_idx]
        exam_id = self.file_handle['exam_id'][local_idx]

        # 5. Conversion en Tensor PyTorch
        # torch.from_numpy crée un Tensor qui partage la mémoire si possible
        return torch.from_numpy(tracing), str(exam_id)


    def __del__(self):
        """
        Destructeur sécurité pour le nettoyage.
        S'assure que le dernier fichier ouvert est bien refermé quand
        l'objet Dataset est détruit (fin du script ou fin de l'epoch).
        """
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass
