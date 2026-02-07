from torch.utils.data import Sampler
import numpy as np



class MegaBatchSortishSampler(Sampler):
    """
    Sampler personnalisé pour optimiser le padding des séquences de longueurs variables.
    
    Stratégie "Global Shuffle, Local Sort" :
    1. Mélange globalement tous les indices du dataset (Random).
    2. Prélève un 'Mega-Batch' (ex: 50 * batch_size).
    3. Trie ce Mega-Batch par longueur de séquence.
    4. Découpe ce Mega-Batch trié en mini-batches finaux.
    """
    def __init__(self, dataset, batch_size, mega_batch_factor=50, shuffle=True):
        """
        Args:
            dataset (Dataset): Ton dataset (doit avoir un moyen d'accéder à la longueur).
            batch_size (int): La taille finale du batch pour le GPU.
            mega_batch_factor (int): Combien de batches on pré-charge pour le tri (défaut 50).
            shuffle (bool): True pour l'entrainement (mélange global), False pour la val.
        """
        # Initialisation des variables
        pass


    def __iter__(self):
        """
        Générateur principal appelé par le DataLoader à chaque époque.
        
        Logique à implémenter :
        1. Créer une liste d'indices de 0 à N.
        2. Si shuffle=True, mélanger cette liste aléatoirement.
        3. Boucler sur cette liste par pas de (batch_size * mega_batch_factor).
        4. Pour chaque 'Mega-Chunk' :
           a. Récupérer les longueurs de chaque indice via le dataset.
           b. Trier les indices du Mega-Chunk par longueur (descendant).
           c. Découper ce Mega-Chunk trié en petits morceaux de 'batch_size'.
           d. YIELD (renvoyer) chaque petit morceau (liste d'indices).
        """
        pass


    def __len__(self):
        """
        Renvoie le nombre total de batches dans une époque.
        Calcul : (Taille Dataset + batch_size - 1) // batch_size
        """
        pass
