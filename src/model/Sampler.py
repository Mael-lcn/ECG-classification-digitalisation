from torch.utils.data import Sampler



class MegaBatchSortishSampler(Sampler):
    """
    Sampler personnalisé pour optimiser le padding des séquences de longueurs variables.

    Stratégie "Global Shuffle, Local Sort":
    1. Mélange globalement tous les indices du dataset (Random).
    2. Prélève un 'Mega-Batch' (ex: 50 * batch_size).
    3. Trie ce Mega-Batch par longueur de séquence.
    4. Découpe ce Mega-Batch trié en mini-batches finaux.
    """
    def __init__(self, dataset, batch_size, mega_batch_factor=20, shuffle=True):
        """
        La carte des longueurs (lengths) : Il doit savoir que l'indice 50 dure 3000 points et l'indice 51 dure 4500 points.
        C'est pour cela qu'on lui passe le Dataset.

        Les paramètres de structure :
        batch_size (64) : Pour savoir comment découper les groupes.
        mega_batch_factor (50) : Pour savoir quelle taille de l'ensemble d'indices il doit mélanger avant de trier.

        Args:
            dataset (Dataset): Ton dataset (doit avoir un moyen d'accéder à la longueur) plutot bas en train et long en eval.
            batch_size (int): La taille finale du batch pour le GPU.
            mega_batch_factor (int): Combien de batches on pré-charge pour le tri (défaut 50).
            shuffle (bool): True pour l'entrainement (mélange global), False pour la val.
        """
        # Initialisation des variables
        self.dataset = dataset
        self.batch_size = batch_size
        self.mega_batch_factor = mega_batch_factor
        self.shuffle = shuffle

        # On récupère la référence directe vers la liste des longueurs
        self.lengths = self.dataset.all_lengths

        # On définit la taille du "Méga-Lot" (ex: 50 * 64 = 3200 indices)
        self.mega_batch_size = self.batch_size * self.mega_batch_factor

        # Sécurité : vérifier que le dataset n'est pas vide
        if len(self.lengths) == 0:
            raise ValueError("Le dataset semble vide ou 'all_lengths' n'a pas été initialisé.")


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
        
        """
        Mélange Global (si self.shuffle est True) : Génères une liste de tous les indices de 0 à N-1 et la mélanges (avec indices = np.random.permutation(len(self.dataset)) ).

        Découpage en Mega-Batches : parcours cette liste mélangée par pas de mega_batch_size.

        Tri Local : Pour chaque Mega-Batch

        Récupère les longueurs de ces indices précis.

        Trie les indices de ce groupe du plus long au plus court.

        Création des mini-batches (Le "Out") : découpes ce groupe trié en morceaux de la taille de batch_size et fais un yield pour chaque morceau.
        """
        pass


    def __len__(self):
        """
        Renvoie le nombre total de batches dans une époque.
        """
        return (self.dataset.length + self.batch_size-1) // self.batch_size
