import argparse
import inspect
import multiprocessing

from Cnn import CNN
from Cnn_TimeFreq import CNN_TimeFreq



# Defini un registre de tout les modèles
MODEL_REGISTRY = {
    "cnn_base": CNN,
    "cnn_spectro": CNN_TimeFreq
}


def get_shared_parser():
    """Retourne un parser avec les arguments communs à Train et Eval."""
    parser = argparse.ArgumentParser(add_help=False)

    # --- 1. PARAMÈTRES SYSTÈME ET DONNÉES ---
    group_sys = parser.add_argument_group("Configuration Système & Fichiers")
    group_sys.add_argument('--output', type=str, default='../../../output/',
                           help="Dossier de sortie standard")
    group_sys.add_argument('--class_map', type=str, default='../../ressources/final_class.json',
                           help="Chemin JSON mappant les indices aux noms de classes")    
    group_sys.add_argument('--workers', type=int, default=min(8, multiprocessing.cpu_count()-1),
                           help="Nombre de processus pour charger les données")

    # --- 2. Choix du modèle ---
    group_model = parser.add_argument_group("Sélection du Modèle")
    group_model.add_argument('--model_name', type=str, default="cnn_base", choices=MODEL_REGISTRY.keys(),
                             help="Nom du modèle à utiliser")

    # --- 3. Architecture partagés par CNN et CNN_TimeFreq ---
    group_arch = parser.add_argument_group("Hyperparamètres de l'Architecture (Communs)")
    group_arch.add_argument('--num_classes', type=int, default=27, help="Nombre de classes de sortie")
    group_arch.add_argument('--in_channels', type=int, default=12, help="Nombre de dérivations ECG (leads)")
    group_arch.add_argument('--ch1', type=int, default=32, help="Filtres de la 1ère couche convolutive")
    group_arch.add_argument('--ch2', type=int, default=64, help="Filtres de la 2ème couche convolutive")
    group_arch.add_argument('--ch3', type=int, default=128, help="Filtres de la 3ème couche convolutive")
    group_arch.add_argument('--dropout', type=float, default=0.5, help="Taux de dropout")

    # Gestion du booléen True par défaut
    group_arch.add_argument('--no_batchnorm', dest='use_batchnorm', action='store_false', default=True,
                            help="Désactive le Batch Normalization (activé par défaut)")
    group_arch.add_argument('--use_fcnn', action='store_true', default=False,
                            help="Active l'architecture FCNN (fenêtre glissante) au lieu de CNN standard")

    # --- 4. Architecture spécifique : CNN 1D ---
    group_cnn1d = parser.add_argument_group("Spécifique au CNN Standard (1D)")
    group_cnn1d.add_argument('--kernel_size', type=int, default=3, help="Taille du kernel de convolution (1D)")
    group_cnn1d.add_argument('--window_size1D', type=int, default=4000, 
                             help="Taille de la fenêtre pour le classifieur FCNN 1D (kernel_size du classifieur)")

    # --- 5. Architecture spécifique : CNN TimeFreq ---
    group_timefreq = parser.add_argument_group("Spécifique au CNN Time-Frequency (Spectrogramme)")
    group_timefreq.add_argument('--n_fft', type=int, default=128, help="Points utilisés pour la transformée de Fourier")
    group_timefreq.add_argument('--hop_length', type=int, default=64, help="Pas de glissement de la fenêtre (stride)")
    group_timefreq.add_argument('--win_length', type=int, default=128, help="Taille de la fenêtre en échantillons")
    group_timefreq.add_argument('--window_size2D', type=int, nargs='+', default=[4, 4],
                                help="Taille de la fenêtre (freq, time) pour le classifieur FCNN 2D. Ex: --window_size_2d 4 4")

    # --- 6. Paramètres du dataloading communs ---
    group_train = parser.add_argument_group("Hyperparamètres Communs du Dataloader/Inférence")
    group_train.add_argument('--batch_size', type=int, default=64, help="Taille du batch")
    group_train.add_argument('--mega_batch_factor', type=int, default=16,
                             help="Granularité du tri. Haut = padding optimisé, Bas = + d'aléatoire")
    group_train.add_argument('--use_static_padding', action='store_true', default=False,
                             help="Force une taille de padding fixe (universelle)")
    group_train.add_argument('--not_use_amp', action='store_false', default=True,
                             help="Désactive l'Automatic Mixed Precision (AMP). Passe en FP32.")

    return parser


def build_model(args_namespace):
    """Instancie le modèle en filtrant dynamiquement les arguments."""
    args_dict = vars(args_namespace)
    model_name = args_dict.get('model_name')

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu. Choix : {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_name]

    # Introspection pour ne donner au modèle que ce qu'il demande
    sig = inspect.signature(ModelClass.__init__)
    valid_kwargs = {k: v for k, v in args_dict.items() if k in sig.parameters}

    print(f"Le modèle: {model_name} à bien été instancié")
    return ModelClass(**valid_kwargs)
