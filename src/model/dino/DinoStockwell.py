import torch
import torch.nn as nn
import torch.nn.functional as F



class DinoStockwell(nn.Module):
    """
    DinoStockwell Mask-Aware
    Architecture hybride Signal-Vision conçue pour la classification d'ECG 12-dérivations.
    Combine la précision temps-fréquence de la Transformée de Stockwell avec la 
    puissance d'extraction de caractéristiques morphologiques de DINOv3.

    Le modèle traite les variations de longueur de signal via un masquage strict pour 
    garantir que le padding n'influence pas les statistiques de normalisation ni 
    l'attention du backbone ViT.
    """
    def __init__(self, num_classes=27, vit_type='dinov3_vitb16', n_fft=256):
        super().__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1

        # Pré-calcul des noyaux gaussiens de Stockwell pour éviter les calculs redondants sur GPU
        # La fenêtre S s'adapte à la fréquence : sigma = 1/f
        self.register_buffer('s_kernels', self._generate_s_kernels(n_fft))

        # Adaptateur de Domaine Conv 1x1 : Apprend à combiner les 12 leads vers 3 canaux sémantiques (RGB).
        self.lead_projector = nn.Conv2d(12, 3, kernel_size=1)

        # Backbone de Vision : DINOv3 ✨
        self.backbone = torch.hub.load('facebookresearch/dinov3', vit_type)
        self.embed_dim = self.backbone.embed_dim

        # Tête de Classification
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def _generate_s_kernels(self, n):
        """
        Génère mathématiquement les fenêtres spectrales pour la transformée S.
        Résolution élevée pour les hautes fréquences (QRS) et basse pour les ondes lentes (T).
        """
        f = torch.arange(n // 2 + 1).view(-1, 1)
        tau = torch.arange(n).view(1, -1) - (n // 2)
        f_safe = f.clone(); f_safe[0] = 1.0  # Évite la division par zéro à la fréquence DC

        # Formule Gaussienne de Stockwell : exp(-2 * pi^2 * tau^2 / f^2)
        gauss = torch.exp(-2 * (torch.pi**2) * (tau**2) / (f_safe**2))
        gauss[0] = 1.0 # Fenêtre plate pour la composante continue
        return gauss / gauss.sum(dim=1, keepdim=True)

    def _masked_norm(self, s_mag, batch_mask):
        """
        Effectue une standardisation (Z-Score) rigoureuse en ignorant le padding.
        
        Args:
            s_mag (Tensor): Magnitude du spectrogramme [B, L, F, T]
            batch_mask (Tensor): Masque binaire [B, T] (1.0 = Signal utile)
        """
        B, L, F, T = s_mag.shape
        mask = batch_mask.view(B, 1, 1, T) # Alignement dimensionnel pour broadcasting

        # Calcul du dénominateur : nombre exact de pixels 'actifs' dans le spectrogramme
        n_elements = mask.sum(dim=-1, keepdim=True) * F * L

        # Calcul de la moyenne masquée (uniquement sur le signal réel)
        sum_val = (s_mag * mask).sum(dim=(1, 2, 3), keepdim=True)
        mean = sum_val / n_elements

        # Calcul de l'écart-type masqué
        var_val = (((s_mag - mean) ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(var_val / n_elements + 1e-8)

        # Normalisation et remise à zéro forcée du padding pour la propreté visuelle
        s_norm = (s_mag - mean) / std
        return s_norm * mask

    def forward(self, x, batch_mask):
        """
        1. Stockwell GPU : Passage au domaine temps-fréquence via convolution spectrale.
        2. Log-Compression : Réduction de la dynamique (essentiel pour voir les ondes P/T).
        3. Masked Norm : Normalisation statistique indépendante du ratio de padding.
        4. Interpolation : Standardisation temporelle vers le format 224x224 de DINOv3.
        """
        B, L, T = x.shape

        # Transformation de Stockwell Vectorisée
        x_flat = x.view(B * L, T)
        X_freq = torch.fft.fft(x_flat, n=self.n_fft) # Passage en fréquence

        # Multiplication complexe par la banque de filtres Gaussienne
        s_complex = torch.fft.ifft(X_freq.unsqueeze(1) * self.s_kernels.unsqueeze(0))
        s_mag = torch.abs(s_complex).view(B, L, self.freq_bins, T)

        # Compression logarithmique et Normalisation "Mask-Aware"
        s_mag = torch.log1p(s_mag)
        s_norm = self._masked_norm(s_mag, batch_mask)

        # --- Section Vision ---
        # Projection 12 leads -> 3 canaux RGB
        s_rgb = self.lead_projector(s_norm)

        # Resize Bilinéaire : Étire la zone utile du signal sur les 224 pixels d'entrée du ViT
        x_vision = F.interpolate(s_rgb, size=(224, 224), mode='bilinear')

        # --- Extraction & Classification ---
        features = self.backbone(x_vision)
        return self.head(features)
