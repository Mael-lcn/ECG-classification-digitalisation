import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



class DinoStockwell(nn.Module):
    """
    Architecture hybride Signal-Vision pour la classification d'électrocardiogrammes (ECG) à 12 dérivations.

    Ce modèle combine la précision de la Transformée de Stockwell (domaine temps-fréquence) avec 
    les capacités d'extraction de caractéristiques d'un Vision Transformer (DINOv3). Il intègre 
    une gestion stricte du masquage pour traiter les signaux de longueur variable sans introduire 
    de biais statistique lié au padding. L'extraction finale fusionne le token [CLS] et le 
    Max Pooling des patchs spatiaux pour capturer simultanément le contexte global et les anomalies locales.
    """

    def __init__(self, num_classes=27, n_fft=256):
        """
        Initialise le modèle DinoStockwell.

        Args:
            num_classes (int): Nombre de classes de sortie pour la classification.
            n_fft (int): Nombre de points pour la Transformée de Fourier Rapide (FFT), 
                         définissant la résolution fréquentielle.
        """
        super().__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1

        self.register_buffer('s_kernels', self._generate_s_kernels(n_fft))
        self.lead_projector = nn.Conv2d(12, 3, kernel_size=1)

        repo_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.backbone = AutoModel.from_pretrained(
            repo_id, 
            local_files_only=True,
            attn_implementation="sdpa" 
        )        

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, num_classes)
        )

    def _generate_s_kernels(self, n):
        """
        Génère les noyaux spectraux (fenêtres gaussiennes) pour la Transformée de Stockwell.

        La transformée de Stockwell utilise des fenêtres gaussiennes dont l'écart-type est inversement 
        proportionnel à la fréquence, offrant une haute résolution temporelle pour les hautes fréquences 
        et une haute résolution fréquentielle pour les basses fréquences.

        Args:
            n (int): Nombre de points de la FFT.

        Returns:
            torch.Tensor: Tenseur des noyaux gaussiens normalisés de forme (n // 2 + 1, n).
        """
        f = torch.arange(n // 2 + 1).view(-1, 1)
        tau = torch.arange(n).view(1, -1) - (n // 2)
        
        f_safe = f.clone()
        f_safe[0] = 1.0  

        gauss = torch.exp(-2 * (torch.pi**2) * (tau**2) / (f_safe**2))
        gauss[0] = 1.0 
        
        return gauss / gauss.sum(dim=1, keepdim=True)

    def _masked_norm(self, s_mag, batch_mask):
        """
        Applique une normalisation statistique (Z-score) en ignorant les zones de padding.

        Calcule la moyenne et l'écart-type uniquement sur les parties valides du signal, 
        puis normalise le spectrogramme. Les zones de padding sont remises à zéro par la suite.

        Args:
            s_mag (torch.Tensor): Tenseur des magnitudes du spectrogramme de forme (B, L, F, T).
            batch_mask (torch.Tensor): Masque binaire indiquant le signal valide de forme (B, T).

        Returns:
            torch.Tensor: Spectrogramme normalisé et masqué de même forme que s_mag.
        """
        B, L, F, T = s_mag.shape
        mask = batch_mask.view(B, 1, 1, T) 

        n_elements = mask.sum(dim=-1, keepdim=True) * F * L

        sum_val = (s_mag * mask).sum(dim=(1, 2, 3), keepdim=True)
        mean = sum_val / n_elements

        var_val = (((s_mag - mean) ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(var_val / n_elements + 1e-8)

        s_norm = (s_mag - mean) / std
        
        return s_norm * mask

    def forward(self, x, batch_mask):
        """
        Exécute la propagation avant (forward pass) du modèle.

        Étapes de traitement :
        1. Transformation de Stockwell pour le passage dans le domaine temps-fréquence.
        2. Compression logarithmique et normalisation stricte via masque.
        3. Projection des dérivations et redimensionnement pour l'encodeur ViT.
        4. Extraction des caractéristiques ViT avec concaténation du token [CLS] et du Max Pooling spatial.

        Args:
            x (torch.Tensor): Tenseur d'entrée des signaux ECG de forme (B, L, T).
            batch_mask (torch.Tensor): Masque binaire associé aux signaux de forme (B, T).

        Returns:
            torch.Tensor: Logits de classification de forme (B, num_classes).
        """
        B, L, T = x.shape

        x_flat = x.view(B * L, T)
        X_freq = torch.fft.fft(x_flat, n=self.n_fft) 

        s_complex = torch.fft.ifft(X_freq.unsqueeze(1) * self.s_kernels.unsqueeze(0))
        s_mag = torch.abs(s_complex).view(B, L, self.freq_bins, T)

        s_mag = torch.log1p(s_mag)
        s_norm = self._masked_norm(s_mag, batch_mask)

        s_rgb = self.lead_projector(s_norm)
        x_vision = F.interpolate(s_rgb, size=(224, 224), mode='bilinear')

        outputs = self.backbone(x_vision)
        last_hidden_state = outputs.last_hidden_state 
        
        cls_token = last_hidden_state[:, 0, :] 
        patch_tokens = last_hidden_state[:, 1:, :] 
        
        max_patches = patch_tokens.max(dim=1)[0] 
        
        combined_features = torch.cat([cls_token, max_patches], dim=-1) 
        
        return self.head(combined_features)
