import torch.nn as nn
from mamba_ssm import Mamba2



class Mamba2_time_series(nn.Module):
    def __init__(
        self, 
        M2_in_channels=12,      # 12 dérivations de l'ECG
        num_classes=27,
        M2_hidden_dim=1024,      # Dimension de la couche cachée du classifieur
        M2_dropout_classifier=0.4
    ):
        super().__init__()

        # Dimension interne envoyée à Mamba (doit être un multiple de headdim)
        mamba_dim = 768

        # 1. Projection d'entrée
        # Prend [Batch, 12, Longueur] et sort [Batch, 768, Longueur/4]
        self.input_proj = nn.Sequential(
            nn.Conv1d(M2_in_channels, mamba_dim // 4, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 4),

            nn.Conv1d(mamba_dim // 4, mamba_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 2),

            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim)
        )

        #  2. Backbone Mamba 2
        # Configuration stricte pour activer les noyaux CUDA ultra-rapides
        self.backbone = Mamba2(
            d_model=mamba_dim, # 768
            d_state=128,       # Dimension de l'état caché (128 est le standard optimal)
            d_conv=4,          # Largeur du filtre convolutif interne de Mamba
            expand=2,          # Facteur d'expansion
            headdim=128,
        )

        #  3. MLP classif
        self.classifier = nn.Sequential(
            nn.Linear(mamba_dim, M2_hidden_dim),
            nn.BatchNorm1d(M2_hidden_dim),
            nn.ReLU(),
            nn.Dropout(M2_dropout_classifier),
            nn.Linear(M2_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask=None, **kwargs):
        # 1. Extraction des features locales (CNN)
        x_features = self.input_proj(x) 

        # 2. Transposition pour Mamba
        # Mamba attend le format [Batch, Longueur_Temporelle, Dimension]
        x_emb = x_features.transpose(1, 2) 

        # 3. Passage dans Mamba 2
        # Contrairement à HF, le Mamba2 officiel renvoie directement le tenseur caché
        hidden_states = self.backbone(x_emb) 

        # 4. Pooling temporel (Moyenne sur toute la séquence)
        # On passe de [Batch, Longueur, 768] à [Batch, 768]
        final_rep = hidden_states.mean(dim=1)

        # 5. Classification finale
        logits = self.classifier(final_rep)

        return logits
