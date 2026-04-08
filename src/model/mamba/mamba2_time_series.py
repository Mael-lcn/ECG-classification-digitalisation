import torch
import torch.nn as nn
from transformers import AutoModel



class mamba2_time_series(nn.Module):
    def __init__(
        self, 
        M2_in_channels=13, # 12 pistes ECG + 1 masque
        num_classes=27, 
        M2_hidden_dim=512,
        M2_dropout_classifier=0.4
    ):
        super().__init__()

        # 1. Chargement de Mamba-2 (Version 130m)
        repo_id = "state-spaces/mamba-2-130m"
        self.backbone = AutoModel.from_pretrained(
            repo_id, 
            trust_remote_code=True
        )

        # On récupère la dimension interne de Mamba
        mamba_dim = self.backbone.config.hidden_size 

        # 2. Couche de projection d'entrée
        # Convertit vos 13 canaux à chaque instant T vers la dimension de Mamba
        self.input_proj = nn.Sequential(
            nn.Conv1d(M2_in_channels, mamba_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 2),
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim)
        )

        # 3. Tête de Classification finale
        self.classifier = nn.Sequential(
            nn.Linear(mamba_dim, M2_hidden_dim),
            nn.BatchNorm1d(M2_hidden_dim),
            nn.ReLU(),
            nn.Dropout(M2_dropout_classifier),
            nn.Linear(M2_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask):
        # 1. On concatène l'ECG avec son masque -> [Batch, 13, 50000]
        combined_x = torch.cat([x, batch_mask], dim=1) 

        # 2. On passe dans notre mini-CNN d'abord
        x_features = self.input_proj(combined_x) # Sortie -> [Batch, 768, 50000]

        # 3. On prépare pour Mamba (qui veut [Batch, Temps, Canaux])
        x_emb = x_features.transpose(1, 2) # Sortie -> [Batch, 50000, 768]

        # 4. Passage dans Mamba
        outputs = self.backbone(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state

        # 5. Agrégation et Classification
        final_rep = hidden_states.mean(dim=1)
        logits = self.classifier(final_rep)

        return logits
