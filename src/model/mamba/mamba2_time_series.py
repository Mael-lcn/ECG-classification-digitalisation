import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel



class mamba2_time_series(nn.Module):
    def __init__(
        self, 
        M2_in_channels=12,  # Ignore le mask
        num_classes=27,
        M2_hidden_dim=512,
        M2_dropout_classifier=0.4
    ):
        super().__init__()

        config = AutoConfig.from_pretrained("state-spaces/mamba2-130m", trust_remote_code=True)

        config.d_model = 768
        config.hidden_size = 768
        config.num_heads = 24

        self.backbone = AutoModel.from_config(config, trust_remote_code=True)

        mamba_dim = 768

        #  (Stride=4 préservé pour ne pas perdre d'infos)
        self.input_proj = nn.Sequential(
            # --- Couche 1 : Extraction initiale ---
            # Noyau de 5 pour capter le contexte immédiat, réduit la longueur par 2
            nn.Conv1d(M2_in_channels, mamba_dim // 4, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 4),

            # --- Couche 2 : Extraction intermédiaire ---
            # Noyau de 3, réduit encore la longueur par 2 (Réduction totale = 4)
            nn.Conv1d(mamba_dim // 4, mamba_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 2),

            # --- Couche 3 : Raffinement sémantique ---
            # Noyau de 3, stride de 1 (pas de réduction). 
            # Projette vers la dimension finale de Mamba (768)
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim)
        )

        # 4. TÊTE DE CLASSIFICATION
        self.classifier = nn.Sequential(
            nn.Linear(mamba_dim, M2_hidden_dim),
            nn.BatchNorm1d(M2_hidden_dim),
            nn.ReLU(),
            nn.Dropout(M2_dropout_classifier),
            nn.Linear(M2_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask=None):
        x_features = self.input_proj(x) 

        x_emb = x_features.transpose(1, 2) 

        outputs = self.backbone(inputs_embeds=x_emb)
        hidden_states = outputs.last_hidden_state

        final_rep = hidden_states.mean(dim=1)
        logits = self.classifier(final_rep)

        return logits
