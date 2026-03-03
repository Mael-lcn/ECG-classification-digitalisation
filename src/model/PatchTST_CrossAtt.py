import torch
import torch.nn as nn

from transformers import PatchTSTConfig, PatchTSTModel



class PatchTST_CrossAtt(nn.Module):
    def __init__(self,
                 in_channels=12,
                 context_length=1600,
                 patch_length=40,
                 stride=20,
                 d_model=128,
                 num_heads=8,
                 encoder_layers=3,
                 revin=False,
                 num_classes=27, 
                 use_cross_att=True):
        super().__init__()

        # Configuration de la backbone
        self.config = PatchTSTConfig(
            num_input_channels=in_channels,
            context_length=context_length,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            revin=revin
        )

        # Backbone
        self.backbone = PatchTSTModel(self.config)

        # Cross-Channel Attention (Mélange les 12 canaux)
        self.use_cross_att = use_cross_att
        if use_cross_att:
            self.cross_att = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=8, 
                batch_first=True
            )

        # Tête de classification (MLP à 2 couches)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, obs_mask=None):   
        x = x.transpose(1, 2)     
        # Passage dans le Transformer
        outputs = self.backbone(past_values=x, past_observed_mask=obs_mask)
        x = outputs.last_hidden_state 

        # Optionnel : Cross-Attention entre les canaux
        if self.use_cross_att:
            x, _ = self.cross_att(x, x, x)

        # Global Average Pooling
        # On passe de [B, N_patches*C, 128] à [B, 128]
        x = torch.mean(x, dim=1) 

        # MLP Final
        logits = self.classifier(x)

        return logits
