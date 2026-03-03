import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PatchTSTConfig, PatchTSTModel



class PatchTST_CrossAtt(nn.Module):
    def __init__(self,
                 in_channels=12,
                 context_length=1600,
                 patch_length=40,
                 stride=20,
                 d_model=128,
                 num_heads=8,
                 cross_att_heads=8,
                 encoder_layers=3,
                 revin=False,
                 num_classes=27, 
                 use_cross_att=True):
        super().__init__()

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

        self.backbone = PatchTSTModel(self.config)

        self.use_cross_att = use_cross_att
        if use_cross_att:
            self.cross_att = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=cross_att_heads, 
                batch_first=True
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, obs_mask=None, **kwargs):
        B, C, T = x.shape
        ctx_len = self.config.context_length

        # Cas 1 : SIGNAL COURT (<= context_length)
        if T <= ctx_len:
            pad_len = ctx_len - T
            x = F.pad(x, (0, pad_len))
            if obs_mask is not None:
                obs_mask = F.pad(obs_mask, (0, 0, 0, pad_len))

            x = x.transpose(1, 2)
            outputs = self.backbone(past_values=x, past_observed_mask=obs_mask)

            out = outputs.last_hidden_state 

            # 1. On moyenne les patchs pour obtenir 1 vecteur de contexte par canal -> [B, C, d_model]
            out = torch.mean(out, dim=2) 
    
            # 2. Vraie corss attention !
            if self.use_cross_att:
                # Q = Notre super token résumé global étendu au batch [B, 1, d_model]
                Q = self.query_token.expand(B, -1, -1)

                # K, V = Les 12 canaux [B, 12, d_model]
                out, _ = self.cross_att(query=Q, key=out, value=out)
                out = out.squeeze(1) # [B, d_model]
            else:
                out = torch.mean(out, dim=1) # [B, d_model]

            logits = self.classifier(out)
            return logits

        # Cas 2 : SIGNAL LONG (> context_length) -> FCNN
        else:
            num_chunks = math.ceil(T / ctx_len)
            pad_len = (num_chunks * ctx_len) - T
            x_padded = F.pad(x, (0, pad_len)) 

            x_chunks = x_padded.view(B * num_chunks, C, ctx_len)
            
            if obs_mask is not None:
                mask_padded = F.pad(obs_mask, (0, 0, 0, pad_len))
                mask_chunks = mask_padded.view(B * num_chunks, ctx_len, C)
            else:
                mask_chunks = None
                
            x_chunks = x_chunks.transpose(1, 2)
            outputs = self.backbone(past_values=x_chunks, past_observed_mask=mask_chunks)
            
            out = outputs.last_hidden_state 

            # Moyenne des patchs -> [B * num_chunks, 12, d_model]
            out = torch.mean(out, dim=2) 

            # 2.Cross attention
            if self.use_cross_att:
                B_current = out.shape[0]
                Q = self.query_token.expand(B_current, -1, -1)
                out, _ = self.cross_att(query=Q, key=out, value=out)
                out = out.squeeze(1) # [B * num_chunks, d_model]
            else:
                out = torch.mean(out, dim=1)

            # 3. Regroupement par examen et moyenne globale temporelle
            out = out.view(B, num_chunks, -1)
            out = torch.mean(out, dim=1) # [B, d_model]

            logits = self.classifier(out)
            return logits
