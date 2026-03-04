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

        # Cas 1 : Signal inférieur ou égal à la taille de contexte
        if T <= ctx_len:
            pad_len = ctx_len - T
            x = F.pad(x, (0, pad_len))
            if obs_mask is not None:
                obs_mask = F.pad(obs_mask, (0, 0, 0, pad_len))

            x = x.transpose(1, 2)
            outputs = self.backbone(past_values=x, past_observed_mask=obs_mask)

            out = outputs.last_hidden_state 
            out = torch.mean(out, dim=2) 

            if self.use_cross_att:
                Q = self.query_token.expand(B, -1, -1)
                out, _ = self.cross_att(query=Q, key=out, value=out)
                out = out.squeeze(1) 
            else:
                out = torch.mean(out, dim=1) 

            logits = self.classifier(out)
            return logits

        # Cas 2 : Signal long traité par mini-lots de blocs (chunks)
        else:
            num_chunks = math.ceil(T / ctx_len)
            pad_len = (num_chunks * ctx_len) - T

            # Application du padding pour obtenir un multiple exact de ctx_len
            x_padded = F.pad(x, (0, pad_len)) 
            x_chunks = x_padded.view(B, num_chunks, C, ctx_len)

            if obs_mask is not None:
                mask_padded = F.pad(obs_mask, (0, 0, 0, pad_len))
                mask_chunks = mask_padded.view(B, num_chunks, ctx_len, C)
            else:
                mask_chunks = None
                
            chunk_outputs = []
            chunk_batch_size = 1  # Paramètre d'optimisation : nombre de blocs traités simultanément

            # Itération sur les blocs avec un pas de 'chunk_batch_size'
            for i in range(0, num_chunks, chunk_batch_size):
                # Extraction des blocs courants (peut être inférieur à chunk_batch_size à la fin)
                x_i = x_chunks[:, i:i+chunk_batch_size, :, :]
                current_step = x_i.shape[1]
    
                # Fusion des dimensions Batch et Chunks pour le passage dans le modèle
                # Shape finale attendue par HF : [B * current_step, ctx_len, C]
                x_i_flat = x_i.reshape(B * current_step, C, ctx_len).transpose(1, 2)

                if mask_chunks is not None:
                    mask_i = mask_chunks[:, i:i+chunk_batch_size, :, :]
                    mask_i_flat = mask_i.reshape(B * current_step, ctx_len, C)
                else:
                    mask_i_flat = None

                # Passage dans le Transformer
                outputs_i = self.backbone(past_values=x_i_flat, past_observed_mask=mask_i_flat)

                # Récupération de l'état caché : [B * current_step, C, N_patches, d_model]
                out_i = outputs_i.last_hidden_state 

                # Moyenne spatiale sur les patchs : [B * current_step, C, d_model]
                out_i = torch.mean(out_i, dim=2) 

                # Restauration des dimensions : [B, current_step, C, d_model]
                out_i = out_i.view(B, current_step, C, -1)
                chunk_outputs.append(out_i)

            # Concaténation de tous les sous-groupes de blocs : [B, num_chunks, C, d_model]
            out = torch.cat(chunk_outputs, dim=1) 

            # Moyenne temporelle sur l'ensemble des blocs : [B, C, d_model]
            out = torch.max(out, dim=1) 

            # Application de la Cross-Attention
            if self.use_cross_att:
                Q = self.query_token.expand(B, -1, -1)
                out, _ = self.cross_att(query=Q, key=out, value=out)
                out = out.squeeze(1) 
            else:
                out = torch.mean(out, dim=1) 

            # Classification finale
            logits = self.classifier(out)
            return logits
