import torch
import torch.nn as nn
import torch.hub



class DinoTraceTemporal(nn.Module):
    """
    Modèle DINOv3 avec gestion VRAM stricte (Micro-Batching),
    suivi d'une Attention Temporelle, Max Pooling et classification MLP.
    """
    def __init__(
        self, 
        num_classes=26, 
        vit_type='dinov3_vitb16',
        D_hidden_dim=512,
        D_dropout_classifier=0.4,
        sub_batch_size=16,
        D_nhead=8
    ):
        super().__init__()
        self.sub_batch_size = sub_batch_size

        # Backbone DINOv3
        self.backbone = torch.hub.load('facebookresearch/dinov3', vit_type)        
        embed_dim = self.backbone.embed_dim 

        # Gel total du backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Buffers de Normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Attention Temporelle (Self/Cross-Attention sur la séquence)
        # Permet aux différents segments ECG d'un même patient de s'influencer mutuellement
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=D_nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.15,
            batch_first=True
        )

        # Tête de Classification finale (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, D_hidden_dim),
            nn.BatchNorm1d(D_hidden_dim),
            nn.ReLU(),
            nn.Dropout(D_dropout_classifier),
            nn.Linear(D_hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [Batch, Segments, 3, H, W] - Images uint8 ou float
        """
        B, S, C, H, W = x.shape
        # Aplatissement 5D -> 4D pour traitement indépendant par DINO
        x_flat = x.view(B * S, C, H, W)

        all_features = []

        with torch.no_grad():
            for i in range(0, B * S, self.sub_batch_size):
                chunk = x_flat[i : i + self.sub_batch_size]

                # Conversion et Normalisation au dernier moment
                if chunk.dtype == torch.uint8:
                    chunk = chunk.float() / 255.0
                chunk = (chunk - self.mean) / self.std

                # Extraction DINO
                feat = self.backbone(chunk)
                all_features.append(feat)

        # Recombinaison temporelle des blocs
        features = torch.cat(all_features, dim=0).view(B, S, -1)

        # Attention Temporelle
        # Chaque fenêtre ECG interagit avec les autres pour capter le rythme global
        attn_out = self.temporal_attention(features)

        # Agrégation MAX
        final_rep, _ = torch.max(attn_out, dim=1) 

        # MLP Classifieur
        logits = self.classifier(final_rep)

        return logits
