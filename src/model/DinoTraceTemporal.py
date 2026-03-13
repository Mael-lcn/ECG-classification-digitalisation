import torch
import torch.nn as nn
import torch.hub



class DinoTraceTemporal(nn.Module):
    """
    Architecture hiérarchique combinant l'extraction de caractéristiques spatiales (DINOv2)
    et temporelles (Transformer) pour la classification de séries temporelles ECG multi-dérivations.

    Le modèle traite les ECG sous forme de séquences d'images (fenêtrage). Il utilise une
    approche de sous-batching (micro-batching) en interne pour éviter la saturation de
    la VRAM lors du passage dans le ViT.

    Attributes:
        adapter (nn.Conv2d): Projette les 12 dérivations ECG vers 3 canaux (format RGB attendu par ViT).
        adapter_norm (nn.GroupNorm): Normalisation spatiale robuste aux variations de taille de batch.
        backbone (nn.Module): Le modèle DINOv2 pré-entraîné agissant comme extracteur de caractéristiques locales (morphologie).
        pos_embedding (nn.Parameter): Encodage de position absolu pour préserver la chronologie des fenêtres.
        temporal_transformer (nn.TransformerEncoder): Encodeur temporel pour capturer les relations rythmiques globales.
        classifier (nn.Sequential): Perceptron multicouche (MLP) pour la classification finale.
    """

    def __init__(
        self, 
        num_classes = 26, 
        vit_type = 'dinov2_vitb14',
        D_hidden_dim = 512,
        D_num_temp_layers = 2,
        D_nhead = 12,
        D_dropout_transformer = 0.3,
        D_dropout_classifier = 0.3,
        D_max_images = 20,
        unfreeze_blocks = 2,
        sub_batch_size = 4
    ):
        """
        Initialise le modèle DinoTraceTemporal.

        Args:
            num_classes (int): Nombre de pathologies/classes à prédire.
            vit_type (str): Version de DINOv2 à charger via Torch Hub (ex: 'dinov2_vits14', 'dinov2_vitb14').
            hidden_dim (int): Dimension de la couche cachée du classifieur final (MLP).
            num_temp_layers (int): Nombre de couches de l'encodeur Transformer temporel.
            nhead (int): Nombre de têtes d'attention pour le Transformer temporel.
            dropout_transformer (float): Taux de dropout dans le Transformer temporel.
            dropout_classifier (float): Taux de dropout dans le classifieur final.
            max_images (int): Longueur maximale de la séquence d'images (pour allouer le Positional Encoding).
            unfreeze_blocks (int): Nombre de blocs terminaux de DINOv2 à dégeler pour le fine-tuning. 0 = gel total.
            sub_batch_size (int): Nombre d'images traitées simultanément par DINOv2 lors du forward (Anti-OOM).
        """
        super().__init__()
        self.sub_batch_size = sub_batch_size

        # Adaptateur Spatial : Compression des 12 leads vers 3 canaux "RGB"
        self.adapter = nn.Conv2d(12, 3, kernel_size=1)
        # GroupNorm car le sous-batching fausserait les statistiques de moyenne/variance d'un BatchNorm classique
        self.adapter_norm = nn.GroupNorm(1, 3) 

        # Extracteur de features DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', vit_type)
        embed_dim = self.backbone.embed_dim 

        # Stratégie de Fine-Tuning Partiel
        # On gèle les couches basses et on libère les couches hautes
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        if unfreeze_blocks > 0:
            # Dégel des N derniers blocs du Transformer
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Dégel de la couche de normalisation finale de DINOv2
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        # 3. Encodage de Position Temporel
        # Permet au Transformer Temporel de différencier le début, le milieu et la fin du signal ECG.
        self.pos_embedding = nn.Parameter(torch.empty(1, D_max_images, embed_dim))
        # Initialisation standard de FAIR (Facebook AI Research) pour les architectures ViT
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Analyse les relations entre les fenêtres morphologiques
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=D_nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=D_dropout_transformer,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=D_num_temp_layers)

        # 5. Tête de Classification (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, D_hidden_dim),
            nn.ReLU(),
            nn.Dropout(D_dropout_classifier),
            nn.Linear(D_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask=None):
        """
        Passe avant du modèle avec gestion sécurisée de la mémoire (Anti-OOM).

        Args:
            x (torch.Tensor): Tenseur d'entrée de forme [Batch, Images, Leads, Hauteur, Largeur].
                              Exemple: [64, 13, 12, 518, 518].
            mask (torch.Tensor, optional): Masque booléen de forme [Batch, Images] pour ignorer le padding
                                           des signaux ECG plus courts que la séquence maximale.

        Returns:
            torch.Tensor: Logits de classification de forme [Batch, num_classes].
        """
        B, I, C, H, W = x.shape

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Aplatissement 5D -> 4D pour traitement indépendant de chaque image par le ViT
        x_flat = x.view(B * I, C, H, W)

        all_features = []

        # --- Phase 1 : backbone ---
        for i in range(0, B * I, self.sub_batch_size):
            chunk = x_flat[i : i + self.sub_batch_size]

            # Adaptation spatiale
            chunk = self.adapter(chunk)
            chunk = self.adapter_norm(chunk)

            # Extraction DINOv2
            feat = self.backbone(chunk)
            all_features.append(feat)

        # Recombinaison temporelle : [Batch * Images, Dim] -> [Batch, Images, Dim]
        features = torch.cat(all_features, dim=0)
        features = features.view(B, I, -1)

        # --- Phase 2 : Annalyse global par patient ---
        # Injection du contexte chronologique
        features = features + self.pos_embedding[:, :I, :]

        # Attention croisée temporelle (ignore les images vides via le src_key_padding_mask)
        temporal_out = self.temporal_transformer(features, src_key_padding_mask=batch_mask)

        # --- Phase 3 : Agégation et classification ---
        final_rep, _ = torch.max(temporal_out, dim=1)

        # Calcul des logits finaux
        return self.classifier(final_rep)
