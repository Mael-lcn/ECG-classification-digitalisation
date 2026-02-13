import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        ch1: int = 32,
        ch2: int = 64,
        ch3: int = 128,
        kernel_size: int = 7,
        window_size: int = 4000,  # kernel size of the last convolution
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        """
        FCNN: Fully Connected Neural Network
            - Variable-length signals, same model for 8s, 10s, 60s ECGs
            - Sliding-window interpretation and GPU-friendly
        
        Input shape: (batch_size, 12, T), T = variable length (no padding) e.g., 3000, 4000, etc.
        Output shape: (batch, num_classes)

        ATTENTION
        Probleme: le GPU n’accepte pas l'input de modèle de taille (batch_size, 12, T) avec
        T differents, du coup il faut le régler dans le Dataset class
        """
        super().__init__()

        padding = 0

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels, ch1, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(ch2, ch3, kernel_size, padding=padding)

        # Batch Norm layers
        # nn.Identity() -> no operation
        self.bn1 = nn.BatchNorm1d(ch1) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(ch2) if use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm1d(ch3) if use_batchnorm else nn.Identity()

        # Pooling layers: Downsample to capture larger rhythm patterns
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2) # Downsample to capture larger rhythm patterns
        # Note: we don't necessarily need a 3rd max pool if the signal's length is not too 
        # long, but it helps increase the receptive field.

        # Drop out layer
        self.dropout = nn.Dropout(dropout)

        # Sliding-window classifier
        self.classifier = nn.Conv1d(
            in_channels=ch3,
            out_channels=num_classes,
            kernel_size=window_size,
            stride=1,
            padding=0
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Layer 1
        # leaky_relu: allows a small non-zero gradient for negative inputs, with alpha=0.01
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.pool1(x)

        # Layer 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = self.pool2(x)

        # Layer 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = self.dropout(x)

        # Sliding window predictions
        x = self.classifier(x)  # (batch_size, num_classes, T_windows)

        # For each channel, averages over the entire time/windows
        x = self.global_pool(x)
        x = x.squeeze(-1)

        return x