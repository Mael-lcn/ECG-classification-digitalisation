import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 12,
        ch1: int = 32,
        ch2: int = 64,
        ch3: int = 128,
        kernel_size: int = 3, 
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        """
        Simple 3-layer 1D CNN for ECG classification

        Args:
            num_classes: Number of output classes (default 27)
            in_channels: Number of ECG leads (default 12)
            ch1, ch2, ch3: Number of channels in conv layers
            kernel_size: Conv kernel size (for now same for all layers cuz 400Hz, might tune later)
            dropout: Dropout probability
            use_batchnorm: Boolean flag to activate or desactivate the usage of batch norm.
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

        # Global Average Pooling: Redundant for fixed length but excellent for 
        # reducing parameters and increasing robustness.
        self.global_pool = nn.AdaptiveAvgPool1d(1) 

        # Fully connected output (just ch3 because global pooling reduces length to 1)
        self.fc = nn.Linear(ch3, num_classes)

    def forward(self, x):
        """
        x: (batch_size, n_leads, n_samples)
        """
        # Layer 1
        # leaky_relu: allows a small non-zero gradient for negative inputs, with alpha=0.01
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01) # (batch_size, ch1, n_samples)
        x = self.pool1(x) # (batch_size, ch1, n_samples / 2)

        # Layer 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01) # (batch_size, ch2, n_samples / 2)
        x = self.pool2(x) # (batch_size, ch2, n_samples / 4)

        # Layer 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01) # (batch_size, ch3, n_samples / 4)

        # Dropout before global pooling: Forces the network to find redundant features across leads
        x = self.dropout(x) # (batch_size, ch3, n_samples / 4)

        # Global average pooling over the time dimension
        x = self.global_pool(x) # (batch_size, ch3, 1)
        x = x.squeeze(-1) # (batch_size, ch3)

        x = self.fc(x) # (batch, num_classes)
        return x
    
