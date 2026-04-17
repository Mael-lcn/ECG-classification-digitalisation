import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Image(nn.Module):
    def __init__(self, num_classes=27, input_h=512, input_w=512, mode='square'):
        super(CNN_Image, self).__init__()
        
        self.mode = mode
        self.input_h = input_h
        
        if mode == 'square':
            kernel = (3, 3)
            stride = (1, 1)
            padding = (1, 1)
        elif mode == 'rectangle':
            kernel = (input_h // 2, 3) 
            stride = (2, 1)
            padding = (0, 1)
        else:
            raise ValueError("Mode must be 'square' or 'rectangle'")

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, 
                               kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, batch_mask=None):
        """ x.shape: (Batch, 12, H, W) """

        if x.dtype == torch.uint8:
            x = x.to(next(self.parameters()).dtype) / 255.0
        
        # Square or Rectangle
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        return logits # (Batch, num_classes)


if __name__ == "__main__":
    # Test Square Mode
    model_sq = CNN_Image(mode='square')
    dummy_input = torch.randn(2, 12, 512, 512)
    out_sq = model_sq(dummy_input)
    print(f"Square output shape: {out_sq.shape}") # [2, 27]

    # Test Rectangle Mode
    model_rect = CNN_Image(mode='rectangle')
    out_rect = model_rect(dummy_input)
    print(f"Rectangle output shape: {out_rect.shape}") # [2, 27]