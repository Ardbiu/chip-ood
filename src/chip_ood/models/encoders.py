import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MNISTEncoder(nn.Module):
    """
    Simple ConvNet for 28x28 images.
    Splits output into z_c and z_s.
    """
    def __init__(self, z_dim_c=32, z_dim_s=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), # 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 4 (after padding logic? 7/2=3.5 -> 3 or 4)
            nn.ReLU(),
            nn.Flatten()
        )
        # 128 * 4 * 4 = 2048
        self.fc_c = nn.Linear(128 * 4 * 4, z_dim_c)
        self.fc_s = nn.Linear(128 * 4 * 4, z_dim_s)
        
    def forward(self, x):
        h = self.conv(x)
        return self.fc_c(h), self.fc_s(h)

class ResNetEncoder(nn.Module):
    """
    ResNet18 backbone.
    """
    def __init__(self, z_dim_c=256, z_dim_s=256, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        
        self.fc_c = nn.Linear(512, z_dim_c)
        self.fc_s = nn.Linear(512, z_dim_s)
        
    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc_c(h), self.fc_s(h)
