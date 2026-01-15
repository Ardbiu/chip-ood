import torch
import torch.nn as nn

class MNISTDecoder(nn.Module):
    """
    Decodes z_c + z_s -> Image (28x28).
    """
    def __init__(self, z_dim=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # 32? Need to check output sizes for MNIST 28x28
            nn.Sigmoid() 
        )
        # Note: 32x32 is close to 28x28. We might need crop or resize.
        # Let's adjust padding or use interpolation.
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 4, 4)
        out = self.deconv(h)
        # Crop to 28x28
        out = out[:, :, 2:30, 2:30]
        # Wait, 32 -> 28 means crop 4 pixels. 2 on each side.
        return out

class ResNetDecoder(nn.Module):
    """
    Lightweight decoder for 224x224 images.
    Using pixel shuffle or simple unsampling for speed.
    """
    def __init__(self, z_dim=512):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), # 224
            nn.Sigmoid()
        )
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 7, 7)
        return self.net(h)
