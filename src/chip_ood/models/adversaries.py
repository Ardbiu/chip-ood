import torch
import torch.nn as nn
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

class Predictor(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class AdversaryClf(nn.Module):
    """
    Predicts label y from spurious z_s.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, z_s, alpha=1.0):
        # Gradient reversal happens inside method usually, or here?
        # Let's put it here for convenience: "forward with reversal"
        z_rev = grad_reverse(z_s, alpha)
        return self.net(z_rev)

class MINEAdversary(nn.Module):
    """
    Statistics network for MINE. T(z, y).
    """
    def __init__(self, z_dim, num_classes, hidden_dim=256):
        super().__init__()
        # We can embed y or concat one-hot. Concat one-hot is simpler.
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z, y_indices):
        # y_indices: (B,) class indices
        y_onehot = torch.nn.functional.one_hot(y_indices, self.num_classes).float()
        inp = torch.cat([z, y_onehot], dim=1)
        return self.net(inp)
