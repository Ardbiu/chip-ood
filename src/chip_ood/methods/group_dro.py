import torch
import torch.nn as nn
import torch.nn.functional as F
from .erm import ERM

class GroupDRO(ERM):
    """
    Group Distributionally Robust Optimization.
    Minimizes the worst-case Group loss.
    """
    def __init__(self, encoder, predictor, num_groups, eta=1e-2):
        super().__init__(encoder, predictor)
        self.eta = eta
        self.num_groups = num_groups
        # Register q (group weights) as buffer
        self.register_buffer("q", torch.ones(num_groups))

    def update(self, x, y, step=None):
        # Requires group/domain annotation 'd'
        # Fallback to ERM if not available
        return super().update(x, y, step)

    def update_with_extras(self, x, y, extras):
        # x: [B, ...]
        # y: [B]
        # d: [B] (in extras)
        
        if 'd' not in extras:
            return super().update(x, y)
            
        d = extras['d']
        device = x.device
        
        z = self.encoder(x)
        # Check if encoder returns tuple (chip)
        if isinstance(z, tuple): z = z[0] # Use Zc
        
        logits = self.predictor(z)
        loss_per_sample = F.cross_entropy(logits, y, reduction='none')
        
        # Group aggregation
        group_losses = []
        group_counts = []
        
        for g in range(self.num_groups):
            mask = (d == g)
            if mask.sum() > 0:
                group_losses.append(loss_per_sample[mask].mean())
                group_counts.append(mask.sum())
            else:
                group_losses.append(torch.tensor(0., device=device))
                group_counts.append(torch.tensor(0, device=device))
                
        group_losses = torch.stack(group_losses)
        
        # DRO Update
        # Exponentiated gradient ascent on q
        with torch.no_grad():
            self.q = self.q * torch.exp(self.eta * group_losses)
            self.q = self.q / self.q.sum()
            
        # Robust Loss
        loss = (self.q * group_losses).sum()
        
        return loss, {"loss": loss.item(), "max_group_loss": group_losses.max().item()}
