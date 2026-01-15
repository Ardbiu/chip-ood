import torch
import torch.nn as nn
import torch.autograd as autograd
from .erm import ERM

class IRM(ERM):
    """
    Invariant Risk Minimization.
    """
    def __init__(self, encoder, predictor, penalty_weight=1e4):
        super().__init__(encoder, predictor)
        self.penalty_weight = penalty_weight

    def _irm_penalty(self, logits, y):
        # IRM v1 penalty
        scale = torch.tensor(1.).to(logits.device).requires_grad_()
        loss = torch.nn.functional.cross_entropy(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def update_with_extras(self, x, y, extras):
        if 'd' not in extras:
             return super().update(x, y)
             
        d = extras['d']
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        
        penalty_sum = 0
        loss_sum = 0
        domains_seen = 0
        
        # Compute per-environment penalty
        unique_d = d.unique()
        for env in unique_d:
            mask = (d == env)
            if mask.sum() == 0: continue
            
            # Recompute graph for each env to avoid tangled graphs?
            # Standard IRM: split batch
            z_env = z[mask]
            y_env = y[mask]
            
            logits_env = self.predictor(z_env)
            env_loss = torch.nn.functional.cross_entropy(logits_env, y_env)
            env_penalty = self._irm_penalty(logits_env, y_env)
            
            loss_sum += env_loss
            penalty_sum += env_penalty
            domains_seen += 1
            
        if domains_seen > 0:
            loss_sum /= domains_seen
            penalty_sum /= domains_seen
            
        total_loss = loss_sum + self.penalty_weight * penalty_sum
        
        return total_loss, {"loss": total_loss.item(), "nll": loss_sum.item(), "penalty": penalty_sum.item()}
