import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseMethod(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def update(self, x, y, step):
        """
        Returns loss, metrics dictionary.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(BaseMethod):
    def __init__(self, encoder, predictor, num_classes):
        super().__init__(num_classes)
        self.encoder = encoder
        self.predictor = predictor
        
    def forward_features(self, x):
        # ERM uses only causal part if split, or just flat output?
        # If we reuse the CIRL encoder (which splits), we need to decide.
        # Usually ERM just uses the whole representation or we designate one part.
        # To make it comparable to CIRL which uses Zc, we can force ERM to use same architecture
        # but ignoring the split constraint (just using Zc to predict).
        # OR we use a standard ResNet without split.
        # For simplicity/fairness, let's assume 'encoder' returns one vector or we concat.
        
        # Duck typing: Check if encoder returns tuple
        out = self.encoder(x)
        if isinstance(out, tuple):
            zc, zs = out
            return zc # Use Zc as the representation for prediction
        return out

    def update(self, x, y, step):
        feats = self.forward_features(x)
        logits = self.predictor(feats)
        loss = F.cross_entropy(logits, y)
        
        return loss, {"loss_pred": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()}

    def predict(self, x):
        feats = self.forward_features(x)
        return self.predictor(feats)
