import torch
import torch.nn as nn
import torch.nn.functional as F

class CHIP(nn.Module):
    def __init__(self, encoder, predictor, decoder, adversary, 
                 lambda_rec=1.0, lambda_adv=1.0, adv_type="clf", num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.adversary = adversary
        
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.adv_type = adv_type
        self.num_classes = num_classes
        
    def update(self, x, y, step):
        # 1. Encode
        zc, zs = self.encoder(x)
        
        # 2. Predict (Task Loss)
        pred_logits = self.predictor(zc)
        loss_pred = F.cross_entropy(pred_logits, y)
        
        # 3. Reconstruct (Reconstruction Loss)
        # We need to concat zc and zs? Decoder expects z?
        # Our decoder impl takes single z tensor.
        z_combined = torch.cat([zc, zs], dim=1)
        x_hat = self.decoder(z_combined)
        
        # MSE for images
        loss_rec = F.mse_loss(x_hat, x)
        
        # 4. Adversary (Independence Loss)
        loss_adv = torch.tensor(0.0, device=x.device)
        adv_acc = 0.0
        
        if self.adv_type == "clf":
            # Gradient Reversal included in adversary.forward typically, OR we simply subtract?
            # Creating min-max: 
            # Main obj: minimize L_pred + L_rec - L_adv_clf (maximize entropy of A)
            # Adversary obj: minimize L_adv_clf
            # With GRL in the forward pass of AdveraryClf:
            # The gradient flowing back from AdversaryClf is inverted.
            # So if we simply MINIMIZE CrossEntropy(Adv(zs), y), the gradient update will:
            # - Move A's weights to minimize CE (improve classification)
            # - Move Encoder's weights to MAXIMIZE CE (confuse A) via GRL.
            
            adv_logits = self.adversary(zs) # Contains GRL
            loss_adv = F.cross_entropy(adv_logits, y)
            
            adv_acc = (adv_logits.argmax(1) == y).float().mean().item()
            
        elif self.adv_type == "mine":
            # Just a placeholder for the skeleton, implementing full MINE loop is complex for minimal.
            pass
            
        # Total Loss
        # For GRL: We want to min (L_pred + L_rec + lambda * L_adv_ce)
        # Because L_adv_ce is effectively 'maximized' by encoder due to GRL.
        loss = loss_pred + self.lambda_rec * loss_rec + self.lambda_adv * loss_adv
        
        metrics = {
            "loss": loss.item(),
            "loss_pred": loss_pred.item(),
            "loss_rec": loss_rec.item(),
            "loss_adv": loss_adv.item(),
            "acc": (pred_logits.argmax(1) == y).float().mean().item(),
            "adv_acc": adv_acc
        }
        
        return loss, metrics

    def predict(self, x):
        zc, zs = self.encoder(x)
        return self.predictor(zc)
