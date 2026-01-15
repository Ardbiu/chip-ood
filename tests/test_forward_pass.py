import torch
from chip_ood.models import MNISTEncoder, MNISTDecoder, Predictor, AdversaryClf
from chip_ood.methods import CHIP

def test_chip_forward():
    # Mock data
    x = torch.randn(4, 3, 28, 28)
    y = torch.randint(0, 2, (4,))
    
    # Models
    enc = MNISTEncoder(z_dim_c=8, z_dim_s=8)
    dec = MNISTDecoder(z_dim=16)
    pred = Predictor(input_dim=8, num_classes=2)
    adv = AdversaryClf(input_dim=8, num_classes=2)
    
    method = CHIP(enc, pred, dec, adv, num_classes=2)
    
    loss, metrics = method.update(x, y, step=0)
    
    assert "loss" in metrics
    assert "loss_pred" in metrics
    assert metrics["loss"] > 0
