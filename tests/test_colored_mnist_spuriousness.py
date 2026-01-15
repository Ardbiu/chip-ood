from chip_ood.data.colored_mnist import ColoredMNIST
from torch.utils.data import DataLoader
import numpy as np

def test_cmnist_generation():
    ds = ColoredMNIST(root="./tmp_test_data", env="train", correlation=0.9, seed=0)
    assert len(ds) > 0
    img, label = ds[0]
    assert img.shape == (3, 28, 28)

def test_colored_mnist_spuriousness():
    # Test strict binary_irm correlations
    # Train: Correlation ~0.9
    ds_train = ColoredMNIST(root="./tmp_test_data", env="train", correlation=0.9, seed=123, variant="binary_irm")
    
    match_count = 0
    total = len(ds_train)
    # Check first 1000 for speed
    check_n = min(total, 1000)
    
    for i in range(check_n):
        img, label = ds_train[i]
        label = int(label)
        # Check color (Red=0 if R>G)
        r_sum = img[0].sum()
        g_sum = img[1].sum()
        color = 0 if r_sum > g_sum else 1
        
        if color == label:
            match_count += 1
            
    corr = match_count / check_n
    print(f"Train Correlation: {corr}")
    assert 0.85 < corr < 0.95, f"Train correlation {corr} out of bounds"

    # Test: Correlation ~0.1
    ds_test = ColoredMNIST(root="./tmp_test_data", env="test", correlation=0.1, seed=123, variant="binary_irm")
    
    match_count = 0
    check_n = min(len(ds_test), 1000)
    
    for i in range(check_n):
        img, label = ds_test[i]
        label = int(label)
        r_sum = img[0].sum()
        g_sum = img[1].sum()
        color = 0 if r_sum > g_sum else 1
        
        if color == label:
            match_count += 1
            
    corr = match_count / check_n
    print(f"Test Correlation: {corr}")
    assert 0.05 < corr < 0.15, f"Test correlation {corr} out of bounds"
