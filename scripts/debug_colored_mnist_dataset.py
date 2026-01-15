import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from chip_ood.data.colored_mnist import ColoredMNIST
from torchvision.utils import make_grid
from collections import defaultdict

def analyze_dataset(root, env, correlation, seed, output_dir):
    print(f"Analyzing {env} set (p={correlation})...")
    ds = ColoredMNIST(root=root, env=env, correlation=correlation, seed=seed)
    
    stats = {
        "total": len(ds),
        "class_counts": defaultdict(int),
        "color_counts": defaultdict(int),
        "match_counts": 0,
        "mismatch_counts": 0,
        "by_class": defaultdict(lambda: {"match": 0, "mismatch": 0})
    }
    
    images_to_show = []
    
    # Iterate
    # dataset[i] returns (img_tensor, label_tensor)
    # We need to deduce color.
    # Color logic: Red=0 (channel 0 high), Green=1 (channel 1 high)
    
    for i in range(len(ds)):
        img, label = ds[i]
        label = int(label)
        
        # Check color
        # img is (3, 28, 28)
        # Sum channels
        r_sum = img[0].sum()
        g_sum = img[1].sum()
        
        if r_sum > g_sum:
            color = 0 # Red
        else:
            color = 1 # Green
            
        stats["class_counts"][label] += 1
        stats["color_counts"][color] += 1
        
        if color == label:
            stats["match_counts"] += 1
            stats["by_class"][label]["match"] += 1
        else:
            stats["mismatch_counts"] += 1
            stats["by_class"][label]["mismatch"] += 1
            
        if i < 32:
            images_to_show.append(img)

    # Analyze
    stats["empirical_correlation"] = stats["match_counts"] / stats["total"]
    print(f"  Empirical Correlation: {stats['empirical_correlation']:.4f}")
    
    # Convert defaultdicts to dicts for JSON
    stats["class_counts"] = dict(stats["class_counts"])
    stats["color_counts"] = dict(stats["color_counts"])
    stats["by_class"] = dict(stats["by_class"])
    
    # Grid
    grid = make_grid(images_to_show, nrow=8)
    # Save grid
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(ndarr)
    plt.axis('off')
    plt.title(f"{env} (p={correlation})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"grid_{env}.png"))
    plt.close()
    
    return stats

def main():
    root = "./data/mnist"
    output_dir = "results/debug"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze Train
    train_stats = analyze_dataset(root, "train", 0.9, 0, output_dir)
    
    # Analyze Test
    test_stats = analyze_dataset(root, "test", 0.1, 0, output_dir)
    
    report = {
        "train": train_stats,
        "test": test_stats
    }
    
    with open(os.path.join(output_dir, "colored_mnist_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print("Report saved.")

if __name__ == "__main__":
    main()
