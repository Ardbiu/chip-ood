import argparse
import os
import json
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegacpython -m chip_ood.evaluation.aggregate --input_dir results --output_dir results/aggregate_dir):
    records = []
    all_learning_curves = []
    
    # Walk directory
    for root, dirs, files in os.walk(input_dir):
        if "final_metrics.json" in files:
            # Found a run
            try:
                # Load final metrics
                with open(os.path.join(root, "final_metrics.json"), "r") as f:
                    final_metrics = json.load(f)
                
                # Load config
                cfg_path = os.path.join(root, ".hydra", "config.yaml") 
                if not os.path.exists(cfg_path):
                     cfg_path = os.path.join(root, "config.yaml") # Fallback
                     
                if os.path.exists(cfg_path):
                    cfg = OmegaConf.load(cfg_path)
                    method = cfg.method.name
                    dataset = cfg.data.name
                    # Handle list or string for test_envs
                    if "test_envs" in cfg.data:
                         test_env = cfg.data.test_envs[0] if isinstance(cfg.data.test_envs, (list, tuple)) else cfg.data.test_envs
                    else:
                        test_env = "all"
                        
                    seed = cfg.seed
                    p_train = cfg.data.get("train_correlation", "N/A")
                    p_test = cfg.data.get("test_correlation", "N/A")
                    
                    records.append({
                        "dataset": dataset,
                        "method": method,
                        "test_env": test_env,
                        "seed": seed,
                        "train_acc": final_metrics.get("final_train_acc", np.nan), # Assuming we add this to final_metrics
                        "test_acc": final_metrics.get("final_test_acc", np.nan),
                        "best_test_acc": final_metrics.get("best_test_acc", np.nan),
                        "p_train": p_train,
                        "p_test": p_test
                    })
                    
                # Load learning curves (metrics.jsonl)
                metrics_path = os.path.join(root, "metrics.jsonl")
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        for line in f:
                            entry = json.loads(line)
                            entry["method"] = method
                            entry["seed"] = seed
                            entry["dataset"] = dataset
                            all_learning_curves.append(entry)
                            
            except Exception as e:
                print(f"Error processing {root}: {e}")
                
    if not records:
        print("No results found.")
        return
        
    df = pd.DataFrame(records)
    
    # Save Main CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "main_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved main table to {csv_path}")
    
    # Summary JSON
    summary = df.groupby(["dataset", "method"]).agg(
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"),
        count=("test_acc", "count")
    ).reset_index()
    
    summary_path = os.path.join(output_dir, "summary.json")
    summary.to_json(summary_path, orient="records", indent=2)
    print(f"Saved summary to {summary_path}")
    
    # --- Plotting ---
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    # 1. Test Accuracy Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="dataset", y="test_acc", hue="method", errorbar="sd", capsize=.1)
    plt.title("Test Accuracy by Method")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_bar.png"))
    plt.close()
    
    # 2. Accuracy Over Epochs
    if all_learning_curves:
        lc_df = pd.DataFrame(all_learning_curves)
        if "epoch" in lc_df.columns and "test_acc" in lc_df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=lc_df, x="epoch", y="test_acc", hue="method", style="dataset", dashes=False, marker="o")
            plt.title("Test Accuracy Over Epochs")
            plt.ylabel("Test Accuracy")
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "accuracy_over_epochs.png"))
            plt.close()
            
    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    aggregate_results(args.input_dir, args.output_dir)
