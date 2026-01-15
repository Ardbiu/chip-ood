import os
from datasets import load_dataset

# Quick inspection to see what's inside these datasets

DATASETS = {
    "OfficeHome": "flwrlabs/office-home",
    "TerraIncognita": "BGLab/TerraIncognita",
    "VLCS": "flwrlabs/vlcs" # Guessing this exists
}

def inspect(name, repo_id):
    print(f"\n--- Inspecting {name} ({repo_id}) ---")
    try:
        ds = load_dataset(repo_id, split="train", trust_remote_code=True)
        print(f"Size: {len(ds)}")
        print(f"Features: {ds.features}")
        
        # Check domain column
        if "domain" in ds.features:
            # If ClassLabel
            if hasattr(ds.features["domain"], "names"):
                print(f"Domain Names: {ds.features['domain'].names}")
            else:
                # Iterate a bit to find unique
                unique = set(ds[:100]["domain"])
                print(f"First 100 unique domains: {unique}")
        else:
            print("No 'domain' column found!")
            
    except Exception as e:
        print(f"Failed to load: {e}")

if __name__ == "__main__":
    for k, v in DATASETS.items():
        inspect(k, v)
