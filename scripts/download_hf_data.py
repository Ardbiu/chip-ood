# scripts/download_hf_data.py
import os
import shutil
from huggingface_hub import snapshot_download

# HuggingFace Dataset Mirrors for DomainBed
# These are community or official mirrors enabling cluster downloads.

HF_DATASETS = {
    "PACS": {
        "repo_id": "flwrlabs/pacs",
        "repo_type": "dataset",
        "subfolder": None # dataset usually has 'data' or is flat
    },
    "OfficeHome": {
        "repo_id": "flwrlabs/office-home",
        "repo_type": "dataset",
        "subfolder": None
    },
    "TerraIncognita": {
        "repo_id": "BGLab/TerraIncognita",
        "repo_type": "dataset",
        "subfolder": None
    }
}

# VLCS is rarer on HF in raw form. We might skip or rely on flwrlabs/vlcs if it exists.
# We will focus on PACS, OfficeHome, Terra first.

def download_hf(root, name, info):
    print(f"\n--- Processing {name} (via HuggingFace) ---")
    target_dir = os.path.join(root, name)
    if os.path.exists(target_dir):
        print(f"Skipping {name}, directory exists at {target_dir}")
        return

    print(f"Downloading {info['repo_id']}...")
    try:
        # Download to cache
        cache_dir = snapshot_download(repo_id=info["repo_id"], repo_type=info["repo_type"])
        
        # Move/Copy to target
        # HF datasets structure varies.
        # flwrlabs/pacs typically checks out as:
        #   cache_dir/
        #       - art_painting/
        #       - ...
        # Or sometimes inside a 'data' folder.
        
        # Check structure
        src_dir = cache_dir
        
        # Simple copy
        print(f"Copying from {src_dir} to {target_dir}...")
        shutil.copytree(src_dir, target_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git*', '*.md'))
        
        print(f"Successfully configured {name}.")
        
    except Exception as e:
        print(f"FAILED {name}: {e}")
        # Clean up partial
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

if __name__ == "__main__":
    ROOT = "data/domainbed"
    print(f"Downloading datasets to {ROOT} using HuggingFace Hub...")
    # Requires: pip install huggingface_hub
    
    for name, info in HF_DATASETS.items():
        download_hf(ROOT, name, info)
        
    print("\nNote: VLCS might not be available via this HF script. Manual download required if needed.")
    print("Please run scripts/verify_domainbed_data.sh to confirm.")
