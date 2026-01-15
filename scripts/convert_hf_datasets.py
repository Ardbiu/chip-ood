import os
import shutil
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

# Mappings for DomainBed
# We need to map HF Dataset structure -> data/domainbed/{Dataset}/{Domain}/{Class}/img.jpg

DATASETS_CONFIG = {
    "PACS": {
        "hf_path": "flwrlabs/pacs",
        "domain_key": "domain", # Integer or string column?
        "label_key": "label",
        "image_key": "image"
    },
    "OfficeHome": {
        "hf_path": "flwrlabs/office-home",
        "domain_key": "domain",
        "label_key": "label",
        "image_key": "image"
    },
    "TerraIncognita": {
        "hf_path": "BGLab/TerraIncognita",
        "domain_key": "domain", # Check if this exists
        "label_key": "label",
        "image_key": "image"
    }
}

def save_item(args):
    img, filepath = args
    if not os.path.exists(filepath):
        img.save(filepath)

def process_dataset(name, cfg, root_dir):
    print(f"\nProcessing {name}...")
    try:
        # Load dataset
        # Trust remote code needed for some?
        ds = load_dataset(cfg["hf_path"], split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return

    # Check features to decode integers to labels
    features = ds.features
    
    # helper to get names
    def get_name(key, idx):
        if hasattr(features[key], "int2str"):
            return features[key].int2str(idx)
        return str(idx)

    tasks = []
    
    print(f"Converting {len(ds)} images...")
    
    output_root = os.path.join(root_dir, name)
    
    for idx, item in tqdm(enumerate(ds), total=len(ds)):
        try:
            domain_idx = item[cfg["domain_key"]]
            label_idx = item[cfg["label_key"]]
            img = item[cfg["image_key"]]
            
            domain_name = get_name(cfg["domain_key"], domain_idx)
            class_name = get_name(cfg["label_key"], label_idx)
            
            # Destination: root/Dataset/Domain/Class/Image.jpg
            dest_dir = os.path.join(output_root, domain_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            file_path = os.path.join(dest_dir, f"{idx}.jpg")
            
            # Save (Directly or buffer)
            # Avoid too many threads for IO
            if not os.path.exists(file_path):
                 img.convert("RGB").save(file_path)
                 
        except Exception as e:
            # print(f"Skipping item {idx}: {e}")
            pass

    print(f"Finished {name}")

if __name__ == "__main__":
    ROOT = "data/domainbed"
    
    # 1. PACS
    process_dataset("PACS", DATASETS_CONFIG["PACS"], ROOT)
    
    # 2. OfficeHome
    process_dataset("OfficeHome", DATASETS_CONFIG["OfficeHome"], ROOT)
    
    # 3. TerraIncognita
    # Note: BGLab/TerraIncognita might have different split names or column names
    # Inspecting schema at runtime or assuming standard
    process_dataset("TerraIncognita", DATASETS_CONFIG["TerraIncognita"], ROOT)
