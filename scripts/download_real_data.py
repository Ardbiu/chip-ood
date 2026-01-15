# scripts/download_domainbed_data.py
import os
import zipfile
import tarfile
import gdown
from torchvision.datasets.utils import download_url

# Standard DomainBed Data URLs
# Source: facebookresearch/DomainBed
DATASETS = {
    "PACS": {
        "url": "https://drive.google.com/uc?id=0B6eKvaijfFvjbU02M1B1bXk5VVk",
        "filename": "PACS.zip",
        "type": "zip",
        "gdrive": True
    },
    "VLCS": {
        "url": "http://www.cs.dartmouth.edu/~chenfang/Datasets/VLCS.tar.gz",
        "filename": "VLCS.tar.gz",
        "type": "tar",
        "gdrive": False
    },
    "OfficeHome": {
        "url": "https://drive.google.com/uc?id=0B81rNlvOA7hJMmxVd19tVldhaW8",
        # Alternative: http://hemanthdv.org/OfficeHome-Dataset/
        # But GDrive is often more stable for direct gdown
        "filename": "OfficeHome.zip",
        "type": "zip",
        "gdrive": True
    },
    "TerraIncognita": {
        "url": "https://lilablobssc.blob.core.windows.net/lilablobs/li-wild-data/locs.zip",
        "filename": "locs.zip", 
        "extract_name": "TerraIncognita",
        "type": "zip",
        "gdrive": False
    }
}

def download_and_extract(root, name, info):
    print(f"--- Processing {name} ---")
    target_dir = os.path.join(root, name)
    if os.path.exists(target_dir):
        print(f"Skipping {name}, directory exists at {target_dir}")
        return

    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, info["filename"])
    
    # Download
    if not os.path.exists(filepath):
        print(f"Downloading {name}...")
        if info.get("gdrive"):
            gdown.download(info["url"], filepath, quiet=False)
        else:
            download_url(info["url"], root, info["filename"])
    else:
        print(f"Archive found at {filepath}, skipping download.")
        
    # Extract
    print(f"Extracting {name}...")
    if info["type"] == "zip":
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(root)
    elif info["type"] == "tar":
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                # CVE-2007-4559 mitigation
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            safe_extract(tar_ref, root)
            
    # Post-process Rename (e.g. locs -> TerraIncognita)
    if info.get("extract_name"):
        extracted_path = os.path.join(root, "locs" if "locs" in info["filename"] else info["extract_name"])
        # TerraIncognita actually extracts as 'TerraIncognita' sometimes or 'locs'
        # Let's check what was verified.
        # User script checks for 'TerraIncognita/location_38'
        # If locs.zip extracts to 'locs', rename it.
        if "locs" in info["filename"] and os.path.exists(os.path.join(root, "locs")):
             os.rename(os.path.join(root, "locs"), target_dir)
             
    print(f"Successfully prepared {name}.")

if __name__ == "__main__":
    ROOT = "data/domainbed"
    print(f"Downloading real datasets to {ROOT}...")
    
    for name, info in DATASETS.items():
        try:
            download_and_extract(ROOT, name, info)
        except Exception as e:
            print(f"FAILED {name}: {e}")
            
    print("\nDone. Please run scripts/verify_domainbed_data.sh to confirm.")
