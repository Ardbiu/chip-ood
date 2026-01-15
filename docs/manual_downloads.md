# Manual Download Instructions

Some datasets (VLCS, TerraIncognita) are difficult to download automatically due to broken links or authentication barriers on the cluster.

If you need these datasets, please download them on your local machine and `scp` them to the cluster.

## 1. VLCS
- **Link**: [Google Drive (TALLY Mirror)](https://drive.google.com/u/0/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8&export=download)
- **File**: `VLCS.tar.gz`
- **Action**:
  ```bash
  # Local
  scp VLCS.tar.gz user@cluster:~/chip-ood/data/domainbed/
  
  # Cluster
  cd data/domainbed
  tar -xzvf VLCS.tar.gz
  ```

## 2. TerraIncognita
- **Link**: [Original Blob](https://lilablobssc.blob.core.windows.net/lilablobs/li-wild-data/locs.zip)
- **File**: `locs.zip`
- **Action**:
  ```bash
  # Local
  scp locs.zip user@cluster:~/chip-ood/data/domainbed/
  
  # Cluster
  cd data/domainbed
  unzip locs.zip
  mv locs TerraIncognita  # Rename folder
  ```

## 3. PACS & OfficeHome
These are successfully handled by `scripts/convert_hf_datasets.py` via HuggingFace.
