# DomainBed Data Setup Guide

To run true ICML-grade experiments, you must download the real datasets (PACS, VLCS, TerraIncognita, OfficeHome) and place them in the correct directory structure.

## Directory Structure
All data should be located under `chip_ood/data/domainbed/`.

```
chip_ood/
  data/
    domainbed/
      PACS/
        art_painting/
        cartoon/
        photo/
        sketch/
      VLCS/
        ...
      TerraIncognita/
        ...
      OfficeHome/
        ...
      DomainNet/
        ...
```

## Download Instructions

### 1. PACS (Physics, Art, Cartoon, Sketch)
- **Source**: [Standard DomainBed PACS](https://domaingeneralization.github.io/#data)
- **Format**: Images in `domain/class/image.jpg` folders.
- **Command**:
  ```bash
  cd data/domainbed
  # Example: Use the official link or a mirror
  wget http://.../PACS.zip
  unzip PACS.zip
  ```

### 2. VLCS (VOC, LabelMe, Caltech, Sun)
- **Source**: [Standard DomainBed VLCS](https://domaingeneralization.github.io/#data)
- **Command**:
  ```bash
  cd data/domainbed
  # Download and extract
  ```

### 3. TerraIncognita
- **Critial Note**: This dataset is tricky due to missing images in some versions. Use the "TerraIncognita100" subset if possible, or the DomainBed standard version.
- **Source**: [Standard DomainBed TerraIncognita](https://domaingeneralization.github.io/#data)

### 4. OfficeHome
- **Source**: [Standard DomainBed OfficeHome](https://domaingeneralization.github.io/#data)

## Verification
Run the verification script to confirm your setup:
```bash
bash scripts/verify_domainbed_data.sh
```

## Config Reference
The configs in `src/chip_ood/configs/data/domainbed_*.yaml` expect `root: ./data/domainbed`.
