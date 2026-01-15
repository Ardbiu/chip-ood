import os
import torch
from torchvision.utils import save_image

def create_mock_pacs(root):
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    
    for domain in domains:
        for cls in classes:
            dir_path = os.path.join(root, "PACS", domain, cls)
            os.makedirs(dir_path, exist_ok=True)
            # Create 1 mock image
            img = torch.randn(3, 224, 224)
            save_image(img, os.path.join(dir_path, "mock_0.jpg"))
            print(f"Created {dir_path}/mock_0.jpg")

def create_mock_terraincognita(root):
    domains = ["location_38", "location_43", "location_46", "location_100"]
    # TerraIncognita has 10 classes
    classes = [str(i) for i in range(10)]
    
    for domain in domains:
        for cls in classes:
            dir_path = os.path.join(root, "TerraIncognita", domain, cls)
            os.makedirs(dir_path, exist_ok=True)
            # Create 1 mock image
            img = torch.randn(3, 224, 224)
            save_image(img, os.path.join(dir_path, "mock_0.jpg"))
            print(f"Created {dir_path}/mock_0.jpg")

if __name__ == "__main__":
    # Ensure we use chip_ood data path
    root = "./data/domainbed"
    os.makedirs(root, exist_ok=True)
    create_mock_pacs(root)
    create_mock_terraincognita(root)
    print("Mock DomainBed Created.")
