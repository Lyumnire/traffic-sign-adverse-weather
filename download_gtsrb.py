import os
import torchvision
from torchvision.datasets import GTSRB

def download_and_verify():
    data_dir = "./datasets/gtsrb"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading GTSRB train set...")
    train_set = GTSRB(root=data_dir, split='train', download=True)
    
    print("Downloading GTSRB test set...")
    test_set = GTSRB(root=data_dir, split='test', download=True)
    
    print(f"GTSRB Train size: {len(train_set)}")
    print(f"GTSRB Test size: {len(test_set)}")

if __name__ == "__main__":
    download_and_verify()
