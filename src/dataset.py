import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 32
IMG_SIZE = 224


def get_data_loaders(data_dir):
    """
    Creates Training and Validation DataLoaders.

    Structure:
    - Applies transformations (Resize, Tensor conversion, Normalization)
    - Splits data into Train (80%) and Validation (20%)
    """

    # 1. Define Transforms
    # ResNet18 expects normalized images in a specific way
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset using ImageFolder
    # It automatically detects classes from folder names (blues, jazz, etc.)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # 3. Split into Train/Val (80:20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"📊 Dataset Loaded from: {data_dir}")
    print(f"   - Total Images: {total_size}")
    print(f"   - Training Set: {train_size}")
    print(f"   - Validation Set: {val_size}")
    print(f"   - Classes: {full_dataset.classes}")

    # 4. Create DataLoaders
    # num_workers=2: Use multi-processing for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.classes