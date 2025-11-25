import torch.nn as nn
from torchvision import models


def get_model(num_classes=10, device='cpu'):
    """
    Loads a pre-trained ResNet18 model and modifies the final layer
    to match the number of music genres (10).
    """

    print("🏗️ Loading Pre-trained ResNet18 Model...")

    # 1. Load Pre-trained ResNet18
    # weights='DEFAULT' loads the best available weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Modify the Final Layer (Fully Connected Layer)
    # Original ResNet detects 1000 classes. We change it to 10 (num_classes).
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 3. Move model to the computation device (CPU or GPU)
    model = model.to(device)

    return model