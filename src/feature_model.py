import torch
import torch.nn as nn
from torchvision import models


class FeatureRegressor(nn.Module):
    def __init__(self, num_features=4):
        super(FeatureRegressor, self).__init__()

        # 1. Load Pre-trained ResNet
        # Use ImageNet weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Outputs a 512-dimensional vector

        # 2. Regression Head
        # Use Sigmoid at the end since we must predict values between 0 and 1
        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_features),
            nn.Sigmoid()  # Output: 0.0 ~ 1.0
        )

    def forward(self, x):
        x = self.backbone(x)  # (Batch, 512)
        return self.head(x)  # (Batch, 4)