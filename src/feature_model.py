import torch
import torch.nn as nn
from torchvision import models


class FeatureRegressor(nn.Module):
    def __init__(self, num_features=4):
        super(FeatureRegressor, self).__init__()

        # 1. Pre-trained ResNet 로드
        # 이미지넷 가중치 사용
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 마지막 레이어 제거
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 512차원 벡터 출력

        # 2. Regression Head
        # 0~1 사이의 값을 예측해야 하므로 마지막에 Sigmoid 사용
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