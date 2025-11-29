import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, base_model_path=None, embedding_dim=128):
        """
        샴 네트워크 구조 정의:
        1. Frozen ResNet18 (이미지 특징 추출)
        2. Projection Head (임베딩 벡터 생성)
        """
        super(SiameseNetwork, self).__init__()

        # 1. Load Pre-trained ResNet18
        # 이미지넷 가중치 대신, 우리가 1단계에서 학습시킨(Genre Classification) 가중치를 쓰면 성능이 더 좋습니다.
        # 경로가 없으면 기본 ImageNet 가중치를 사용합니다.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Freeze the Backbone (기존 지식 보존)
        # ResNet의 모든 파라미터를 고정하여 학습되지 않도록 합니다.
        for param in resnet.parameters():
            param.requires_grad = False

        # ResNet의 마지막 FC 레이어 제거 (Feature Extractor로만 사용)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 3. Trainable Head (Fine-tuning)
        # 512차원(ResNet 출력) -> 128차원(임베딩 공간)으로 압축
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 과적합 방지
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        # x: (Batch, 3, 224, 224)
        x = self.backbone(x)  # Output: (Batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (Batch, 512)
        x = self.fc(x)  # Projection: (Batch, 128)

        # L2 Normalize (유클리드 거리를 사용할 때 중요)
        # 벡터의 길이를 1로 맞춰서 방향(유사도)만 비교하도록 함
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2=None, input3=None):
        """
        Triplet Loss 학습을 위해 3개의 입력을 받거나,
        추론(Inference)을 위해 1개의 입력을 처리합니다.
        """
        # 추론 모드 (입력이 1개일 때)
        if input2 is None:
            return self.forward_one(input1)

        # 학습 모드 (Triplet: Anchor, Positive, Negative)
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3) if input3 is not None else None

        return output1, output2, output3