import torch
import torch.nn as nn

class VocalTimbreCNN(nn.Module):
    def __init__(self, num_tags=5):
        """
        보컬 음색(Timbre) 분석을 위한 CNN 모델
        Input: (Batch, 1, n_mfcc, time_steps) -> MFCC 이미지를 처리
        Output: 각 태그(Bright, Warm 등)에 속할 확률 (0~1)
        """
        super(VocalTimbreCNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # (N, 32, 20, T/2)
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # (N, 64, 10, T/4)
        )

        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # (N, 128, 5, T/8)
        )

        # Global Average Pooling (시간 축에 상관없이 특징 요약)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_tags)
            # 마지막에 Sigmoid를 쓰지 않는 이유:
            # 학습 시 BCEWithLogitsLoss가 내부적으로 Sigmoid를 수행하여 더 안정적임
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x