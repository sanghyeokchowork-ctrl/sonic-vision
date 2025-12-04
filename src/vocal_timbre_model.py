import torch
import torch.nn as nn

class VocalTimbreCNN(nn.Module):
    def __init__(self, num_tags=5):
        """
        CNN model for vocal timbre analysis.
        Input: (Batch, 1, n_mfcc, time_steps) -> Processes MFCC images.
        Output: Probability (0~1) of belonging to each tag (Bright, Warm, etc.).
        """
        super(VocalTimbreCNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output size example: (N, 32, 20, T/2) if n_mfcc=40
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output size example: (N, 64, 10, T/4)
        )

        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output size example: (N, 128, 5, T/8)
        )

        # Global Average Pooling (Summarizes features regardless of time axis length)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_tags)
            # Sigmoid is not used here because:
            # BCEWithLogitsLoss performs Sigmoid internally during training for stability
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x