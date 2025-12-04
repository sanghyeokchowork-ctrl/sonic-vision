import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, base_model_path=None, embedding_dim=128):
        """
        Define the Siamese Network architecture:
        1. Frozen ResNet18 (Image feature extraction)
        2. Projection Head (Embedding vector generation)
        """
        super(SiameseNetwork, self).__init__()

        # 1. Load Pre-trained ResNet18
        # Using weights trained in Phase 1 (Genre Classification) instead of ImageNet weights often yields better performance.
        # If the path is not provided, default ImageNet weights are used.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Freeze the Backbone (Preserve existing knowledge)
        # Freeze all parameters of ResNet so they are not updated during training.
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove ResNet's final FC layer (Use only as a Feature Extractor)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 3. Trainable Head (Fine-tuning)
        # Compress from 512 dimensions (ResNet output) to 128 dimensions (Embedding space)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        # x: (Batch, 3, 224, 224)
        x = self.backbone(x)  # Output: (Batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (Batch, 512)
        x = self.fc(x)  # Projection: (Batch, 128)

        # L2 Normalize (Important when using Euclidean distance)
        # Normalize the vector length to 1, ensuring only direction (similarity) is compared
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2=None, input3=None):
        """
        Receives 3 inputs for Triplet Loss training, or processes 1 input for inference.
        """
        # Inference mode (when only 1 input is provided)
        if input2 is None:
            return self.forward_one(input1)

        # Training mode (Triplet: Anchor, Positive, Negative)
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3) if input3 is not None else None

        return output1, output2, output3