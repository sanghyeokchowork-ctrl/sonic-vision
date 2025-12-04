import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time

from siamese_model import SiameseNetwork

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
MARGIN = 1.0  # Margin value for Triplet Loss (how much distance to enforce)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TripletGTZANDataset(Dataset):
    """
    Generates Anchor, Positive, and Negative pairs in real-time from the GTZAN dataset.
    - Anchor: A random image
    - Positive: Another image of the same genre as the Anchor
    - Negative: An image from a different genre than the Anchor
    """