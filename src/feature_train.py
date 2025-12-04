import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from feature_model import FeatureRegressor

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class FeatureDataset(Dataset):
    def __init__(self, csv_file, img_root):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Find image path (located inside the genre folder)
        img_name = row['filename']
        genre = row['genre']
        img_path = os.path.join(self.img_root, genre, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Target values (Energy, Danceability, Acousticness, Valence)
        targets = torch.tensor([
            row['energy'],
            row['danceability'],
            row['acousticness'],
            row['valence']
        ], dtype=torch.float32)

        return image, targets


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    csv_path = os.path.join(project_root, 'data', 'feature_labels.csv')
    img_root = os.path.join(project_root, 'data', 'processed')
    save_path = os.path.join(project_root, 'models', 'feature_model.pth')

    # 1. Check Data
    if not os.path.exists(csv_path):
        print("❌ CSV not found. Run 'create_feature_dataset.py' first.")
        return

    dataset = FeatureDataset(csv_path, img_root)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model
    model = FeatureRegressor(num_features=4).to(DEVICE)
    criterion = nn.MSELoss()  # Use MSE since this is a regression problem
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 Training Feature Regressor on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    torch.save(model.state_dict(), save_path)
    print(f"🎉 Model saved: {save_path}")


if __name__ == "__main__":
    train()