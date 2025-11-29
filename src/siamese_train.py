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
MARGIN = 1.0  # Triplet Loss의 마진 값 (거리를 얼마나 벌릴 것인가)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TripletGTZANDataset(Dataset):
    """
    GTZAN 데이터셋에서 Anchor, Positive, Negative 쌍을 실시간으로 생성합니다.
    - Anchor: 랜덤 이미지
    - Positive: Anchor와 같은 장르의 다른 이미지
    - Negative: Anchor와 다른 장르의 이미지
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 장르별로 이미지 파일 경로 정리
        # { 'blues': ['path/to/blues1.png', ...], 'jazz': [...] }
        self.data = {}
        self.genres = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        print("📂 Indexing Triplet Dataset...")
        for genre in self.genres:
            genre_dir = os.path.join(root_dir, genre)
            files = [os.path.join(genre_dir, f) for f in os.listdir(genre_dir) if f.endswith('.png')]
            if len(files) > 0:
                self.data[genre] = files

        # 전체 이미지 리스트 (인덱싱용)
        self.all_images = []
        for genre in self.genres:
            for img_path in self.data[genre]:
                self.all_images.append((img_path, genre))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # 1. Anchor 선택
        anchor_path, anchor_genre = self.all_images[index]

        # 2. Positive 선택 (같은 장르, 다른 파일)
        # 리스트에서 자기 자신을 제외하고 선택하면 좋지만,
        # 간단하게 랜덤 선택 후 같으면 다시 뽑는 방식 사용
        pos_path = anchor_path
        while pos_path == anchor_path:
            pos_path = random.choice(self.data[anchor_genre])

        # 3. Negative 선택 (다른 장르)
        neg_genre = anchor_genre
        while neg_genre == anchor_genre:
            neg_genre = random.choice(self.genres)
        neg_path = random.choice(self.data[neg_genre])

        # 4. 이미지 로드 및 변환
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img


def train_siamese():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'processed')
    save_path = os.path.join(project_root, 'models', 'siamese_net.pth')

    # 1. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TripletGTZANDataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Mac에서는 0 권장

    # 2. Model setup
    model = SiameseNetwork().to(DEVICE)

    # 3. Loss & Optimizer
    # TripletMarginLoss: max(d(a, p) - d(a, n) + margin, 0)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    # Optimizer: Head 부분만 학습 (model.fc)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Start Siamese Training on {DEVICE}")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            optimizer.zero_grad()

            # Forward (3개의 임베딩 추출)
            # forward 함수가 (a, p, n)을 받아 3개를 리턴하도록 수정했으므로 호출
            embed_a, embed_p, embed_n = model(anchor, positive, negative)

            # Loss 계산
            loss = criterion(embed_a, embed_p, embed_n)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(dataloader)
        print(f"📉 Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        # 매 에포크마다 저장
        torch.save(model.state_dict(), save_path)

    print(f"🎉 Training Complete! Model saved to {save_path}")


if __name__ == "__main__":
    train_siamese()