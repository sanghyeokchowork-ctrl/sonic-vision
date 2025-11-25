import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import our custom modules
from dataset import get_data_loaders
from model import get_model

# ==========================================
# Configuration
# ==========================================
NUM_EPOCHS = 10  # 전체 데이터를 몇 번 볼 것인가
LEARNING_RATE = 0.001  # 학습 속도
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU 사용


def train_model():
    # 1. Setup Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'processed')
    model_save_path = os.path.join(project_root, 'models', 'best_model.pth')

    # Create models directory if not exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"🚀 Training Started on Device: {DEVICE}")

    # 2. Get Data & Model
    train_loader, val_loader, class_names = get_data_loaders(data_path)
    model = get_model(num_classes=len(class_names), device=DEVICE)

    # 3. Define Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track best accuracy
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 4. Training Loop
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # tqdm creates a progress bar
            pbar = tqdm(dataloader, desc=f"{phase.upper()} Phase", leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar description with current loss
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f"   🎉 New Best Model Saved! (Acc: {best_acc:.4f})")

    time_elapsed = time.time() - start_time
    print('=' * 50)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print('=' * 50)


if __name__ == "__main__":
    train_model()