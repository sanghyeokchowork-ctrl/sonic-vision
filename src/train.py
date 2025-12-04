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
NUM_EPOCHS = 10  # How many times to see the entire dataset
LEARNING_RATE = 0.001  # Learning rate
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Use Mac GPU (MPS) if available


def train_model():
    # 1. Setup Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # Define path to processed data
    data_path = os.path.join(project_root, 'data', 'processed')
    # Define path to save the best model weights
    model_save_path = os.path.join(project_root, 'models', 'best_model.pth')

    # Create models directory if not exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"🚀 Training Started on Device: {DEVICE}")

    # 2. Get Data & Model
    # Load training and validation data loaders, and class names
    train_loader, val_loader, class_names = get_data_loaders(data_path)
    # Instantiate the model and move it to the device
    model = get_model(num_classes=len(class_names), device=DEVICE)

    # 3. Define Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track best accuracy
    best_acc = 0.0
    # Deep copy the initial model weights for saving the best state
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
                # Move inputs and labels to the designated device
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history/compute gradients only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the predicted class index
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics (accumulate loss and correct predictions)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar description with current batch loss
                pbar.set_postfix({'loss': loss.item()})

            # Calculate average loss and accuracy for