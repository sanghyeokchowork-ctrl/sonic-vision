import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

from vocal_timbre_model import VocalTimbreCNN

# ==========================================
# Configuration
# ==========================================
TAGS = ['Bright', 'Warm', 'Breathy', 'Rough', 'Clean']
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
# Automatically set device to MPS (Apple Silicon GPU) or CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TimbreDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        # Target length of audio samples (SAMPLE_RATE * DURATION)
        self.target_length = int(SAMPLE_RATE * DURATION)

        # Load CSV safely
        self.annotations = self._load_csv_safe(csv_file)

    def _load_csv_safe(self, csv_file):
        """
        Attempts to open the file by iterating through common encodings and separators.
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'latin1']
        separators = [',', '\t', ';']  # Try comma, tab, and semicolon

        for enc in encodings:
            for sep in separators:
                try:
                    # Use default (C) engine
                    df = pd.read_csv(csv_file, encoding=enc, sep=sep)

                    # Success criterion: Must have more than 1 column
                    if df.shape[1] > 1:
                        sep_name = 'Tab' if sep == '\t' else sep
                        print(f"✅ Loaded CSV successfully! (Encoding: {enc}, Separator: '{sep_name}')")

                        # Remove whitespace from column names
                        df.columns = df.columns.str.strip()
                        return df

                except Exception:
                    continue

        # If all attempts fail
        print("\n❌ CRITICAL ERROR: Could not read CSV file.")
        print("   The file content is completely corrupted or the format is incorrect.")
        print("   Please re-check if you exported it as 'Export to CSV' from Numbers.")
        raise RuntimeError("CSV Load Failed")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. File Path
        file_name = self.annotations.iloc[index, 0]
        audio_path = os.path.join(self.root_dir, file_name)

        # 2. Read Labels
        # Fill NaN values with 0
        raw_labels = self.annotations.iloc[index, 1:].fillna(0).values

        try:
            # Force conversion of strings/mixed types to numeric
            clean_labels = pd.to_numeric(raw_labels, errors='coerce')
            clean_labels = np.nan_to_num(clean_labels) # Convert NaN (from errors='coerce') to 0
            labels = torch.tensor(clean_labels, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ Label parsing error in {file_name}: {e}")
            labels = torch.zeros(len(TAGS), dtype=torch.float32)

        # 3. Load Audio and Extract Features
        try:
            # Load audio at SAMPLE_RATE, for DURATION seconds
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

            # Handle padding for files shorter than DURATION
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)))
            # Handle clipping for files longer than DURATION
            else:
                y = y[:self.target_length]

            # Extract MFCC features (n_mfcc=40)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            # Convert to PyTorch tensor and unsqueeze channel dimension (1, n_mfcc, time_steps)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

            return mfcc_tensor, labels

        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            # Return dummy tensor on failure (1, N_MFCC, ~130 time steps for 3s audio)
            return torch.zeros(1, N_MFCC, 130), labels


def create_dummy_csv(csv_path, audio_dir):
    """
    Creates a template CSV file if the label file is missing.
    """
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
        return False

    # Find all .wav files in the directory
    files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not files:
        print(f"⚠️ No .wav files found in {audio_dir}. Please add vocal tracks.")
        return False

    # Create DataFrame with filenames and tag columns initialized to 0
    df = pd.DataFrame(columns=['filename'] + TAGS)
    df['filename'] = files
    for tag in TAGS:
        df[tag] = 0

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📝 Created template CSV: {csv_path}")
    print("👉 Please open this CSV and mark tags (1 for yes, 0 for no) for each file.")
    return True


def train_timbre_model():
    """
    Main function to set up and run the training process.
    """
    # Define project structure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    audio_dir = os.path.join(project_root, 'data', 'vocals')
    csv_path = os.path.join(project_root, 'data', 'timbre_labels.csv')
    model_save_path = os.path.join(project_root, 'models', 'vocal_timbre.pth')

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Check for CSV and prompt for labeling if missing
    if not os.path.exists(csv_path):
        ready = create_dummy_csv(csv_path, audio_dir)
        if ready:
            print("🛑 Setup required: Label the CSV file and run this script again.")
        return

    print("🚀 Loading Dataset...")
    try:
        # Initialize the custom Dataset
        dataset = TimbreDataset(csv_file=csv_path, root_dir=audio_dir)
    except RuntimeError as e:
        print(e)
        return

    # Initialize the DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer
    model = VocalTimbreCNN(num_tags=len(TAGS)).to(DEVICE)
    # BCEWithLogitsLoss is suitable for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🔥 Start Training Timbre Model on {DEVICE}")
    model.train() # Set model to training mode

    # Training loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        # Wrap DataLoader with tqdm for progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for inputs, labels in pbar:
            # Move data to the specified device (GPU/CPU)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() # Zero the gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item()
            # Update progress bar with current batch loss
            pbar.set_postfix({'loss': loss.item()})

    # Save the model state dictionary after training
    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 Model Saved: {model_save_path}")


if __name__ == "__main__":
    train_timbre_model()