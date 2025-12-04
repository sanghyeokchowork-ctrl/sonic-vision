import os
import torch
import numpy as np
import librosa
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Siamese Model Import
from siamese_model import SiameseNetwork

# ==========================================
# Configuration
# ==========================================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG_SIZE = 224


def load_siamese_model(model_path):
    """Load the trained Siamese Network"""
    print(f"🏗️ Loading Siamese Network from {os.path.basename(model_path)}...")
    model = SiameseNetwork().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("⚠️ Warning: Model file not found. Recommendations will be random.")
    model.eval()
    return model


def audio_to_tensor(audio_path):
    """
    Audio (.wav) -> Mel Spectrogram Image -> Tensor Conversion
    (For Siamese Network input)
    """
    try:
        # 1. Load Audio (Use only 3 seconds - representative segment)
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)

        # Pad if the length is too short
        target_len = 22050 * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 2. Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        # 3. Save to Buffer (Convert to pixel values without Matplotlib)
        # Directly generate the image using min-max normalization for speed (instead of plt)
        min_val = log_mels.min()
        max_val = log_mels.max()
        img_arr = (log_mels - min_val) / (max_val - min_val) * 255
        img_arr = img_arr.astype(np.uint8)

        # Convert to PIL Image (for Resize)
        img = Image.fromarray(img_arr).convert('RGB')  # Copy to 3 channels

        # 4. Transform
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(img).unsqueeze(0).to(DEVICE)

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


def build_database_index(model, data_dir):
    """
    Scans the dataset folder (e.g., GTZAN) and pre-calculates the embedding vector for all songs.
    Returns: { 'filename': vector (numpy array) }
    """
    print("📂 Building Similarity Index (This may take a while)...")

    vectors = {}

    # If the 'data/processed' folder exists, use the images directly (faster)
    # If not, convert the 'data/raw' audio (slower)

    # Here we assume 'data/processed' (images) exists (after running preprocess.py)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Iterate through genre folders
    genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for genre in tqdm(genres, desc="Indexing Genres"):
        genre_dir = os.path.join(data_dir, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith('.png')]

        # Sample only 20 files per genre if there are too many (for speed optimization demo)
        # In a real service, all should be processed
        files = files[:20]

        for f in files:
            img_path = os.path.join(genre_dir, f)
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Use the forward_one method of the Siamese Network
                    emb = model.forward_one(input_tensor)
                    vectors[f"{genre}/{f}"] = emb.cpu().numpy().flatten()
            except:
                continue

    return vectors


def find_similar_songs(target_audio_path, model, db_vectors, top_k=5):
    """
    Finds the K most similar songs to the input audio in the DB.
    """
    # 1. Extract target audio embedding
    target_tensor = audio_to_tensor(target_audio_path)
    if target_tensor is None:
        return []

    with torch.no_grad():
        target_vec = model.forward_one(target_tensor).cpu().numpy().flatten()

    # 2. Calculate Cosine Similarity
    db_keys = list(db_vectors.keys())
    db_vals = np.array(list(db_vectors.values()))

    target_vec = target_vec.reshape(1, -1)

    # (1, 128) vs (N, 128)
    sim_scores = cosine_similarity(target_vec, db_vals)[0]

    # 3. Extract Top K
    top_indices = sim_scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = sim_scores[idx]
        name = db_keys[idx]
        # Clean up filename (e.g., blues/blues.00000_slice0.png -> blues.00000)
        clean_name = name.split('/')[-1].split('_slice')[0]
        results.append((clean_name, score))

    return results