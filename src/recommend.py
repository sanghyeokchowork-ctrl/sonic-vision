import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt  # For spectrogram generation

# Import our model structure
from model import get_model

# ==========================================
# Configuration
# ==========================================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64
IMG_SIZE = 224


def get_feature_extractor(model_path):
    """
    Loads the trained model and removes the final classification layer.
    Output: 512-dimensional feature vector instead of class probabilities.
    """
    # 1. Load trained model
    model = get_model(num_classes=10, device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 2. Replace the last layer (fc) with Identity
    # This makes the model return the 512 features directly
    model.fc = nn.Identity()
    model.eval()
    return model


def extract_dataset_features(model, data_dir):
    """
    Extracts features for ALL songs in the GTZAN dataset.
    Returns: A dictionary { 'genre/song_name': vector }
    """
    print("running feature extraction database...")

    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset (Using the processed images)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    features_dict = {}

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Indexing Database"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)  # Output shape: (Batch, 512)

            # Move to CPU and numpy
            current_features = outputs.cpu().numpy()

            # Since we process in batches, we need to map features back to filenames
            # Note: This is a simplified mapping. In a real app, we'd map indices strictly.
            # Here, we just aggregate features for demo purposes.
            pass

            # [Optimization]
    # Since mapping individual slices back to song names in batch mode is complex,
    # and we want to average vectors per song (Song-Level Embedding),
    # let's iterate by folders manually for clarity.

    song_vectors = {}

    genres = os.listdir(data_dir)
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path): continue

        # Group slices by song name (e.g., blues.00000)
        song_groups = {}
        files = os.listdir(genre_path)
        for f in files:
            if not f.endswith('.png'): continue
            song_name = f.split('_slice')[0]  # blues.00000
            if song_name not in song_groups:
                song_groups[song_name] = []
            song_groups[song_name].append(os.path.join(genre_path, f))

        # Extract features for each song (Average of slices)
        for song_name, img_paths in tqdm(song_groups.items(), desc=f"Indexing {genre}", leave=False):
            song_feats = []

            # Process slices in small batches or one by one
            slice_tensors = []
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                slice_tensors.append(transform(img))

            if not slice_tensors: continue

            batch_input = torch.stack(slice_tensors).to(DEVICE)
            with torch.no_grad():
                batch_out = model(batch_input)
                # Average all slices to get ONE vector for the song
                avg_feat = torch.mean(batch_out, dim=0).cpu().numpy()

            song_vectors[f"{genre}/{song_name}"] = avg_feat

    return song_vectors


def process_target_song(model, file_path):
    """
    Converts a raw .mp3/.wav into a feature vector using the model.
    """
    # 1. Audio to Spectrogram Slices (Same logic as predict.py)
    import matplotlib
    matplotlib.use('Agg')

    y, sr = librosa.load(file_path, sr=22050)
    samples_per_slice = 22050 * 3
    num_slices = int(len(y) / samples_per_slice)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    slice_tensors = []

    for i in range(num_slices):
        start = i * samples_per_slice
        end = start + samples_per_slice
        slice_y = y[start:end]
        if len(slice_y) != samples_per_slice: continue

        mels = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=128)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(log_mels, sr=sr, hop_length=512)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        img_pil = Image.fromarray(img_arr).convert('RGB')
        plt.close(fig)

        slice_tensors.append(transform(img_pil))

    if not slice_tensors: return None

    # 2. Extract Feature
    batch_input = torch.stack(slice_tensors).to(DEVICE)
    with torch.no_grad():
        batch_out = model(batch_input)
        avg_feat = torch.mean(batch_out, dim=0).cpu().numpy()

    return avg_feat


def recommend_songs(target_vec, database_vectors, top_k=5):
    """
    Calculates Cosine Similarity and returns top K matches.
    """
    db_keys = list(database_vectors.keys())
    db_vals = np.array(list(database_vectors.values()))

    # Reshape target for sklearn (1, 512)
    target_vec = target_vec.reshape(1, -1)

    # Calculate Similarity: Result (1, 1000)
    sim_scores = cosine_similarity(target_vec, db_vals)[0]

    # Get Top K indices
    top_indices = sim_scores.argsort()[-top_k:][::-1]

    recommendations = []
    for idx in top_indices:
        score = sim_scores[idx]
        song_name = db_keys[idx]
        recommendations.append((song_name, score))

    return recommendations


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    data_dir = os.path.join(project_root, 'data', 'processed')
    my_song_dir = os.path.join(project_root, 'data', 'my_songs')

    # 1. Setup Feature Extractor
    print("🚀 Initializing Recommendation Engine...")
    model = get_feature_extractor(model_path)

    # 2. Index Database (GTZAN)
    # In a real app, we would save this to a file (e.g., features.npy) to avoid re-running.
    print("📂 Indexing GTZAN Database (This may take a moment)...")
    db_vectors = extract_dataset_features(model, data_dir)
    print(f"✅ Indexed {len(db_vectors)} songs from database.")

    # 3. Process My Songs
    my_songs = [f for f in os.listdir(my_song_dir) if f.lower().endswith(('.mp3', '.wav'))]

    for song in my_songs:
        song_path = os.path.join(my_song_dir, song)
        print(f"\n🎵 Analyzing Target: {song}")

        target_vec = process_target_song(model, song_path)
        if target_vec is None: continue

        # 4. Get Recommendations
        recs = recommend_songs(target_vec, db_vectors)

        print(f"   ❤️  Top 5 Similar Songs:")
        for name, score in recs:
            print(f"      - [{score * 100:.1f}%] {name}")