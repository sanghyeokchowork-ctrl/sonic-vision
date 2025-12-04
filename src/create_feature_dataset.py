import os
import pandas as pd
import librosa
from tqdm import tqdm
from feature_utils import extract_advanced_features

# Configuration
SAMPLE_RATE = 22050
DURATION = 3  # Slice duration for training (data augmentation)

def create_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'gtzan', 'genres')
    output_csv = os.path.join(project_root, 'data', 'feature_labels.csv')

    if not os.path.exists(raw_data_path):
        print("❌ GTZAN dataset not found.")
        return

    data = []
    genres = [d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]

    print("🚀 Generating High-Quality Feature Labels using DSP...")

    for genre in genres:
        genre_dir = os.path.join(raw_data_path, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]

        for f in tqdm(files, desc=f"Processing {genre}"):
            file_path = os.path.join(genre_dir, f)
            try:
                # Load the full song
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)

                # Generate data by sliding 3-second windows (10 samples per song)
                # This approach allows the model to learn "instantaneous" energy and rhythm.
                samples_per_slice = SAMPLE_RATE * DURATION
                num_slices = int(len(y) / samples_per_slice)

                for i in range(num_slices):
                    start = i * samples_per_slice
                    end = start + samples_per_slice
                    y_slice = y[start:end]

                    if len(y_slice) < samples_per_slice: continue

                    # Extract advanced features
                    feats = extract_advanced_features(y_slice, sr)

                    # Filename format: genre.00000_slice0.png (Match the image filename convention)
                    # Although we analyzed the WAV file, we use this naming convention because the model
                    # will ultimately be trained using spectrogram images.
                    img_name = f"{f[:-4]}_slice{i}.png"

                    row = {"filename": img_name, "genre": genre}
                    row.update(feats)
                    data.append(row)

            except Exception as e:
                print(f"Error {f}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset created: {output_csv} ({len(df)} samples)")


if __name__ == "__main__":
    create_dataset()