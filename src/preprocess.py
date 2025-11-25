import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

# Suppress warnings (e.g., MP3 header issues) to keep console clean
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
# Standard sample rate for music processing
SAMPLE_RATE = 22050
# Duration of each slice in seconds (Data Augmentation Strategy)
SLICE_DURATION = 3
# Number of samples per slice (22050 * 3 = 66150)
SAMPLES_PER_SLICE = SAMPLE_RATE * SLICE_DURATION


def create_spectrograms(source_path, target_path):
    """
    Converts .wav audio files into Mel-Spectrogram images (.png).
    Splits each 30s song into ten 3s slices to increase dataset size.
    """

    # 1. Get List of Genres
    genres = os.listdir(source_path)
    genres = [g for g in genres if os.path.isdir(os.path.join(source_path, g))]

    print(f"🚀 Start Preprocessing...")
    print(f"📂 Source: {source_path}")
    print(f"📂 Target: {target_path}")
    print(f"✂️ Slicing Strategy: {SLICE_DURATION}s segments (Data Augmentation)")

    # 2. Process each genre
    for genre in genres:
        print(f"\n🎵 Processing Genre: {genre.upper()}...")

        # Define input/output directories for this genre
        genre_dir = os.path.join(source_path, genre)
        output_dir = os.path.join(target_path, genre)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all wav files
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]

        # Use tqdm for progress bar
        for f in tqdm(files, desc=f"Converting {genre}", unit="file"):
            file_path = os.path.join(genre_dir, f)

            try:
                # 3. Load Audio
                # librosa loads audio as a floating point time series
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # 4. Slice Audio (Data Augmentation)
                # A generic GTZAN song is 30 seconds. We loop to create 3s slices.
                num_slices = int(len(y) / SAMPLES_PER_SLICE)

                for i in range(num_slices):
                    start_sample = i * SAMPLES_PER_SLICE
                    end_sample = start_sample + SAMPLES_PER_SLICE

                    slice_y = y[start_sample:end_sample]

                    # Skip if slice is too short (just in case)
                    if len(slice_y) != SAMPLES_PER_SLICE:
                        continue

                    # 5. Generate Mel-Spectrogram
                    # n_mels=128: Vertical resolution of the image (frequency bands)
                    mels = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=128)

                    # Convert to Decibels (Log Scale) because human hearing is logarithmic
                    log_mels = librosa.power_to_db(mels, ref=np.max)

                    # 6. Save as Image
                    # We create a figure without axes/frames to save ONLY the data visual
                    plt.figure(figsize=(2.56, 2.56), dpi=100)  # Result: 256x256 pixel image
                    plt.axis('off')  # Remove axis
                    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove padding

                    # Display the spectrogram
                    librosa.display.specshow(log_mels, sr=sr, hop_length=512)

                    # Save path: data/processed/blues/blues.00000_slice0.png
                    filename = f"{f[:-4]}_slice{i}.png"
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path, bbox_inches=None, pad_inches=0)
                    plt.close()  # Close figure to free memory

            except Exception as e:
                print(f"❌ Error processing {f}: {e}")


if __name__ == "__main__":
    # Define paths based on project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    raw_data_path = os.path.join(project_root, 'data', 'raw', 'gtzan', 'genres')
    processed_data_path = os.path.join(project_root, 'data', 'processed')

    create_spectrograms(raw_data_path, processed_data_path)
    print("\n🎉 All Processing Complete! Check 'data/processed' folder.")