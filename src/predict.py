import os
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
from torchvision import transforms
from PIL import Image

# [Fix] 맥북 Retina Display 문제 해결을 위한 백엔드 설정
import matplotlib

matplotlib.use('Agg')  # 화면에 창을 띄우지 않고 내부 연산만 수행
import matplotlib.pyplot as plt

# Import our model structure
from model import get_model

# ==========================================
# Configuration
# ==========================================
CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 22050
SLICE_DURATION = 3


def process_audio(file_path):
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None

    samples_per_slice = SAMPLE_RATE * SLICE_DURATION
    num_slices = int(len(y) / samples_per_slice)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensors = []
    print(f"🔪 Slicing audio into {num_slices} parts...")

    for i in range(num_slices):
        start = i * samples_per_slice
        end = start + samples_per_slice
        slice_y = y[start:end]

        if len(slice_y) != samples_per_slice:
            continue

        mels = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=128)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        # Create Plot
        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(log_mels, sr=sr, hop_length=512)

        # Convert to Image (Buffer)
        fig.canvas.draw()

        # [Fix] Robust buffer handling for any backend
        # Get the buffer size dynamically
        width, height = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)

        # Convert to PIL Image
        img = Image.fromarray(img_arr).convert('RGB')

        plt.close(fig)  # Explicitly close the figure to free memory
        tensors.append(data_transforms(img))

    if not tensors:
        return None
    return torch.stack(tensors).to(DEVICE)


def predict_genre(file_path, model_path):
    filename = os.path.basename(file_path)
    print("\n" + "=" * 50)
    print(f"🎵 Analyzing: {filename}")
    print("=" * 50)

    # Check Model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    model = get_model(num_classes=10, device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    inputs = process_audio(file_path)
    if inputs is None:
        print("❌ Error: Audio file is too short or invalid.")
        return

    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probabilities, dim=0)
        top_probs, top_indices = torch.topk(avg_probs, 3)

    print("📊 AI Analysis Result:")
    for i in range(3):
        genre = CLASSES[top_indices[i]]
        score = top_probs[i].item() * 100
        print(f"   Rank {i + 1}: {genre.upper()} ({score:.2f}%)")

    # Simple Insight
    top_genre = CLASSES[top_indices[0]]
    if top_genre in ['jazz', 'classical', 'pop']:
        print(f"💡 Insight: '{filename}' mapped to '{top_genre.upper()}' (Smooth/Harmonic Focus).")
    elif top_genre in ['hiphop', 'disco', 'metal', 'rock', 'reggae']:
        print(f"💡 Insight: '{filename}' mapped to '{top_genre.upper()}' (Rhythmic/Beat Focus).")
    elif top_genre in ['blues', 'country']:
        print(f"💡 Insight: '{filename}' mapped to '{top_genre.upper()}' (Vocal/Acoustic Focus).")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    my_song_dir = os.path.join(project_root, 'data', 'my_songs')

    # Find Songs
    if not os.path.exists(my_song_dir):
        print(f"❌ Directory not found: {my_song_dir}")
        exit()

    songs = [f for f in os.listdir(my_song_dir) if f.lower().endswith(('.mp3', '.wav'))]

    if not songs:
        print(f"⚠️ No songs found in: {my_song_dir}")
    else:
        print(f"🚀 Found {len(songs)} songs. Processing...")
        for song in songs:
            song_path = os.path.join(my_song_dir, song)
            try:
                predict_genre(song_path, model_path)
            except Exception as e:
                print(f"❌ Failed to analyze {song}: {e}")