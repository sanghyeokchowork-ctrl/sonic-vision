import os
import torch
import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import get_model

# ==========================================
# Configuration
# ==========================================
CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def save_gradcam(file_path, model_path, output_path):
    filename = os.path.basename(file_path)
    print(f"\n🔍 Processing: {filename}...")

    # 1. Load Model
    model = get_model(num_classes=10, device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Select Target Layer
    target_layers = [model.layer4[-1]]

    # 3. Process Audio (Find Peak)
    try:
        y, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print(f"❌ Read Error: {e}")
        return

    # Find peak volume
    rms = librosa.feature.rms(y=y)[0]
    peak_frame = np.argmax(rms)
    peak_time = librosa.frames_to_time(peak_frame, sr=sr)

    # 3 seconds slice
    start_sample = max(0, int((peak_time - 1.5) * sr))
    end_sample = start_sample + (3 * 22050)
    if end_sample > len(y):
        start_sample = 0
        end_sample = 3 * 22050
    y_slice = y[start_sample:end_sample]

    # Generate Spectrogram
    mels = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128)
    log_mels = librosa.power_to_db(mels, ref=np.max)

    # Plot to Buffer
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

    # 4. Input Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    # 5. Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    outputs = model(input_tensor)
    _, preds = torch.max(outputs, 1)
    predicted_class = preds.item()

    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 6. Save
    rgb_img = np.array(img_pil.resize((224, 224)))
    rgb_img = np.float32(rgb_img) / 255
    from pytorch_grad_cam.utils.image import show_cam_on_image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {os.path.basename(output_path)} (Pred: {CLASSES[predicted_class].upper()})")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'best_model.pth')

    my_song_dir = os.path.join(project_root, 'data', 'my_songs')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Find files
    songs = [f for f in os.listdir(my_song_dir) if f.lower().endswith(('.mp3', '.wav'))]

    print("=" * 50)
    print(f"📂 Found {len(songs)} songs in folder:")
    for s in songs:
        print(f"   - {s}")
    print("=" * 50)

    if not songs:
        print(f"⚠️ No songs found! Check path: {my_song_dir}")
    else:
        for song in songs:
            song_path = os.path.join(my_song_dir, song)
            # Use distinct filename for each song
            output_path = os.path.join(results_dir, f"heatmap_{song}.jpg")

            try:
                save_gradcam(song_path, model_path, output_path)
            except Exception as e:
                print(f"❌ Fatal Error on {song}: {e}")

    print("\n🎉 Process Finished. Check 'results' folder.")