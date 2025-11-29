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
    """학습된 샴 네트워크 로드"""
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
    오디오(.wav) -> 멜 스펙트로그램 이미지 -> 텐서 변환
    (Siamese Network 입력용)
    """
    try:
        # 1. Load Audio (3초만 사용 - 대표 구간)
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)

        # 길이가 짧으면 패딩
        target_len = 22050 * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 2. Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mels = librosa.power_to_db(mels, ref=np.max)

        # 3. Save to Buffer (Matplotlib 없이 픽셀값 변환)
        # 속도를 위해 plt 대신 min-max 정규화로 직접 이미지 생성
        min_val = log_mels.min()
        max_val = log_mels.max()
        img_arr = (log_mels - min_val) / (max_val - min_val) * 255
        img_arr = img_arr.astype(np.uint8)

        # PIL Image로 변환 (Resize를 위해)
        img = Image.fromarray(img_arr).convert('RGB')  # 3채널 복사

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
    데이터셋 폴더(GTZAN 등)를 스캔하여 모든 곡의 임베딩 벡터를 미리 계산합니다.
    Returns: { 'filename': vector (numpy array) }
    """
    print("📂 Building Similarity Index (This may take a while)...")

    vectors = {}

    # data/processed 폴더가 있다면 이미지를 바로 씀 (빠름)
    # 없다면 data/raw 오디오를 변환 (느림)

    # 여기서는 'data/processed' (이미지)가 있다고 가정합니다 (preprocess.py 실행 후)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 장르 폴더 순회
    genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for genre in tqdm(genres, desc="Indexing Genres"):
        genre_dir = os.path.join(data_dir, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith('.png')]

        # 너무 많으면 장르당 20개만 샘플링 (속도 최적화 데모용)
        # 실제 서비스에선 다 해야 함
        files = files[:20]

        for f in files:
            img_path = os.path.join(genre_dir, f)
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Siamese Network의 forward_one 사용
                    emb = model.forward_one(input_tensor)
                    vectors[f"{genre}/{f}"] = emb.cpu().numpy().flatten()
            except:
                continue

    return vectors


def find_similar_songs(target_audio_path, model, db_vectors, top_k=5):
    """
    입력된 오디오와 가장 유사한 곡 K개를 DB에서 찾습니다.
    """
    # 1. 타겟 오디오 임베딩 추출
    target_tensor = audio_to_tensor(target_audio_path)
    if target_tensor is None:
        return []

    with torch.no_grad():
        target_vec = model.forward_one(target_tensor).cpu().numpy().flatten()

    # 2. 코사인 유사도 계산
    db_keys = list(db_vectors.keys())
    db_vals = np.array(list(db_vectors.values()))

    target_vec = target_vec.reshape(1, -1)

    # (1, 128) vs (N, 128)
    sim_scores = cosine_similarity(target_vec, db_vals)[0]

    # 3. Top K 추출
    top_indices = sim_scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = sim_scores[idx]
        name = db_keys[idx]
        # 파일명 정리 (blues/blues.00000_slice0.png -> blues.00000)
        clean_name = name.split('/')[-1].split('_slice')[0]
        results.append((clean_name, score))

    return results