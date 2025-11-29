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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TimbreDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        self.target_length = int(SAMPLE_RATE * DURATION)

        # 안전하게 CSV 로드
        self.annotations = self._load_csv_safe(csv_file)

    def _load_csv_safe(self, csv_file):
        """
        인코딩과 구분자(Separator)를 직접 순회하며 파일 열기를 시도합니다.
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'latin1']
        separators = [',', '\t', ';']  # 콤마, 탭, 세미콜론 순으로 시도

        for enc in encodings:
            for sep in separators:
                try:
                    # engine='python' 없이 기본(C) 엔진 사용
                    df = pd.read_csv(csv_file, encoding=enc, sep=sep)

                    # 성공 기준: 컬럼이 2개 이상이어야 함
                    if df.shape[1] > 1:
                        # [수정] f-string 문법 오류 수정 (백슬래시 제거)
                        sep_name = 'Tab' if sep == '\t' else sep
                        print(f"✅ Loaded CSV successfully! (Encoding: {enc}, Separator: '{sep_name}')")

                        # 컬럼 이름 공백 제거
                        df.columns = df.columns.str.strip()
                        return df

                except Exception:
                    continue

        # 모든 시도 실패 시
        print("\n❌ CRITICAL ERROR: Could not read CSV file.")
        print("   파일 내용이 완전히 깨졌거나 형식이 맞지 않습니다.")
        print("   Numbers에서 'CSV로 내보내기'를 했는지 다시 확인해주세요.")
        raise RuntimeError("CSV Load Failed")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. 파일 경로
        file_name = self.annotations.iloc[index, 0]
        audio_path = os.path.join(self.root_dir, file_name)

        # 2. 라벨 읽기
        raw_labels = self.annotations.iloc[index, 1:].fillna(0).values

        try:
            # 문자열 등 강제 형변환
            clean_labels = pd.to_numeric(raw_labels, errors='coerce')
            clean_labels = np.nan_to_num(clean_labels)
            labels = torch.tensor(clean_labels, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ Label parsing error in {file_name}: {e}")
            labels = torch.zeros(len(TAGS), dtype=torch.float32)

        # 3. 오디오 로드
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)))
            else:
                y = y[:self.target_length]

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

            return mfcc_tensor, labels

        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return torch.zeros(1, N_MFCC, 130), labels


def create_dummy_csv(csv_path, audio_dir):
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
        return False

    files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not files:
        print(f"⚠️ No .wav files found in {audio_dir}. Please add vocal tracks.")
        return False

    df = pd.DataFrame(columns=['filename'] + TAGS)
    df['filename'] = files
    for tag in TAGS:
        df[tag] = 0

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📝 Created template CSV: {csv_path}")
    print("👉 Please open this CSV and mark tags (1 for yes, 0 for no) for each file.")
    return True


def train_timbre_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    audio_dir = os.path.join(project_root, 'data', 'vocals')
    csv_path = os.path.join(project_root, 'data', 'timbre_labels.csv')
    model_save_path = os.path.join(project_root, 'models', 'vocal_timbre.pth')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if not os.path.exists(csv_path):
        ready = create_dummy_csv(csv_path, audio_dir)
        if ready:
            print("🛑 Setup required: Label the CSV file and run this script again.")
        return

    print("🚀 Loading Dataset...")
    try:
        dataset = TimbreDataset(csv_file=csv_path, root_dir=audio_dir)
    except RuntimeError as e:
        print(e)
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VocalTimbreCNN(num_tags=len(TAGS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🔥 Start Training Timbre Model on {DEVICE}")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 Model Saved: {model_save_path}")


if __name__ == "__main__":
    train_timbre_model()