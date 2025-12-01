import os
import pandas as pd
import librosa
from tqdm import tqdm
from feature_utils import extract_advanced_features

# 설정
SAMPLE_RATE = 22050
DURATION = 3  # 3초 단위로 잘라서 학습 (데이터 증강)


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
                # 전체 곡 로드
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)

                # 3초씩 슬라이딩하며 데이터 생성 (한 곡당 10개의 데이터)
                # 이렇게 해야 "순간적인" 에너지나 리듬을 학습할 수 있음
                samples_per_slice = SAMPLE_RATE * DURATION
                num_slices = int(len(y) / samples_per_slice)

                for i in range(num_slices):
                    start = i * samples_per_slice
                    end = start + samples_per_slice
                    y_slice = y[start:end]

                    if len(y_slice) < samples_per_slice: continue

                    # 고급 특징 추출
                    feats = extract_advanced_features(y_slice, sr)

                    # 파일명: genre.00000_slice0.png (이미지 파일명과 매칭되게 저장)
                    # 실제로는 wav를 분석했지만, 학습은 spectrogram 이미지로 할 것이므로 이름 규칙 통일
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