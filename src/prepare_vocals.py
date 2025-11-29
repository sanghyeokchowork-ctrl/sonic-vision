import os
import shutil
from tqdm import tqdm
from separate import separate_audio


def prepare_vocal_dataset():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    source_dir = os.path.join(project_root, 'data', 'my_songs')  # 원본 노래 폴더
    target_dir = os.path.join(project_root, 'data', 'vocals')  # 보컬만 모을 폴더
    temp_dir = os.path.join(project_root, 'temp_stems')  # 임시 분리 폴더

    # 타겟 폴더 생성
    os.makedirs(target_dir, exist_ok=True)

    # 원본 노래 목록 가져오기
    if not os.path.exists(source_dir):
        print(f"❌ Error: '{source_dir}' directory not found.")
        return

    songs = [f for f in os.listdir(source_dir) if f.lower().endswith(('.mp3', '.wav'))]

    if not songs:
        print("⚠️ No songs found in 'data/my_songs'. Please put some music files there first.")
        return

    print(f"🚀 Found {len(songs)} songs. Extracting vocals for training...")

    for song in tqdm(songs, desc="Processing Tracks"):
        song_path = os.path.join(source_dir, song)

        # 1. 분리 수행 (Demucs)
        # separate_audio 함수는 분리된 파일들의 경로 딕셔너리를 반환합니다.
        stems = separate_audio(song_path, output_dir=temp_dir)

        # 2. 보컬 파일 이동
        if stems and 'vocals' in stems:
            vocal_src = stems['vocals']

            # 파일명 충돌 방지를 위해 노래 제목을 붙여서 저장
            # 예: "MySong.wav" -> "MySong_vocals.wav"
            safe_name = os.path.splitext(song)[0] + "_vocals.wav"
            vocal_dst = os.path.join(target_dir, safe_name)

            shutil.move(vocal_src, vocal_dst)
            # print(f"   ✅ Saved: {safe_name}")
        else:
            print(f"   ❌ Failed to extract vocals from {song}")

    # 임시 폴더 정리
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n🎉 Preparation Complete!")
    print(f"📂 Check '{target_dir}' folder.")
    print("👉 Now, run 'python src/vocal_timbre_train.py' again to generate the labeling CSV.")


if __name__ == "__main__":
    prepare_vocal_dataset()