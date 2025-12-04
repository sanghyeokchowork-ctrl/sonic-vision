import os
import shutil
from tqdm import tqdm
from separate import separate_audio


def prepare_vocal_dataset():
    # Path settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    source_dir = os.path.join(project_root, 'data', 'my_songs')  # Original songs folder
    target_dir = os.path.join(project_root, 'data', 'vocals')  # Folder to collect only vocals
    temp_dir = os.path.join(project_root, 'temp_stems')  # Temporary separation folder

    # Create target folder
    os.makedirs(target_dir, exist_ok=True)

    # Get the list of original songs
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

        # 1. Perform separation (Demucs)
        # The separate_audio function returns a dictionary of separated file paths.
        stems = separate_audio(song_path, output_dir=temp_dir)

        # 2. Move the vocal file
        if stems and 'vocals' in stems:
            vocal_src = stems['vocals']

            # Save with the song title appended to prevent file name collisions
            # Example: "MySong.wav" -> "MySong_vocals.wav"
            safe_name = os.path.splitext(song)[0] + "_vocals.wav"
            vocal_dst = os.path.join(target_dir, safe_name)

            shutil.move(vocal_src, vocal_dst)
            # print(f"   ✅ Saved: {safe_name}")
        else:
            print(f"   ❌ Failed to extract vocals from {song}")

    # Clean up temporary folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n🎉 Preparation Complete!")
    print(f"📂 Check '{target_dir}' folder.")
    print("👉 Now, run 'python src/vocal_timbre_train.py' again to generate the labeling CSV.")


if __name__ == "__main__":
    prepare_vocal_dataset()