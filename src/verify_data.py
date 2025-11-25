import os
import glob


def verify_dataset_structure():
    """
    Verifies if the GTZAN dataset is correctly placed in 'data/raw/gtzan/genres'.
    Counts the number of audio files per genre.
    """
    # 1. Define Path
    # Project Root -> data -> raw -> gtzan -> genres
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(current_dir), 'data', 'raw', 'gtzan', 'genres')

    print("=" * 50)
    print("  Verifying Dataset Structure...")
    print(f"  Target Path: {dataset_path}")
    print("=" * 50)

    # 2. Check if directory exists
    if not os.path.exists(dataset_path):
        print(f" Error: Directory not found!")
        print(f"   Expected: .../sonic-vision/data/raw/gtzan/genres")
        print("   Please check your folder structure again.")
        return

    # 3. Check Genres and File Counts
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    total_files = 0
    all_good = True

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)

        # Check if genre folder exists
        if not os.path.isdir(genre_path):
            print(f" Missing Folder: {genre}")
            all_good = False
            continue

        # Count .wav files
        wav_files = glob.glob(os.path.join(genre_path, "*.wav"))
        count = len(wav_files)
        total_files += count

        if count == 100:
            print(f" {genre:<10}: {count} files (OK)")
        else:
            print(f" {genre:<10}: {count} files (Expected 100)")
            all_good = False

    print("-" * 50)

    # 4. Final Result
    if all_good and total_files == 1000:
        print(f" SUCCESS! Total {total_files} audio files found.")
        print(" You are ready for Phase 3 (Preprocessing).")
    else:
        print(" Something is wrong.")
        print("   Please make sure you unzipped 'genres_original' inside 'data/raw/gtzan/'.")


if __name__ == "__main__":
    verify_dataset_structure()