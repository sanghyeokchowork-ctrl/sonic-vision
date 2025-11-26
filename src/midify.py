import os
import sys

# [CRITICAL] Disable GPU for TensorFlow on Mac (Metal) and CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Force CPU using TensorFlow Config (맥북 Metal 가속 끄기)
try:
    import tensorflow as tf

    # Hide GPUs from TensorFlow
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow is forced to use CPU only.")
except Exception as e:
    print(f"⚠️ Could not configure TensorFlow: {e}")

from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH


def audio_to_midi(audio_path, output_dir):
    """
    Converts Audio to MIDI using Spotify's Basic Pitch.
    Executed as a standalone script to avoid memory locks.
    """
    try:
        print(f"🎹 MIDI Process Started for: {audio_path}")
        os.makedirs(output_dir, exist_ok=True)

        # Run Prediction
        predict_and_save(
            audio_path_list=[audio_path],
            output_directory=output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )

        # Rename generated file for consistency
        filename = os.path.basename(audio_path)
        base_name = os.path.splitext(filename)[0]

        # Basic Pitch creates file like: "songname_basic_pitch.mid"
        expected_name = f"{base_name}_basic_pitch.mid"

        src = os.path.join(output_dir, expected_name)
        dst = os.path.join(output_dir, "converted.mid")

        # If exact name match found
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            print(f"✅ SUCCESS: {dst}")

        else:
            # Fallback: Find ANY .mid file in the folder (Safety net)
            files = [f for f in os.listdir(output_dir) if f.endswith('.mid')]
            if files:
                src = os.path.join(output_dir, files[0])
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
                print(f"✅ SUCCESS: {dst}")
            else:
                print("❌ FAILURE: No MIDI file generated.")

    except Exception as e:
        print(f"❌ FAILURE: {e}")


if __name__ == "__main__":
    # Command Line Interface Logic
    # app.py calls this script like: python src/midify.py [input_file] [output_dir]

    if len(sys.argv) < 3:
        print("Usage: python src/midify.py <input_audio> <output_dir>")
    else:
        input_file = sys.argv[1]
        output_folder = sys.argv[2]

        if os.path.exists(input_file):
            audio_to_midi(input_file, output_folder)
        else:
            print(f"❌ Error: Input file not found: {input_file}")