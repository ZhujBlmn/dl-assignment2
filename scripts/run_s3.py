import os
import subprocess
import glob

def main():
    # Config
    librispeech_dir = r"D:\EduKillers\25Second\DeepLearning\Assignment2\origin_data\LibriSpeech\test-clean"
    out_dir = "features"
    onnx_path = r"D:\EduKillers\25Second\DeepLearning\Assignment2\models\CosyVoice-300M\speech_tokenizer_v1.onnx"
    extract_script = os.path.join("CosyVoice", "tools", "extract_speech_token.py")
    LIMIT = 1e9

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    # 2. generate wav.scp
    wav_scp_path = os.path.join(out_dir, "wav.scp")
    print(f"Generating {wav_scp_path} ...")
    
    count = 0
    stop_processing = False

    with open(wav_scp_path, "w", encoding="utf-8") as f:
        # find .flac or .wav
        for root, dirs, files in os.walk(librispeech_dir):
            if stop_processing:
                break
                
            for file in files:
                if file.endswith(".flac") or file.endswith(".wav"):
                    # check limit
                    if count >= LIMIT:
                        print(f"⚠️ Reached limit of {LIMIT} files. Stopping scan.")
                        stop_processing = True
                        break

                    full_path = os.path.join(root, file)
                    file_id = os.path.splitext(file)[0]
                    f.write(f"{file_id} {full_path}\n")
                    count += 1
    
    print(f"Generated wav.scp with {count} lines.")

    print("Starting speech token extraction...")
    
    cmd = [
        "python", extract_script,
        "--dir", out_dir,
        "--onnx_path", onnx_path
    ]
    
    print("Executing command:", " ".join(cmd))
    
    try:
        # run the extraction script
        subprocess.run(cmd, check=True)
        print("✅ Extraction finished successfully!")
    except subprocess.CalledProcessError as e:
        print("❌ Extraction failed.")
        print("Error details:", e)
    except FileNotFoundError:
        print(f"❌ Error: Could not find python or the script '{extract_script}'.")
        print("Please check if you are in the root directory of 'dl-assignment2'.")

if __name__ == "__main__":
    main()