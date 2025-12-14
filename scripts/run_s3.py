import os
import subprocess
import glob

def main():
    # ================= 配置区域 (请根据你的本地路径修改这里) =================
    
    # 1. LibriSpeech 数据集的根目录
    librispeech_dir = r"D:\EduKillers\25Second\DeepLearning\Assignment2\origin_data\LibriSpeech\test-clean"
    
    # 2. 输出目录
    out_dir = "features"
    
    # 3. ONNX 模型路径
    onnx_path = r"D:\EduKillers\25Second\DeepLearning\Assignment2\models\CosyVoice-300M\speech_tokenizer_v1.onnx"
    
    # 4. 提取脚本的路径
    extract_script = os.path.join("CosyVoice", "tools", "extract_speech_token.py")
    
    # 5. 限制处理的数据量 
    LIMIT = 1e9
    
    # =======================================================================

    # 1. 创建输出目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    # 2. 生成 wav.scp 文件 (带数量限制)
    wav_scp_path = os.path.join(out_dir, "wav.scp")
    print(f"Generating {wav_scp_path} ...")
    
    count = 0
    stop_processing = False # 停止标志位

    with open(wav_scp_path, "w", encoding="utf-8") as f:
        # 遍历目录找 .flac 或 .wav 文件
        for root, dirs, files in os.walk(librispeech_dir):
            if stop_processing:
                break
                
            for file in files:
                if file.endswith(".flac") or file.endswith(".wav"):
                    # 如果达到限制，停止写入并退出循环
                    if count >= LIMIT:
                        print(f"⚠️ Reached limit of {LIMIT} files. Stopping scan.")
                        stop_processing = True
                        break

                    full_path = os.path.join(root, file)
                    
                    # 生成 ID (直接用文件名作为 ID)
                    file_id = os.path.splitext(file)[0]
                    
                    f.write(f"{file_id} {full_path}\n")
                    count += 1
    
    print(f"Generated wav.scp with {count} lines.")

    # 3. 调用提取脚本
    print("Starting speech token extraction...")
    
    # 构造运行命令
    cmd = [
        "python", extract_script,
        "--dir", out_dir,
        "--onnx_path", onnx_path
    ]
    
    print("Executing command:", " ".join(cmd))
    
    try:
        # 运行命令
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