import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_jsonl(data_dir, output_jsonl):
    """LibriSpeech数据Prepare"""
    
    items = []
    data_path = Path(data_dir)
    
    # 查找所有转录文件
    trans_files = list(data_path.rglob("*.trans.txt"))
    print(f"找到 {len(trans_files)} 个转录文件")
    
    # 添加进度条
    for trans_file in tqdm(trans_files, desc="处理转录文件"):
        with open(trans_file, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                
                # 分割utterance_id和文本
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    
                    # 直接在同一目录查找音频文件
                    audio_dir = trans_file.parent
                    for ext in ['.wav', '.flac']:
                        audio_path = audio_dir / f"{utterance_id}{ext}"
                        if audio_path.exists():
                            items.append({
                                "audio_path": str(audio_path),
                                "text": text.strip(),
                                "utterance_id": utterance_id
                            })
                            break
    
    # 保存结果
    with open(output_jsonl, "w", encoding="utf8") as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(items)} items to {output_jsonl}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    
    prepare_jsonl(args.librispeech_dir, args.output_jsonl)