import torch
import os
import glob

s3_dir = "features/s3"
output_path = "features/utt2s3.pt"
utt2s3 = {}

# 假设结果保存为 .pt 或类似格式，这里需要根据 extract_speech_token_mp.py 的实际输出来写
# 如果它输出的是 jsonl，就读 jsonl；如果是分开的 pt，就读 pt
# 假设生成了 features/s3/speech_tokens.jsonl
import json
with open(f"{s3_dir}/speech_tokens.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)
        # item 结构通常是 {'utt': 'xxx', 'token': [1, 2, 3...]}
        utt2s3[item['utt']] = torch.tensor(item['token'], dtype=torch.long)

torch.save(utt2s3, output_path)
print(f"Saved {len(utt2s3)} s3 tokens to {output_path}")