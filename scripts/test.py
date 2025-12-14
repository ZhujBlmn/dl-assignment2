import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import CosyVoiceS3Model, S3Dataset, collate_fn, load_cosyvoice_llm, IGNORE_ID, S3_PAD_ID

TEST_S3_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\test_utt2speech_token.pt"
TEST_TEXT_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\test_utt2text.pt"
TEST_WHISPER_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\test_utt2whisper.pt"
CKPT_PATH = r"checkpoints\epoch_10_loss_4.6890.pt"

def clean_key(k):
    k = str(k).replace('\\', '/')
    return k.split('/')[-1].split('.')[0]

def load_test_data():
    s3 = torch.load(TEST_S3_PATH, map_location='cpu')
    text = torch.load(TEST_TEXT_PATH, map_location='cpu')
    whisper = torch.load(TEST_WHISPER_PATH, map_location='cpu')

    print(f" - S3 Keys: {len(s3)}")
    print(f" - Text Keys: {len(text)}")
    print(f" - Whisper Keys: {len(whisper['mid'])}")

    keys = set(s3.keys()) & set(text.keys()) & set(whisper['mid'].keys())
    
    if len(keys) == 0:
        print("need to clean the key...")
        
        # 建立 {clean_key: original_key} 的映射
        s3_map = {clean_key(k): k for k in s3.keys()}
        text_map = {clean_key(k): k for k in text.keys()}
        whisper_map = {clean_key(k): k for k in whisper['mid'].keys()}
        
        clean_keys = set(s3_map.keys()) & set(text_map.keys()) & set(whisper_map.keys())
        
        samples = []
        for ck in clean_keys:

            orig_s3 = s3_map[ck]
            orig_text = text_map[ck]
            orig_whisper = whisper_map[ck]
            
            samples.append({
                "utt_id": ck,
                "text_emb": text[orig_text],
                "s3_tokens": s3[orig_s3],
                "speech_mid": whisper['mid'][orig_whisper],
                "speech_last": whisper['final'][orig_whisper] 
            })
    else:
        samples = [{
            "utt_id": k, 
            "text_emb": text[k], 
            "s3_tokens": s3[k], 
            "speech_mid": whisper['mid'][k], 
            "speech_last": whisper['final'][k]
        } for k in keys]

    print(f"test samples: {len(samples)}")
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. prepare data loader
    samples = load_test_data()
    loader = DataLoader(S3Dataset(samples), batch_size=1, collate_fn=collate_fn)
    
    # 2. build model
    sample = samples[0]
    llm = load_cosyvoice_llm(device)
    model = CosyVoiceS3Model(
        llm=llm, 
        text_dim=sample["text_emb"].shape[-1], 
        speech_last_dim=sample["speech_last"].shape[-1],
        speech_mid_dim=sample["speech_mid"].shape[-1],
        hidden_dim=sample["text_emb"].shape[-1],
        s3_vocab_size=4096
    ).to(device)

    # 3. load checkpoint
    print(f"loading checkpoint: {CKPT_PATH}")
    if not os.path.exists(CKPT_PATH):
        print(f"cannot find the file: {CKPT_PATH}")
        return
        
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 4. calculate accuracy
    total_correct, total_valid = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if batch is None: continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            
            _, logits, _ = model(**batch)
            preds = torch.argmax(logits, dim=-1)

            s3_targets = batch['s3_targets']
            text_len = batch['text_mask'].sum(dim=1)
            
            for i in range(len(preds)):
                tgt_len = (s3_targets[i] != S3_PAD_ID).sum().item()
                if tgt_len == 0: continue
                
                start = 1 + text_len[i].item()
                valid_len = min(tgt_len, preds.size(1) - start)
                
                if valid_len > 0:
                    p = preds[i, start : start+valid_len]
                    t = s3_targets[i, :valid_len]
                    total_correct += (p == t).sum().item()
                    total_valid += valid_len

    print(f"Top-1 Acc: {total_correct/total_valid:.4%}")

if __name__ == "__main__":
    main()