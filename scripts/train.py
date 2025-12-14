#!/usr/bin/env python3
import math
import random
from pathlib import Path
import gc
import os 
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler 
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice

# --- huggingface_hub compatibility patch (for CosyVoice) ---
try:
    import huggingface_hub as _hfh
    if not hasattr(_hfh, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download

        def cached_download(*args, **kwargs):
            return _hf_hub_download(*args, **kwargs)

        _hfh.cached_download = cached_download
except Exception:
    pass


# ======================
#  Config
# ======================

# 1. S3 Token (Label)
UTT2_S3_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\utt2speech_token.pt"        
# 2. Text Embedding (Input A)
UTT2_TEXT_EMB_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\utt2text.pt"   
# 3. Whisper Features (Input B)
UTT2_WHISPER_PATH = r"D:\EduKillers\25Second\DeepLearning\Assignment2\features\utt2whisper.pt" 
# 4. Pre-trained CosyVoice Model Directory
COSYVOICE_MODEL_DIR = r"D:\EduKillers\25Second\DeepLearning\Assignment2\models\CosyVoice-300M"

S3_PAD_ID = 0
S3_VOCAB_SIZE = 4096

# 【关键修改】Batch Size 改为 1 以防显存溢出
BATCH_SIZE = 1

LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 10
GRAD_CLIP = 1.0
TRAIN_RATIO = 0.95
IGNORE_ID = -100


# ======================
#  CosyVoice LLM wrapper
# ======================

def load_cosyvoice_llm(device):
    cosy = CosyVoice(COSYVOICE_MODEL_DIR)
    return cosy.model.llm.llm


class SimpleTextSpeechAggregator(nn.Module):
    def __init__(self, text_dim, speech_last_dim, speech_mid_dim, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim)
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        Q = self.q_proj(text_emb)        
        K = self.k_proj(speech_last)     
        V = self.v_proj(speech_mid)      

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if speech_mask is not None:
            mask = speech_mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e4)

        attn = F.softmax(scores, dim=-1) 
        z = torch.matmul(attn, V)        

        return z, attn


class CosyVoiceS3Model(nn.Module):
    def __init__(
        self,
        llm,
        text_dim,
        speech_last_dim,
        speech_mid_dim,
        hidden_dim,
        s3_vocab_size,
        s3_pad_id=0,
        freeze_llm=True,
    ):
        super().__init__()
        self.llm = llm
        self.aggregator = SimpleTextSpeechAggregator(
            text_dim=text_dim,
            speech_last_dim=speech_last_dim,
            speech_mid_dim=speech_mid_dim,
            hidden_dim=hidden_dim,
        )
        self.s3_pad_id = s3_pad_id
        self.s3_vocab_size = s3_vocab_size
        self.s3_vocab_size_with_eos = s3_vocab_size + 1 
        
        # 自动检测 LLM 维度
        if hasattr(self.llm, "output_size"):
             llm_dim = self.llm.output_size()
        elif hasattr(self.llm, "d_model"):
             llm_dim = self.llm.d_model
        else:
             llm_dim = 896 
             for m in self.llm.modules():
                 if isinstance(m, nn.Linear):
                     llm_dim = m.out_features
                     break

        self.input_proj = nn.Linear(text_dim, llm_dim)
        self.proj = nn.Linear(llm_dim, self.s3_vocab_size_with_eos)
        self.llm_embedding = nn.Embedding(2, llm_dim)
        self.speech_embedding = nn.Embedding(self.s3_vocab_size_with_eos, llm_dim)

        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(0.0))

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward(
        self,
        text_emb,
        speech_last,
        speech_mid,
        speech_mask=None,
        text_mask=None,
        s3_targets=None,
    ):
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask)
        fused = self.ln_text(text_emb) + torch.sigmoid(self.fuse_alpha) * self.ln_z(z)

        if text_mask is not None:
            text_lens = text_mask.sum(dim=1).to(dtype=torch.int32, device=fused.device)
        else:
            text_lens = torch.full(
                (fused.size(0),),
                fused.size(1),
                dtype=torch.int32,
                device=fused.device,
            )

        fused_llm = self.input_proj(fused) 
        B = fused_llm.size(0)
        device = fused_llm.device

        sos_eos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1).expand(B, 1, -1)
        task_id_emb = self.llm_embedding.weight[1].reshape(1, 1, -1).expand(B, 1, -1)

        speech_ids = s3_targets.clamp(min=0, max=self.s3_vocab_size - 1)
        speech_embeds = self.speech_embedding(speech_ids)

        s3_lens = (s3_targets != self.s3_pad_id).sum(dim=1).to(dtype=torch.int32, device=device)

        lm_input = torch.cat([sos_eos_emb, fused_llm, task_id_emb, speech_embeds], dim=1)
        lm_input_len = (1 + text_lens + 1 + s3_lens).to(dtype=torch.int32, device=device)

        # 【修复】直接调用 encoder
        if hasattr(self.llm, "encoder"):
             hidden, _ = self.llm.encoder(lm_input, lm_input_len)
        else:
             hidden, _ = self.llm(lm_input, lm_input_len)

        logits = self.proj(hidden) 

        total_len = logits.size(1)
        lm_target = torch.full((B, total_len), IGNORE_ID, dtype=torch.long, device=device)
        
        for i in range(B):
            slen = s3_lens[i]
            if slen > 0:
                start_predict_idx = 1 + text_lens[i]
                lm_target[i, start_predict_idx : start_predict_idx + slen] = s3_targets[i, :slen]
                if start_predict_idx + slen < total_len:
                    lm_target[i, start_predict_idx + slen] = self.s3_vocab_size

        loss = F.cross_entropy(
            logits.reshape(-1, self.s3_vocab_size_with_eos), 
            lm_target.reshape(-1), 
            ignore_index=IGNORE_ID
        )

        return loss, logits, attn


# ======================
#  Dataset / DataLoader
# ======================

class S3Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "utt_id": item["utt_id"],
            "text_emb": item["text_emb"].float(),
            "speech_mid": item["speech_mid"].float(),
            "speech_last": item["speech_last"].float(),
            "s3_tokens": item["s3_tokens"]
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    B = len(batch)
    
    text_lens = [b["text_emb"].size(0) for b in batch]
    speech_lens = [b["speech_mid"].size(0) for b in batch]
    s3_lens = []
    for b in batch:
        tokens = b["s3_tokens"]
        if torch.is_tensor(tokens):
            s3_lens.append(int(tokens.numel()))
        else:
            s3_lens.append(len(tokens))

    max_T_text = max(text_lens)
    max_T_speech = max(speech_lens)
    max_T_s3 = max(s3_lens)

    text_dim = batch[0]["text_emb"].size(-1)
    d_last = batch[0]["speech_last"].size(-1)
    d_mid = batch[0]["speech_mid"].size(-1)

    text_emb = torch.zeros(B, max_T_text, text_dim)
    speech_last = torch.zeros(B, max_T_speech, d_last)
    speech_mid = torch.zeros(B, max_T_speech, d_mid)
    speech_mask = torch.zeros(B, max_T_speech, dtype=torch.bool)
    s3_targets = torch.full((B, max_T_s3), S3_PAD_ID, dtype=torch.long)
    text_mask = torch.zeros(B, max_T_text, dtype=torch.bool)

    for i, b in enumerate(batch):
        tt = text_lens[i]
        ts = speech_lens[i]
        ts3 = s3_lens[i]

        text_emb[i, :tt] = b["text_emb"]
        speech_last[i, :ts] = b["speech_last"]
        speech_mid[i, :ts] = b["speech_mid"]
        speech_mask[i, :ts] = True
        tokens = b["s3_tokens"]
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.to(dtype=torch.long)
        s3_targets[i, :ts3] = tokens[:ts3]
        text_mask[i, :tt] = True

    return {
        "text_emb": text_emb,
        "speech_last": speech_last,
        "speech_mid": speech_mid,
        "speech_mask": speech_mask,
        "s3_targets": s3_targets,
        "text_mask": text_mask,
    }


def load_samples():
    print(f"[DEBUG] Loading S3 from {UTT2_S3_PATH} ...")
    utt2s3 = torch.load(UTT2_S3_PATH, map_location="cpu")
    print(f"[DEBUG] Loading TEXT from {UTT2_TEXT_EMB_PATH} ...")
    utt2text = torch.load(UTT2_TEXT_EMB_PATH, map_location="cpu")
    print(f"[DEBUG] Loading WHISPER from {UTT2_WHISPER_PATH} ...")
    utt2whisper = torch.load(UTT2_WHISPER_PATH, map_location="cpu")
    mid_keys = set(utt2whisper['mid'].keys())
    final_keys = set(utt2whisper['final'].keys())
    
    del utt2whisper
    gc.collect()

    s3_set = set(utt2s3.keys())
    text_set = set(utt2text.keys())
    whisper_set = mid_keys & final_keys
    
    intersection = s3_set & text_set & whisper_set
    
    keys = []
    if len(intersection) == 0:
        print("⚠️ [WARNING] 0 samples matched! Attempting CLEANED keys matching...")
        def clean_key(k):
            k = str(k).replace('\\', '/')
            return k.split('/')[-1].split('.')[0]
            
        s3_clean = {clean_key(k): k for k in s3_set}
        text_clean = {clean_key(k): k for k in text_set}
        whisper_clean = {clean_key(k): k for k in whisper_set}
        
        clean_intersection = set(s3_clean.keys()) & set(text_clean.keys()) & set(whisper_clean.keys())
        if len(clean_intersection) > 0:
            print(f"✅ Cleaned Keys Matched: {len(clean_intersection)}")
            for clean_k in clean_intersection:
                keys.append({
                    'clean_id': clean_k,
                    's3_raw': s3_clean[clean_k],
                    'text_raw': text_clean[clean_k],
                    'whisper_mid_raw': whisper_clean[clean_k], 
                    'whisper_final_raw': whisper_clean[clean_k]
                })
    else:
        keys = [{'clean_id': k, 's3_raw': k, 'text_raw': k, 'whisper_mid_raw': k, 'whisper_final_raw': k} for k in intersection]

    # Reload whisper needed parts properly
    utt2whisper = torch.load(UTT2_WHISPER_PATH, map_location="cpu") # Reload since we deleted it
    
    samples = []
    for item in keys:
        s3_tokens = utt2s3[item['s3_raw']]
        text_emb = utt2text[item['text_raw']]
        speech_mid = utt2whisper['mid'][item['whisper_mid_raw']]
        speech_last = utt2whisper['final'][item['whisper_final_raw']]

        if (s3_tokens is None) or (text_emb is None) or (speech_mid is None) or (speech_last is None): continue
        if (getattr(text_emb, "numel", lambda: 0)() == 0): continue
        
        samples.append({
            "utt_id": item['clean_id'],
            "text_emb": text_emb,
            "speech_mid": speech_mid,
            "speech_last": speech_last,
            "s3_tokens": s3_tokens,
        })
    
    # Clean up again
    del utt2s3, utt2text, utt2whisper
    gc.collect()
    
    return samples


# ======================
#  Train / Eval / Predict
# ======================

# 【修改】增加 scaler 参数
def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_losses = []

    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        if batch is None: continue
        
        text_emb = batch["text_emb"].to(device)
        speech_last = batch["speech_last"].to(device)
        speech_mid = batch["speech_mid"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        s3_targets = batch["s3_targets"].to(device)

        optimizer.zero_grad()

        # 【新增】混合精度上下文
        with autocast():
            loss, _, _ = model(
                text_emb=text_emb,
                speech_last=speech_last,
                speech_mid=speech_mid,
                speech_mask=speech_mask,
                text_mask=text_mask,
                s3_targets=s3_targets,
            )

        # 【新增】Scaler 反向传播
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()

        # 【记录】
        loss_val = loss.item()
        step_losses.append(loss_val)  # 记录这一步的 loss

        num_valid = (s3_targets != S3_PAD_ID).sum().item()
        total_loss += loss.item() * num_valid 
        total_tokens += num_valid
        
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, step_losses  # 【修改】返回平均值和详细列表


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    # 只需要打印一次预览
    preview_done = False

    for batch in tqdm(dataloader, desc="Evaluating"):
        if batch is None: continue
        
        text_emb = batch["text_emb"].to(device)
        speech_last = batch["speech_last"].to(device)
        speech_mid = batch["speech_mid"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        s3_targets = batch["s3_targets"].to(device)

        with autocast():
            # 获取 logits
            loss, logits, _ = model(
                text_emb=text_emb,
                speech_last=speech_last,
                speech_mid=speech_mid,
                speech_mask=speech_mask,
                text_mask=text_mask,
                s3_targets=s3_targets,
            )

        # ================== 【开始】关键修复部分 ==================
        
        # 1. 获取预测结果 [Batch, Total_Len]
        preds = torch.argmax(logits, dim=-1)
        
        # 2. 我们需要手动构建一个和 preds 形状一样的目标 tensor (lm_target)
        # 这段逻辑必须和 model.forward 里的逻辑完全一致，才能对齐
        B = preds.size(0)
        total_len = preds.size(1)
        
        # 初始化全是 IGNORE_ID (-100)
        lm_target_aligned = torch.full((B, total_len), IGNORE_ID, dtype=torch.long, device=device)
        
        # 计算长度
        text_lens = text_mask.sum(dim=1).to(dtype=torch.int32)
        s3_lens = (s3_targets != S3_PAD_ID).sum(dim=1).to(dtype=torch.int32)
        
        # 填充 valid 的标签到正确位置
        for i in range(B):
            slen = s3_lens[i].item()
            if slen > 0:
                # 起始位置 = 1(SOS) + text_len + 1(TaskID) - 1 (索引从0开始修正) -> 实际上是 1 + text_lens
                # 参照 forward 里的逻辑： start_predict_idx = 1 + text_lens[i]
                start_idx = 1 + text_lens[i].item()
                
                # 确保不越界
                valid_len = min(slen, total_len - start_idx)
                if valid_len > 0:
                    lm_target_aligned[i, start_idx : start_idx + valid_len] = s3_targets[i, :valid_len]
                    
                    # (可选) 如果 forward 里也预测了 EOS，这里也可以加上
                    # if start_idx + valid_len < total_len:
                    #     lm_target_aligned[i, start_idx + valid_len] = model.s3_vocab_size

        # 3. 计算 Accuracy
        # 只比较那些不是 IGNORE_ID 的位置
        valid_mask = (lm_target_aligned != IGNORE_ID)
        correct = (preds == lm_target_aligned) & valid_mask
        
        total_correct += correct.sum().item()
        num_valid_tokens = valid_mask.sum().item() # 使用对齐后的有效token数
        
        # ================== 【结束】关键修复部分 ==================

        total_loss += loss.item() * num_valid_tokens
        total_tokens += num_valid_tokens

        if not preview_done:
            # 找到第一个非空的样本看看
            idx = 0
            valid_indices = torch.where(lm_target_aligned[idx] != IGNORE_ID)[0]
            if len(valid_indices) > 0:
                start = valid_indices[0].item()
                end = valid_indices[-1].item() + 1
                # 只打印有效片段的前10个
                print(f"\n[Preview] Target (Aligned): {lm_target_aligned[idx, start:end][:10].tolist()}")
                print(f"[Preview] Pred   (Aligned): {preds[idx, start:end][:10].tolist()}")
                preview_done = True

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = total_correct / max(1, total_tokens)
    
    return avg_loss, accuracy


@torch.no_grad()
def predict_s3(model, text_emb, speech_last, speech_mid, device):
    model.eval()
    
    text_emb = text_emb.unsqueeze(0).to(device)       
    speech_last = speech_last.unsqueeze(0).to(device) 
    speech_mid = speech_mid.unsqueeze(0).to(device)   
    
    speech_mask = torch.ones((1, speech_mid.size(1)), dtype=torch.bool, device=device)
    
    z, _ = model.aggregator(text_emb, speech_last, speech_mid, speech_mask)
    fused = model.ln_text(text_emb) + torch.sigmoid(model.fuse_alpha) * model.ln_z(z)
    
    fused_llm = model.input_proj(fused)
    
    sos_emb = model.llm_embedding.weight[0].reshape(1, 1, -1)
    task_emb = model.llm_embedding.weight[1].reshape(1, 1, -1)
    
    curr_input = torch.cat([sos_emb, fused_llm, task_emb], dim=1) 
    
    generated_ids = []
    max_steps = 400 
    
    for _ in range(max_steps):
        L = curr_input.size(1)
        length_tensor = torch.tensor([L], dtype=torch.int32, device=device)
        
        if hasattr(model.llm, "encoder"):
             hidden, _ = model.llm.encoder(curr_input, length_tensor)
        else:
             hidden, _ = model.llm(curr_input, length_tensor)
        
        last_hidden = hidden[:, -1:, :]
        logits = model.proj(last_hidden) 
        
        next_id = torch.argmax(logits, dim=-1).item()
        
        if next_id == model.s3_vocab_size:
            break
            
        generated_ids.append(next_id)
        
        next_id_clamped = min(next_id, model.s3_vocab_size - 1)
        next_token_tensor = torch.tensor([[next_id_clamped]], dtype=torch.long, device=device)
        next_emb = model.speech_embedding(next_token_tensor) 
        
        curr_input = torch.cat([curr_input, next_emb], dim=1)
        
    return generated_ids


# ======================
#  Main
# ======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_samples()
    print(f"Loaded {len(samples)} samples.")
    
    if len(samples) == 0:
        print("No samples found. Exiting.")
        return

    random.shuffle(samples)

    n_train = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:n_train]
    valid_samples = samples[n_train:]

    train_ds = S3Dataset(train_samples)
    valid_ds = S3Dataset(valid_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 # Windows下多进程可能导致OOM或Pickle错误，设为0
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    example = train_samples[0]
    text_dim = example["text_emb"].size(-1)
    d_last = example["speech_last"].size(-1)
    d_mid = example["speech_mid"].size(-1)
    
    llm_wrapper = load_cosyvoice_llm(device)

    # 记录开始时间
    start_time = time.time()
    
    # 【新增】用于保存所有日志数据
    history = {
        "config": {
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "optimizer": "AdamW",
            "model": "CosyVoice-300M + Adapter"
        },
        "steps": [],          # 所有的 step loss (画平滑曲线用)
        "epochs": []          # 每个 epoch 的验证集 accuracy (画点用)
    }

    model = CosyVoiceS3Model(
        llm=llm_wrapper,
        text_dim=text_dim,
        speech_last_dim=d_last,
        speech_mid_dim=d_mid,
        hidden_dim=text_dim,  
        s3_vocab_size=S3_VOCAB_SIZE,
        s3_pad_id=S3_PAD_ID,
        freeze_llm=True,         
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # 【新增】初始化 Scaler
    scaler = GradScaler()
    
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(save_dir)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        # 接收 step_losses
        train_loss, step_losses = train_one_epoch(model, train_loader, optimizer, device, scaler)
        
        # 接收 accuracy
        valid_loss, valid_acc = eval_one_epoch(model, valid_loader, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4%}")

        # 【记录数据】
        history["steps"].extend(step_losses) # 将这一轮的所有step loss加入总表
        history["epochs"].append({
            "epoch": epoch,
            "train_loss_avg": train_loss,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc
        })

        # 保存 checkpoint (文件名加上 acc 更直观)
        save_path = os.path.join(save_dir, f"epoch_{epoch:02d}_acc_{valid_acc:.4f}.pt")
        adapter_state = {k: v for k, v in model.state_dict().items() if "llm." not in k}
        torch.save(adapter_state, save_path)

        # 【实时保存日志】防止半路崩了数据丢失
        with open("training_log.json", "w") as f:
            json.dump(history, f, indent=4)

    # 记录总耗时
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    
    # 最终更新一下日志里的时间
    history["total_time_seconds"] = total_time
    with open("training_log.json", "w") as f:
        json.dump(history, f, indent=4)

    print("✅ Training Log saved to 'training_log.json'.")

if __name__ == "__main__":
    main()