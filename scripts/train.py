#!/usr/bin/env python3
import math
import random
import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice

# --- Config Defaults ---
# 你可以在运行命令中修改这些，也可以在这里改默认值
DEFAULT_S3_VOCAB_SIZE = 4096
IGNORE_ID = -100

# ======================
#  1. Model Components
# ======================

def load_cosyvoice_llm(model_dir):
    print(f"Loading CosyVoice model from {model_dir}...")
    # 这一步会加载整个 CosyVoice，但我们只需要 LLM 部分
    cosy = CosyVoice(model_dir)
    return cosy.model.llm

class SimpleTextSpeechAggregator(nn.Module):
    def __init__(self, text_dim, speech_last_dim, speech_mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
        # Projections
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim)
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        # Q: Text (B, T_text, H)
        # K: Speech Last (B, T_speech, H)
        # V: Speech Mid (B, T_speech, H)
        Q = self.q_proj(text_emb)
        K = self.k_proj(speech_last)
        V = self.v_proj(speech_mid)

        # Attention Scores: (B, T_text, T_speech)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if speech_mask is not None:
            # mask: (B, 1, T_speech) - True is valid
            mask = speech_mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        z = torch.matmul(attn, V) # (B, T_text, H)
        
        return z, attn

class CosyVoiceS3Model(nn.Module):
    def __init__(self, llm, text_dim, speech_last_dim, speech_mid_dim, hidden_dim, s3_vocab_size):
        super().__init__()
        self.llm = llm
        self.s3_vocab_size = s3_vocab_size
        # S3 vocab + 1 for EOS/Special token
        self.s3_vocab_size_with_eos = s3_vocab_size + 1 
        
        self.aggregator = SimpleTextSpeechAggregator(
            text_dim, speech_last_dim, speech_mid_dim, hidden_dim
        )
        
        # Adapters
        self.input_proj = nn.Linear(hidden_dim, self.llm.output_size())
        self.proj = nn.Linear(self.llm.output_size(), self.s3_vocab_size_with_eos)
        
        # Embeddings (mimicking CosyVoice structure)
        # 0: sos/eos, 1: task_id
        self.llm_embedding = nn.Embedding(2, self.llm.output_size()) 
        self.speech_embedding = nn.Embedding(self.s3_vocab_size_with_eos, self.llm.output_size())

        # Fusion
        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(0.0))

        # Freeze LLM
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, text_emb, speech_last, speech_mid, speech_mask, text_mask, s3_targets):
        device = text_emb.device
        
        # 1. Aggregation & Fusion
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask)
        
        # Fuse: text + aligned_speech
        # Note: assuming text_dim == hidden_dim for simplicity here, otherwise project text first
        fused = self.ln_text(text_emb) + torch.sigmoid(self.fuse_alpha) * self.ln_z(z)
        
        # 2. Prepare LLM Inputs
        # Text Part
        fused_llm = self.input_proj(fused) # (B, T_text, D_llm)
        B = fused_llm.size(0)
        
        # Calculate lengths
        if text_mask is not None:
            text_lens = text_mask.sum(dim=1).int()
        else:
            text_lens = torch.full((B,), fused_llm.size(1), dtype=torch.int32, device=device)
            
        # S3 Part (Teacher Forcing Input)
        # Clamp to ensure valid indices
        speech_ids = s3_targets.clamp(min=0, max=self.s3_vocab_size - 1)
        speech_embeds = self.speech_embedding(speech_ids)
        
        # Calculate S3 lengths (assuming padding is handled externally or via simple check)
        # Here we assume s3_targets has 0 or some pad_id for padding.
        # But wait, standard s3 vocab includes 0? Let's assume PAD is handled in collate.
        # Using a mask from targets:
        s3_lens = (s3_targets != IGNORE_ID).sum(dim=1).int() # Rough check, better to pass lengths

        # Construct full input: [SOS] + [Text_Fused] + [Task] + [S3_Input]
        sos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1).expand(B, 1, -1)
        task_emb = self.llm_embedding.weight[1].reshape(1, 1, -1).expand(B, 1, -1)
        
        lm_input = torch.cat([sos_emb, fused_llm, task_emb, speech_embeds], dim=1)
        
        # Lengths for LLM (needed for rotary embedding etc)
        # Length = 1(SOS) + text + 1(Task) + s3
        lm_input_len = (1 + text_lens + 1 + s3_lens).int()

        # 3. LLM Forward
        hidden, _ = self.llm(lm_input, lm_input_len)
        logits = self.proj(hidden) # (B, Total_L, V)

        # 4. Loss Calculation
        # We only care about predicting S3 tokens.
        # Target Sequence construction:
        # Input pos:  [Task]   [S3_0] ... [S3_N-1]
        # Target pos: [S3_0]   [S3_1] ... [EOS]
        
        total_L = logits.size(1)
        lm_target = torch.full((B, total_L), IGNORE_ID, dtype=torch.long, device=device)
        
        for i in range(B):
            # The definition of prefix here: everything BEFORE the first predicted S3 token
            # Input: [SOS, ...Text..., Task, S3_0...]
            # The 'Task' token is at index: 1 + text_lens[i]
            # Its output should predict S3_0.
            
            # Start index in LOGITS that corresponds to predicting the first S3 token
            pred_start_idx = 1 + text_lens[i] + 1 - 1 # -1 because logits aligns with input
            # Actually, standard AR: input[k] predicts target[k].
            # Input index of 'Task' is (1 + text_lens[i]). 
            # So logits[1 + text_lens[i]] should predict S3_targets[0].
            
            t_len = text_lens[i]
            s_len = s3_lens[i]
            
            if s_len > 0:
                # Indices in Logits
                start = 1 + t_len  # Position of Task token
                end = start + s_len
                
                # Indices in Targets
                # We want targets to be: s3_targets[0], s3_targets[1]... s3_targets[last], EOS
                # But inputs were: Task, s3_targets[0]... s3_targets[last-1]
                
                # Copy s3 targets
                current_targets = s3_targets[i, :s_len]
                
                # Ensure we don't go out of bounds
                valid_len = min(s_len, total_L - start)
                lm_target[i, start : start + valid_len] = current_targets[:valid_len]
                
                # Add EOS if there is space
                if start + valid_len < total_L:
                     lm_target[i, start + valid_len] = self.s3_vocab_size # EOS
        
        loss = F.cross_entropy(
            logits.reshape(-1, self.s3_vocab_size_with_eos),
            lm_target.reshape(-1),
            ignore_index=IGNORE_ID
        )
        
        return loss, logits, attn

# ======================
#  2. Lazy Dataset
# ======================

class LazyS3Dataset(Dataset):
    def __init__(self, feature_dir, utt2s3_dict):
        """
        feature_dir: 存放 part_xx.pt 的目录
        utt2s3_dict: {utt_id: s3_tensor} (在内存中)
        """
        self.feature_dir = feature_dir
        self.utt2s3 = utt2s3_dict
        
        # 扫描文件
        self.pt_files = sorted(glob.glob(os.path.join(feature_dir, "part_*.pt")))
        if not self.pt_files:
            raise ValueError(f"No .pt files found in {feature_dir}")
        
        print(f"Indexing dataset from {len(self.pt_files)} chunks...")
        self.index = []
        
        # 建立索引：我们需要知道每个 key 在哪个文件里
        # 为了加快速度，这里假设文件名是 part_xx.pt，我们不得不加载一次来建立索引
        # 或者，如果文件名里没有信息，必须遍历。
        # 警告：这步会比较慢，但在 main 开始时只运行一次。
        for f_idx, f_path in enumerate(tqdm(self.pt_files, desc="Indexing")):
            try:
                # map_location='cpu' 稍微省点显存
                chunk = torch.load(f_path, map_location="cpu")
                # 假设 chunk结构: {'text_emb': {k:v}, 'whisper_mid':{k:v}...}
                # 我们只需要 key 列表
                keys = list(chunk["text_emb"].keys())
                
                for k in keys:
                    # 只有当我们在 utt2s3 里也有这个 key 时才加入训练集
                    if k in self.utt2s3:
                        self.index.append( (f_idx, k) )
                        
                del chunk # 立即释放
            except Exception as e:
                print(f"Error reading {f_path}: {e}")
                
        print(f"Indexed {len(self.index)} valid samples.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f_idx, key = self.index[idx]
        f_path = self.pt_files[f_idx]
        
        # 【性能瓶颈】
        # 每次为了读 1 个样本加载整个 pt 文件。
        # 在 Kaggle 上这是为了不 OOM 的妥协。
        # 建议设置 num_workers > 0 来掩盖 IO 延迟。
        chunk = torch.load(f_path, map_location="cpu")
        
        sample = {
            "utt_id": key,
            "text_emb": chunk["text_emb"][key],
            "speech_mid": chunk["whisper_mid"][key], # 对应你的 keys
            "speech_last": chunk["whisper_final"][key],
            "s3_targets": self.utt2s3[key] # 从内存字典取
        }
        return sample

def collate_fn(batch):
    # Filter out None
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    B = len(batch)
    
    # Get dimensions
    text_dim = batch[0]["text_emb"].size(-1)
    d_mid = batch[0]["speech_mid"].size(-1)
    d_last = batch[0]["speech_last"].size(-1)
    
    # Lengths
    text_lens = [b["text_emb"].size(0) for b in batch]
    speech_lens = [b["speech_mid"].size(0) for b in batch]
    s3_lens = [b["s3_targets"].size(0) for b in batch]
    
    max_text = max(text_lens)
    max_speech = max(speech_lens)
    max_s3 = max(s3_lens)
    
    # Alloc tensors
    text_emb = torch.zeros(B, max_text, text_dim)
    speech_mid = torch.zeros(B, max_speech, d_mid)
    speech_last = torch.zeros(B, max_speech, d_last)
    s3_targets = torch.full((B, max_s3), IGNORE_ID, dtype=torch.long)
    
    text_mask = torch.zeros(B, max_text, dtype=torch.bool)
    speech_mask = torch.zeros(B, max_speech, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        t_len = text_lens[i]
        s_len = speech_lens[i]
        s3_len = s3_lens[i]
        
        text_emb[i, :t_len] = b["text_emb"]
        text_mask[i, :t_len] = True
        
        speech_mid[i, :s_len] = b["speech_mid"]
        speech_last[i, :s_len] = b["speech_last"]
        speech_mask[i, :s_len] = True
        
        s3_targets[i, :s3_len] = b["s3_targets"]
        
    return {
        "text_emb": text_emb,
        "speech_mid": speech_mid,
        "speech_last": speech_last,
        "speech_mask": speech_mask,
        "text_mask": text_mask,
        "s3_targets": s3_targets
    }

# ======================
#  3. Main Logic
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=True, help="Directory containing part_*.pt")
    parser.add_argument("--utt2s3", required=True, help="Path to utt2s3.pt")
    parser.add_argument("--model_dir", required=True, help="CosyVoice model dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load S3 Dictionary (Small enough for RAM)
    print(f"Loading S3 tokens from {args.utt2s3}...")
    utt2s3_dict = torch.load(args.utt2s3, map_location="cpu")
    
    # 2. Dataset & DataLoader
    dataset = LazyS3Dataset(args.feature_dir, utt2s3_dict)
    
    # Split
    train_size = int(0.95 * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    # IMPORTANT: num_workers > 0 is crucial for LazyLoading efficiency
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=2
    )

    # 3. Init Model
    # We need to know dims from a sample (hacky but works)
    # Let's peek at the first item in the dataset logic
    # Or just hardcode if known. Whisper Base=512, Large=1280.
    # Safe way: Load one sample
    print("Peeking at data dimensions...")
    sample = dataset[0]
    text_dim = sample["text_emb"].shape[-1]
    speech_mid_dim = sample["speech_mid"].shape[-1]
    speech_last_dim = sample["speech_last"].shape[-1]
    
    llm = load_cosyvoice_llm(args.model_dir) # Only LLM part
    
    model = CosyVoiceS3Model(
        llm=llm,
        text_dim=text_dim,
        speech_last_dim=speech_last_dim,
        speech_mid_dim=speech_mid_dim,
        hidden_dim=text_dim, # Usually align with text dim
        s3_vocab_size=DEFAULT_S3_VOCAB_SIZE
    ).to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 4. Loop
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch} Train"):
            if batch is None: continue
            
            # Move to device
            t_emb = batch["text_emb"].to(device)
            s_mid = batch["speech_mid"].to(device)
            s_last = batch["speech_last"].to(device)
            s_mask = batch["speech_mask"].to(device)
            t_mask = batch["text_mask"].to(device)
            targets = batch["s3_targets"].to(device)
            
            loss, _, _ = model(t_emb, s_last, s_mid, s_mask, t_mask, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        # Validation (Simplified)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None: continue
                t_emb = batch["text_emb"].to(device)
                s_mid = batch["speech_mid"].to(device)
                s_last = batch["speech_last"].to(device)
                s_mask = batch["speech_mask"].to(device)
                t_mask = batch["text_mask"].to(device)
                targets = batch["s3_targets"].to(device)
                
                loss, _, _ = model(t_emb, s_last, s_mid, s_mask, t_mask, targets)
                val_loss += loss.item()
        
        avg_val = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

if __name__ == "__main__":
    main()