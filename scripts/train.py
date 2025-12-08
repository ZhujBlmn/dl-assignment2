#!/usr/bin/env python3
import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# 请确保这些路径指向你的文件
# 如果你只跑了前10%的数据，确保这些文件只包含那部分数据，否则加载时可能会爆内存
UTT2_S3_PATH = "features/utt2speech_token.pt"        
UTT2_TEXT_EMB_PATH = "features/chunks/part_000.pt"   # 示例：直接指向生成的那个 chunk 文件
# 注意：如果你有多个 chunk，你需要先合并它们或者在 load_samples 里写循环读取
# 为了简单演示，这里假设你把所有需要的特征都存到了这个文件里，或者只用这一个文件跑

UTT2_WHISPER_PATH = "features/chunks/part_000.pt" # 通常 text 和 whisper 在同一个 chunk 里
COSYVOICE_MODEL_DIR = "pretrained_models/CosyVoice-300M"

S3_PAD_ID = 0
S3_VOCAB_SIZE = 4096
BATCH_SIZE = 8
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
    print(f"Loading CosyVoice from {COSYVOICE_MODEL_DIR}...")
    cosy = CosyVoice(COSYVOICE_MODEL_DIR)
    return cosy.model.llm


class SimpleTextSpeechAggregator(nn.Module):
    """
    Q = text_emb         : (B, T_text, D_text)
    K = speech_last      : (B, T_speech, D_last)
    V = speech_mid       : (B, T_speech, D_mid)

    Output:
        z   : (B, T_text, hidden_dim)
        att : (B, T_text, T_speech)
    """
    def __init__(self, text_dim, speech_last_dim, speech_mid_dim, hidden_dim):
        super().__init__()
        # -------------------------------------------------------
        # TODO (init): Implement Linear layers
        # -------------------------------------------------------
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim)
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        # -------------------------------------------------------
        # TODO (forward): Scaled Dot-Product Attention
        # -------------------------------------------------------
        
        # 1) Project inputs
        Q = self.q_proj(text_emb)       # (B, T_text, H)
        K = self.k_proj(speech_last)    # (B, T_speech, H)
        V = self.v_proj(speech_mid)     # (B, T_speech, H)

        # 2) Compute attention scores: Q * K^T
        # (B, T_text, H) @ (B, H, T_speech) -> (B, T_text, T_speech)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 3) Masking
        if speech_mask is not None:
            # speech_mask: (B, T_speech). Expand to (B, 1, T_speech) for broadcasting
            mask = speech_mask.unsqueeze(1)
            # Fill padded positions with very small negative number
            scores = scores.masked_fill(~mask, -1e9)

        # 4) Softmax
        attn = F.softmax(scores, dim=-1)  # (B, T_text, T_speech)

        # 5) Compute attended representation: z = Attn * V
        # (B, T_text, T_speech) @ (B, T_speech, H) -> (B, T_text, H)
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
        self.s3_vocab_size_with_eos = s3_vocab_size + 1  # extra EOS
        
        self.input_proj = nn.Linear(text_dim, self.llm.output_size())
        self.proj = nn.Linear(self.llm.output_size(), self.s3_vocab_size_with_eos)
        
        # prefix embeddings
        self.llm_embedding = nn.Embedding(2, self.llm.output_size())  # 0: sos_eos, 1: task_id
        self.speech_embedding = nn.Embedding(self.s3_vocab_size_with_eos, self.llm.output_size())

        # fusion
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
        # -------------------------------------------------------
        # TODO (step 1: aggregation + fusion)
        # -------------------------------------------------------
        
        # 1) Call Aggregator
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask)

        # 2) Fusion: v + sigmoid(alpha) * z
        fused = self.ln_text(text_emb) + torch.sigmoid(self.fuse_alpha) * self.ln_z(z)

        # ========== Below: text lengths and LLM input construction ==========

        if text_mask is not None:
            text_lens = text_mask.sum(dim=1).to(dtype=torch.int32, device=fused.device)
        else:
            text_lens = torch.full(
                (fused.size(0),),
                fused.size(1),
                dtype=torch.int32,
                device=fused.device,
            )

        # project fused into llm input space
        fused_llm = self.input_proj(fused)  # (B, T_text, D_llm_in)
        B = fused_llm.size(0)
        device = fused_llm.device

        # prepare prefixes
        sos_eos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1).expand(B, 1, -1)
        task_id_emb = self.llm_embedding.weight[1].reshape(1, 1, -1).expand(B, 1, -1)

        speech_ids = s3_targets.clamp(min=0, max=self.s3_vocab_size - 1)  # (B, T_s3)
        speech_embeds = self.speech_embedding(speech_ids)  # (B, T_s3, D_llm_in)

        s3_lens = (s3_targets != self.s3_pad_id).sum(dim=1).to(dtype=torch.int32, device=device)  # (B,)

        lm_input = torch.cat([sos_eos_emb, fused_llm, task_id_emb, speech_embeds], dim=1)  # (B, L, D)
        lm_input_len = (1 + text_lens + 1 + s3_lens).to(dtype=torch.int32, device=device)  # (B,)

        hidden, _ = self.llm(lm_input, lm_input_len)  # (B, L, H)
        logits = self.proj(hidden)                    # (B, L, V+1)

        # -------------------------------------------------------
        # TODO (targets + loss)
        # -------------------------------------------------------
        
        total_L = logits.size(1)
        lm_target = torch.full((B, total_L), IGNORE_ID, dtype=torch.long, device=device)
        
        for i in range(B):
            # Input: [SOS, Text..., Task, S3_0, S3_1... S3_Last]
            # Prediction Target:
            #   Task   -> S3_0
            #   S3_0   -> S3_1
            #   ...
            #   S3_Last-> EOS
            
            t_len = text_lens[i]
            s_len = s3_lens[i]
            
            # The 'Task' token is at index: 1 + t_len
            # We want to start predicting from there.
            if s_len > 0:
                start_idx = 1 + t_len
                end_idx = start_idx + s_len
                
                # Copy S3 targets
                # Be careful with bounds
                valid_len = min(s_len, total_L - start_idx)
                lm_target[i, start_idx : start_idx + valid_len] = s3_targets[i, :valid_len]
                
                # Add EOS if possible
                if start_idx + valid_len < total_L:
                    lm_target[i, start_idx + valid_len] = self.s3_vocab_size # EOS

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
        return self.samples[idx]


def collate_fn(batch):
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
    print(f"Loading data from {UTT2_TEXT_EMB_PATH} ...")
    
    # 假设你之前跑的 chunk 文件里包含了 text_emb, whisper_mid, whisper_final
    # 如果你是分开存的，请相应修改加载逻辑
    # 这里为了兼容之前的 chunk 结构：
    # chunk = {'text_emb': {id: ...}, 'whisper_mid': {id: ...}, 'whisper_final': {id: ...}}
    
    chunk_data = torch.load(UTT2_TEXT_EMB_PATH, map_location="cpu")
    utt2s3 = torch.load(UTT2_S3_PATH, map_location="cpu")
    
    # 如果你的chunk文件里也是字典结构
    utt2text = chunk_data.get("text_emb", {})
    utt2whisper_mid = chunk_data.get("whisper_mid", {})
    utt2whisper_final = chunk_data.get("whisper_final", {})
    
    # 求交集
    keys = sorted(set(utt2s3.keys()) & set(utt2text.keys()) & set(utt2whisper_mid.keys()))
    
    samples = []
    for key in keys:
        s3_tokens = utt2s3[key]
        text_emb = utt2text[key]
        speech_mid = utt2whisper_mid[key]
        speech_last = utt2whisper_final[key]

        if (s3_tokens is None) or (text_emb is None) or (speech_mid is None) or (speech_last is None):
            continue
        
        # 转换为 float (以防存的是 half)
        if text_emb.dtype == torch.float16: text_emb = text_emb.float()
        if speech_mid.dtype == torch.float16: speech_mid = speech_mid.float()
        if speech_last.dtype == torch.float16: speech_last = speech_last.float()

        samples.append({
            "utt_id": key,
            "text_emb": text_emb,
            "speech_mid": speech_mid,
            "speech_last": speech_last,
            "s3_tokens": s3_tokens,
        })
    
    print(f"Loaded {len(samples)} samples.")
    return samples


# ======================
#  Train / Eval / Predict
# ======================

def train_one_epoch(model, dataloader, optimizer, device):
    # -------------------------------------------------------
    # TODO (train_one_epoch)
    # -------------------------------------------------------
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None: continue
        
        # 1) Move to device
        text_emb = batch["text_emb"].to(device)
        speech_last = batch["speech_last"].to(device)
        speech_mid = batch["speech_mid"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        s3_targets = batch["s3_targets"].to(device)
        
        # 2) Forward
        loss, logits, _ = model(
            text_emb, speech_last, speech_mid, speech_mask, text_mask, s3_targets
        )
        
        # 3) Backward
        optimizer.zero_grad()
        loss.backward()
        
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(1, batch_count)


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    # -------------------------------------------------------
    # TODO (eval_one_epoch)
    # -------------------------------------------------------
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        if batch is None: continue
        
        text_emb = batch["text_emb"].to(device)
        speech_last = batch["speech_last"].to(device)
        speech_mid = batch["speech_mid"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        s3_targets = batch["s3_targets"].to(device)
        
        loss, _, _ = model(
            text_emb, speech_last, speech_mid, speech_mask, text_mask, s3_targets
        )
        
        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(1, batch_count)


@torch.no_grad()
def predict_s3(model, text_emb, speech_last, speech_mid, device, max_steps=500):
    """
    Autoregressive decoding
    """
    # -------------------------------------------------------
    # TODO (predict_s3)
    # -------------------------------------------------------
    model.eval()
    
    # 1) Add Batch Dim & Move to device
    text_emb = text_emb.unsqueeze(0).to(device)       # (1, T_txt, D)
    speech_last = speech_last.unsqueeze(0).to(device) # (1, T_sp, D)
    speech_mid = speech_mid.unsqueeze(0).to(device)   # (1, T_sp, D)
    
    # 2) Full Speech Mask (since no padding in single inference)
    speech_mask = torch.ones(1, speech_last.size(1), dtype=torch.bool, device=device)
    
    # 3) Aggregation & Fusion
    z, attn = model.aggregator(text_emb, speech_last, speech_mid, speech_mask)
    fused = model.ln_text(text_emb) + torch.sigmoid(model.fuse_alpha) * model.ln_z(z)
    fused_llm = model.input_proj(fused) # (1, T_txt, D_llm)
    
    # 4) Build Initial Sequence: [SOS] + [Fused] + [Task]
    sos_emb = model.llm_embedding.weight[0].reshape(1, 1, -1)
    task_emb = model.llm_embedding.weight[1].reshape(1, 1, -1)
    
    curr_input = torch.cat([sos_emb, fused_llm, task_emb], dim=1)
    
    generated_ids = []
    
    # 5) Autoregressive Loop
    for _ in range(max_steps):
        # Length
        input_len = torch.tensor([curr_input.size(1)], dtype=torch.int32, device=device)
        
        # Forward LLM
        hidden, _ = model.llm(curr_input, input_len)
        
        # Predict next token from the last position
        last_hidden = hidden[:, -1, :] # (1, D)
        logits = model.proj(last_hidden) # (1, V+1)
        
        # Greedy decoding
        next_id = torch.argmax(logits, dim=-1).item()
        
        # Stop if EOS
        if next_id == model.s3_vocab_size:
            break
            
        generated_ids.append(next_id)
        
        # Prepare next input embedding
        # Clamp to valid range
        safe_id = min(next_id, model.s3_vocab_size - 1)
        next_emb = model.speech_embedding(torch.tensor([[safe_id]], device=device))
        
        # Append to input
        curr_input = torch.cat([curr_input, next_emb], dim=1)
    
    return torch.tensor(generated_ids), attn


# ======================
#  Main
# ======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_samples()
    random.shuffle(samples)

    # 简单切分训练/验证集
    n_train = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:n_train]
    valid_samples = samples[n_train:]
    
    print(f"Train samples: {len(train_samples)}, Valid samples: {len(valid_samples)}")

    train_ds = S3Dataset(train_samples)
    valid_ds = S3Dataset(valid_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if len(train_samples) == 0:
        print("No samples loaded. Exiting.")
        return

    example = train_samples[0]
    text_dim = example["text_emb"].size(-1)
    d_last = example["speech_last"].size(-1)
    d_mid = example["speech_mid"].size(-1)
    
    llm = load_cosyvoice_llm(device)

    model = CosyVoiceS3Model(
        llm=llm,
        text_dim=text_dim,
        speech_last_dim=d_last,
        speech_mid_dim=d_mid,
        hidden_dim=text_dim,  
        s3_vocab_size=S3_VOCAB_SIZE,
        s3_pad_id=S3_PAD_ID,
        freeze_llm=True,        
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = eval_one_epoch(model, valid_loader, device)
        print(f"Epoch {epoch:02d} | train_loss = {train_loss:.4f} | valid_loss = {valid_loss:.4f}")

    # Example usage of predict_s3
    if len(valid_samples) > 0:
        print("\n--- Running Inference Example ---")
        ex = valid_samples[0]
        pred_s3, _ = predict_s3(
            model,
            ex["text_emb"].float(),
            ex["speech_last"].float(),
            ex["speech_mid"].float(),
            device,
        )
        print(f"Ground Truth S3: {ex['s3_tokens'][:20].tolist()}...")
        print(f"Predicted S3:    {pred_s3[:20].tolist()}...")

if __name__ == "__main__":
    main()