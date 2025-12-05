#!/usr/bin/env python3
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

UTT2_S3_PATH = "YOUR_UTT2_S3_PATH"
UTT2_TEXT_EMB_PATH = "YOUR_UTT2_TEXT_EMB_PATH"
UTT2_WHISPER_PATH = "YOUR_UTT2_WHISPER_PATH"
COSYVOICE_MODEL_DIR = r"D:\EduKillers\25下\深度学习\Assignment2\models\CosyVoice-300M"

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
    cosy = CosyVoice(COSYVOICE_MODEL_DIR)
    return cosy.model.llm.llm


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
        # Implement three Linear layers:
        #   self.q_proj: text_dim        -> hidden_dim
        #   self.k_proj: speech_last_dim -> hidden_dim
        #   self.v_proj: speech_mid_dim  -> hidden_dim
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim)
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        # text_emb: (B, T_text, D_text)
        # speech_last: (B, T_speech, D_last)
        # speech_mid: (B, T_speech, D_mid)
        # speech_mask: (B, T_speech) bool (True = valid)

        # 1) Project inputs:
        Q = self.q_proj(text_emb)       # (B, T_text, H)
        K = self.k_proj(speech_last)    # (B, T_speech, H)
        V = self.v_proj(speech_mid)     # (B, T_speech, H)

        # 2) Compute attention scores
        # scores: (B, T_text, T_speech)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(max(1.0, d_k))

        # 3) Apply mask if given (True=valid)
        if speech_mask is not None:
            # speech_mask: (B, T_speech) -> expand to (B, 1, T_speech)
            mask = speech_mask.unsqueeze(1)  # (B,1,T_speech)
            scores = scores.masked_fill(~mask, -1e9)

        # 4) Softmax over last dim
        att = F.softmax(scores, dim=-1)  # (B, T_text, T_speech)

        # 5) Attended representation
        z = torch.matmul(att, V)  # (B, T_text, H)

        # 6) Return
        return z, att


class CosyVoiceS3Model(nn.Module):
    """
    CosyVoice LLM + aggregator

    Inputs:
        text_emb    : (B, T_text, D_text)
        speech_last : (B, T_speech, D_last)
        speech_mid  : (B, T_speech, D_mid)
        speech_mask : (B, T_speech) bool
        s3_targets  : (B, T_s3) long

    Outputs:
        loss        : scalar 
        logits      : (B, T_text+prefix+T_s3, S3_VOCAB_SIZE+1)
        attn        : (B, T_text, T_speech)
    """
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
        self.s3_vocab_size_with_eos = s3_vocab_size + 1  # extra EOS like STAGE1_TRAIN
        self.input_proj = nn.Linear(text_dim, self.llm.output_size())
        self.proj = nn.Linear(self.llm.output_size(), self.s3_vocab_size_with_eos)
        # prefix embeddings to mimic [SOS/EOS] and [TASK] in STAGE1_TRAIN
        self.llm_embedding = nn.Embedding(2, self.llm.output_size())  # 0: sos_eos, 1: task_id
        # speech token embedding for teacher forcing (V + 1 to include EOS index)
        self.speech_embedding = nn.Embedding(self.s3_vocab_size_with_eos, self.llm.output_size())

        # fusion: add normalization and learnable weight (minimal change)
        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

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
        #################
        # step 1: aggregation + fusion
        # 1) Call the aggregator:
        #       z, attn = ...
        # 2) Fuse text embeddings and aggregated speech:
        #       w = sigmoid(fuse_alpha)
        #       fused = w * ln_z(z) + (1 - w) * ln_text(text_emb)
        # Shapes:
        #   z, fused : (B, T_text, hidden_dim/text_dim)
        #   attn     : (B, T_text, T_speech)
        #################
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask=speech_mask)
        # layer-norm
        z_ln = self.ln_z(z)
        text_ln = self.ln_text(text_emb)
        w = torch.sigmoid(self.fuse_alpha)
        # ensure shapes match - usually hidden_dim == text_dim; if not, input_proj / adapter needed upstream
        fused = w * z_ln + (1.0 - w) * text_ln  # (B, T_text, D)

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

        #################
        # targets + loss:
        # Build teacher-forced target `lm_target` and compute cross-entropy loss:
        #   - L = lm_input.size(1)
        #   - Build `lm_target` of shape (B, L) initialized with IGNORE_ID
        #   - For each batch i:
        #       * prefix_len = 2 + text_lens[i]  # [SOS] + fused_len + [TASK]
        #       * If slen > 1, copy s3_targets[i, :slen-1] to
        #         lm_target[i, prefix_len : prefix_len + slen - 1]
        #       * Write EOS (value = s3_vocab_size) at lm_target[i, prefix_len + slen - 1]
        #   - Calculate the Cross-Entropy Loss:
        #################
        L = lm_input.size(1)
        lm_target = torch.full((B, L), IGNORE_ID, dtype=torch.long, device=device)

        for i in range(B):
            prefix_len = 2 + int(text_lens[i].item())  # 1 (sos) + text_len + 1 (task)
            slen = int(s3_lens[i].item())
            if slen <= 0:
                # still place an EOS? If no s3 tokens, we can place EOS right away
                # put EOS at position prefix_len (which is valid if prefix_len < L)
                if prefix_len < L:
                    lm_target[i, prefix_len] = self.s3_vocab_size
            else:
                # copy s3_targets up to slen-1
                if slen - 1 > 0:
                    lm_target[i, prefix_len: prefix_len + (slen - 1)] = s3_targets[i, : (slen - 1)].to(device)
                # EOS at the final position
                eos_pos = prefix_len + (slen - 1)
                if eos_pos < L:
                    lm_target[i, eos_pos] = self.s3_vocab_size

        # compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
        loss = loss_fct(logits.view(-1, self.s3_vocab_size_with_eos), lm_target.view(-1))

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
    utt2s3 = torch.load(UTT2_S3_PATH, map_location="cpu")
    utt2text = torch.load(UTT2_TEXT_EMB_PATH, map_location="cpu")
    utt2whisper = torch.load(UTT2_WHISPER_PATH, map_location="cpu")

    keys = sorted(set(utt2s3.keys()) & set(utt2text.keys()) & set(utt2whisper['mid'].keys()) & set(utt2whisper['final'].keys()))
    samples = []

    for key in keys:
        s3_tokens = utt2s3[key]                         # 1D long tensor or list
        text_emb = utt2text[key]                        # (T_text, D_text)
        speech_mid = utt2whisper['mid'][key]            # (T_speech, D_mid)
        speech_last = utt2whisper['final'][key]          # (T_speech, D_last)

        if (s3_tokens is None) or (text_emb is None) or (speech_mid is None) or (speech_last is None):
            continue
        if (getattr(text_emb, "numel", lambda: 0)() == 0) or (getattr(speech_mid, "numel", lambda: 0)() == 0) or (getattr(speech_last, "numel", lambda: 0)() == 0):
            continue

        # ensure s3_tokens is a 1D long tensor
        if not torch.is_tensor(s3_tokens):
            s3_tokens = torch.as_tensor(s3_tokens, dtype=torch.long)

        samples.append({
            "utt_id": key,
            "text_emb": text_emb,
            "speech_mid": speech_mid,
            "speech_last": speech_last,
            "s3_tokens": s3_tokens,
        })
    return samples


# ======================
#  Train / Eval / Predict
# ======================

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Implement one training epoch:
      1) model.train()
      2) Loop over batches from dataloader.
      3) Move all tensors in batch to `device`.
      4) Call the model and get loss
      5) Backprop, clip, step
      6) Count valid tokens where s3_targets != S3_PAD_ID
      7) Return average loss per token.
    """
    model.train()
    total_loss_tokens = 0.0
    total_tokens = 0

    for batch in dataloader:
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        loss, logits, _ = model(
            text_emb=batch["text_emb"],
            speech_last=batch["speech_last"],
            speech_mid=batch["speech_mid"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            s3_targets=batch["s3_targets"],
        )

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # count valid tokens (where != pad)
        num_tokens = int((batch["s3_targets"] != S3_PAD_ID).sum().item())
        if num_tokens > 0:
            total_loss_tokens += float(loss.item()) * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0
    return total_loss_tokens / total_tokens


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    """
    Implement evaluation loop:
      1) model.eval()
      2) Loop over batches (no backward).
      3) Move tensors and call model.
      4) Count valid tokens and accumulate loss*token.
      5) Return average loss per token.
    """
    model.eval()
    total_loss_tokens = 0.0
    total_tokens = 0

    for batch in dataloader:
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        loss, logits, _ = model(
            text_emb=batch["text_emb"],
            speech_last=batch["speech_last"],
            speech_mid=batch["speech_mid"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            s3_targets=batch["s3_targets"],
        )

        num_tokens = int((batch["s3_targets"] != S3_PAD_ID).sum().item())
        if num_tokens > 0:
            total_loss_tokens += float(loss.item()) * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0
    return total_loss_tokens / total_tokens


@torch.no_grad()
def predict_s3(model, text_emb, speech_last, speech_mid, device, max_steps_factor=4):
    """
    Autoregressive decoding:

    text_emb    : (T_text, D_text)
    speech_last : (T_speech, D_last)
    speech_mid  : (T_speech, D_mid)

    Return:
        pred_s3 : (L,) long (until EOS or max_steps)
    """
    model.eval()

    # 1) Add batch dim and move to device
    text_emb_b = text_emb.unsqueeze(0).to(device)        # (1, T_text, D_text)
    speech_last_b = speech_last.unsqueeze(0).to(device)  # (1, T_speech, D_last)
    speech_mid_b = speech_mid.unsqueeze(0).to(device)    # (1, T_speech, D_mid)

    # 2) Full-True speech mask (no padding at inference)
    speech_mask = torch.ones((1, speech_mid_b.size(1)), dtype=torch.bool, device=device)

    # 3) Aggregator + fusion -> fused_llm
    z, _ = model.aggregator(text_emb_b, speech_last_b, speech_mid_b, speech_mask=speech_mask)
    z_ln = model.ln_z(z)
    text_ln = model.ln_text(text_emb_b)
    w = torch.sigmoid(model.fuse_alpha)
    fused = w * z_ln + (1.0 - w) * text_ln  # (1, T_text, D)
    fused_llm = model.input_proj(fused)      # (1, T_text, D_llm)

    B = 1
    text_len = fused_llm.size(1)
    device = fused_llm.device

    # prefixes
    sos_eos_emb = model.llm_embedding.weight[0].reshape(1, 1, -1).expand(B, 1, -1)
    task_id_emb = model.llm_embedding.weight[1].reshape(1, 1, -1).expand(B, 1, -1)

    # 4) initial sequence: [sos_eos, fused_llm, task_id]
    seq = torch.cat([sos_eos_emb, fused_llm, task_id_emb], dim=1)  # (1, L0, D)
    seq_len = torch.tensor([seq.size(1)], dtype=torch.int32, device=device)

    # autoregressive decode
    generated = []
    max_steps = max_steps_factor * max(1, text_len)

    for step in range(max_steps):
        # run llm on current seq
        hidden, _ = model.llm(seq, seq_len)  # (1, L, H)
        logits = model.proj(hidden)          # (1, L, V+1)
        last_logits = logits[:, -1, :]       # (1, V+1)
        next_id = int(torch.argmax(last_logits, dim=-1).item())

        # if EOS, stop
        if next_id == model.s3_vocab_size:
            break

        # clamp and append
        next_id_clamped = max(0, min(next_id, model.s3_vocab_size - 1))
        generated.append(next_id_clamped)

        # embed and append to seq
        next_embed = model.speech_embedding(torch.tensor([[next_id_clamped]], device=device))  # (1,1,D)
        seq = torch.cat([seq, next_embed], dim=1)
        seq_len = torch.tensor([seq.size(1)], dtype=torch.int32, device=device)

    if len(generated) == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(generated, dtype=torch.long, device="cpu")


# ======================
#  Main
# ======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples()
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
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

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
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = eval_one_epoch(model, valid_loader, device)
        print(f"Epoch {epoch:02d} | train_loss/token = {train_loss:.4f} | valid_loss/token = {valid_loss:.4f}")

    # Example usage of predict_s3
    if len(valid_samples) > 0:
        ex = valid_samples[0]
        pred_s3 = predict_s3(
            model,
            ex["text_emb"],
            ex["speech_last"],
            ex["speech_mid"],
            device,
        )
        print("Example predicted S3 len:", pred_s3.numel())

if __name__ == "__main__":
    main()
