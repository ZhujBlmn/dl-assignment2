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
COSYVOICE_MODEL_DIR = "YOUR_COSYVOICE_MODEL_DIR"

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
        #################
        # TODO (init):
        # Implement three Linear layers:
        #   self.q_proj: text_dim        -> hidden_dim
        #   self.k_proj: speech_last_dim -> hidden_dim
        #   self.v_proj: speech_mid_dim  -> hidden_dim
        #
        # Hint: use nn.Linear(in_features, out_features).
        #################
        # your code here
        #################
        pass

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        #################
        # TODO (forward):
        # Implement scaled dot-product attention from text to speech:
        #
        # 1) Project inputs:
        #       Q = ...
        #       K = ...
        #       V = ...
        #
        # 2) Compute attention scores
        #
        # 3) If speech_mask is given (True = valid, False = pad),
        #    mask out padded positions in `scores` with a large negative value
        #    before softmax.
        #
        # 4) Apply softmax over the last dimension to get attention weights:
        #       att = ...
        #
        # 5) Compute attended representation:
        #       z = ...
        #
        # 6) Return (z, att).
        #################
        # your code here
        #################
        pass


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
        logits      : (B, T_text, S3_VOCAB_SIZE)
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
        # TODO (step 1: aggregation + fusion):
        #
        # 1) Call the aggregator:
        #       z, attn = ...
        #
        # 2) Fuse text embeddings and aggregated speech:
        #       w = ...
        #       fused = ...
        #
        # Shapes:
        #   z, fused : ...
        #   attn     : ...
        #################
        # your code here (z, attn, fused)
        #################

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
        # TODO (targets + loss):
        #
        # Build teacher-forced target `lm_target` and compute cross-entropy loss:
        #   - L = lm_input.size(1)
        #   - Build `lm_target` of shape (B, L) initialized with IGNORE_ID
        #   - For each batch i:
        #       * prefix_len = 2 + text_lens[i]  # [SOS] + fused_len + [TASK]
        #       * If slen > 1, copy s3_targets[i, :slen-1] to
        #         lm_target[i, prefix_len : prefix_len + slen - 1]
        #       * Write EOS (value = s3_vocab_size) at lm_target[i, prefix_len + slen - 1]
        #   - Calculate the Cross-Entropy Loss:
        #
        # your code here (lm_target, loss)
        #################
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
        s3_tokens = utt2s3[key]                         # 1D long tensor
        text_emb = utt2text[key]                        # (T_text, D_text)
        speech_mid = utt2whisper['mid'][key]            # (T_speech, D_mid)
        speech_last = utt2whisper['final'][key]          # (T_speech, D_last)

        if (s3_tokens is None) or (text_emb is None) or (speech_mid is None) or (speech_last is None):
            continue
        if (getattr(text_emb, "numel", lambda: 0)() == 0) or (getattr(speech_mid, "numel", lambda: 0)() == 0) or (getattr(speech_last, "numel", lambda: 0)() == 0):
            continue

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
    #################
    # TODO (train_one_epoch):
    #
    # Implement one training epoch:
    #   1) model.train()
    #   2) Loop over batches from dataloader.
    #   3) Move all tensors in batch to `device`.
    #   4) Call the model:
    #        loss, logits, _ = model(
    #            text_emb=...,
    #            speech_last=...,
    #            speech_mid=...,
    #            speech_mask=...,
    #            text_mask=...,
    #            s3_targets=...,
    #        )
    #   5) Backprop:
    #        optimizer.zero_grad()
    #        loss.backward()
    #        (optional) clip gradients with nn.utils.clip_grad_norm_
    #        optimizer.step()
    #   6) Count valid tokens where s3_targets != S3_PAD_ID
    #      (you can ignore the first position to match causal shift),
    #      and accumulate total loss * num_tokens.
    #   7) Return average loss per token.
    #################
    # your code here
    #################
    pass


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    #################
    # TODO (eval_one_epoch):
    #
    # Implement evaluation loop:
    #   1) model.eval()
    #   2) Loop over batches (no backward, no optimizer step).
    #   3) Move tensors to `device` and call model (same as training).
    #   4) Count valid tokens (s3_targets != S3_PAD_ID, optionally skip first).
    #   5) Accumulate total loss * num_tokens.
    #   6) Return average validation loss per token.
    #################
    # your code here
    #################
    pass


@torch.no_grad()
def predict_s3(model, text_emb, speech_last, speech_mid, device):
    """
    text_emb    : (T_text, D_text)
    speech_last : (T_speech, D_last)
    speech_mid  : (T_speech, D_mid)

    Return:
        pred_s3 : (L,) long (until EOS or max_steps)
    """
    #################
    # TODO (predict_s3):
    #
    # Implement autoregressive decoding:
    #
    # 1) Add batch dimension and move inputs to `device`:
    #       text_emb    -> ...
    #       speech_last -> ...
    #       speech_mid  -> ...
    #
    # 2) Create a full-True speech_mask since there is no padding at inference.
    #
    # 3) Reuse the aggregator + fusion:
    #       z, _ = ...
    #       w = ...
    #       fused = ...
    #       fused_llm = ...
    #
    # 4) Build initial sequence:
    #       seq = ...
    #    using model.llm_embedding.weight[0] and [1].
    #
    # 5) For each decoding step:
    #       - Run LLM on current seq: hidden, _ = ...
    #       - Project to logits: logits = ...
    #       - Take last-step logits and choose argmax as next_id.
    #       - If next_id == EOS (...), stop.
    #       - Otherwise:
    #           * clamp next_id into [0, ...-1]
    #           * embed with ...
    #           * append to seq, update seq_len
    #       - Stop when steps reach some max_steps (e.g. 4 * T_text).
    #
    # 6) Collect all generated ids into a 1D LongTensor on CPU and return.
    #################
    # your code here
    #################
    pass


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

if __name__ == "__main__":
    main()
