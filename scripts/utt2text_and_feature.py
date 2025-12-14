#!/usr/bin/env python3
import argparse
import json
import os
import gc  # 引入垃圾回收
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
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


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    return audio.squeeze(0), target_sr  # (T,), sr


def extract_whisper_encoder_feats(waveform, model, processor, device, max_duration=30.0):
    # waveform: 1D torch tensor at 16k
    num_seconds = waveform.numel() / 16000.0
    if num_seconds > max_duration:
        return None, None

    audio_np = waveform.numpy()
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        enc_out = model.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
        )

    hidden_states = enc_out.hidden_states  # list: [layer0, layer1, ..., last]
    mid_idx = len(hidden_states) // 2
    
    # 【优化2】转为 half (float16) 节省显存和内存
    mid_layer = hidden_states[mid_idx].detach().cpu().half().squeeze(0)   # (T, D)
    final_layer = hidden_states[-1].detach().cpu().half().squeeze(0)      # (T, D)
    
    return mid_layer, final_layer


def main(args):
    # load CosyVoice for text embedding
    cosy = CosyVoice(args.model_dir)
    emb_layer = cosy.model.llm.text_embedding

    # load Whisper encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper Base to {device}...")
    
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-base"
    ).to(device)
    whisper_model.eval()

    # 1. 加载所有数据
    all_data = load_jsonl(args.jsonl)
    total_len = len(all_data)
    
    # 【优化1】只取前 1000
    subset_size = int(1000)
    if subset_size == 0: subset_size = 1 # 防止数据太少报错
    data = all_data[:subset_size]
    
    print(f"⚠️ PROCESSING SUBSET: {len(data)} items (10% of {total_len})")

    utt2text_emb = {}
    utt2whisper_mid = {}
    utt2whisper_final = {}

    for i, item in enumerate(tqdm(data)):
        audio_path = item["audio_path"]
        text = item["text"]

        try:
            # ----- CosyVoice text embedding -----
            text_token, text_token_len = cosy.frontend._extract_text_token(text)  # [1, L], [1]
            with torch.no_grad():
                text_token = text_token.to(emb_layer.weight.device).long()
                # 【优化2】转 float16
                text_emb = emb_layer(text_token).squeeze(0).cpu().half()  # [L, D]
            utt2text_emb[audio_path] = text_emb

            # ----- Whisper encoder features -----
            waveform, _ = load_audio(audio_path, target_sr=16000)
            mid_feat, final_feat = extract_whisper_encoder_feats(
                waveform, whisper_model, processor, device, max_duration=args.max_duration
            )
            
            if mid_feat is not None:
                utt2whisper_mid[audio_path] = mid_feat
                utt2whisper_final[audio_path] = final_feat
            
            # 【优化3】手动垃圾回收，防止内存堆积
            if i % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # 检查输出目录
    os.makedirs(os.path.dirname(args.output_text), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_whisper), exist_ok=True)

    # save outputs
    torch.save(utt2text_emb, args.output_text)
    print(f"Saved CosyVoice text embeddings for {len(utt2text_emb)} items to {args.output_text}")

    whisper_output = {
        "mid": utt2whisper_mid,
        "final": utt2whisper_final,
    }
    torch.save(whisper_output, args.output_whisper)
    print(f"Saved Whisper features for {len(utt2whisper_mid)} items to {args.output_whisper}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Input jsonl with audio_path and text")
    parser.add_argument("--model_dir", type=str, required=True, help="CosyVoice1 model dir")
    parser.add_argument("--output_text", type=str, required=True, help="Output .pt for CosyVoice text embeddings")
    parser.add_argument("--output_whisper", type=str, required=True, help="Output .pt for Whisper features")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio length (seconds) to process")
    main(parser.parse_args())