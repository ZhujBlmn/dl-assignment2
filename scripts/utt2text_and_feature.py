#!/usr/bin/env python3
import argparse
import json
import os
import gc
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from cosyvoice.cli.cosyvoice import CosyVoice

# --- huggingface_hub patch ---
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
            if not line: continue
            items.append(json.loads(line))
    return items

def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    return audio.squeeze(0), target_sr

def extract_whisper_encoder_feats(waveform, model, processor, device, max_duration=30.0):
    num_seconds = waveform.numel() / 16000.0
    if num_seconds >= max_duration:
        return None, None
    
    MAX_SAMPLES = int(16000 * max_duration)
    if waveform.numel() > MAX_SAMPLES:
        waveform = waveform[:MAX_SAMPLES]

    audio_np = waveform.numpy()
    inputs = processor(
        audio_np, sampling_rate=16000, return_tensors="pt",
        padding="max_length", truncation=True,
    )
    input_features = inputs.input_features.to(device)
    
    with torch.no_grad():
        enc_out = model.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
        )

    hidden_states = enc_out.hidden_states
    mid_idx = len(hidden_states) // 2
    # 关键：必须 detach().cpu() 否则显存爆炸
    mid_layer = hidden_states[mid_idx].detach().cpu().squeeze(0)
    final_layer = hidden_states[-1].detach().cpu().squeeze(0)
    return mid_layer, final_layer

def main(args):
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    cosy = CosyVoice(args.model_dir)
    emb_layer = cosy.model.llm.text_embedding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper to {device}...")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base").to(device)
    whisper_model.eval()

    data = load_jsonl(args.jsonl)
    print(f"Total items: {len(data)}")

    # 2. Chunk Processing
    CHUNK_SIZE = 1000  # 每1000条保存一个文件
    chunk_idx = 0
    
    # 临时存储当前 chunk 的数据
    current_chunk = {
        "text_emb": {},
        "whisper_mid": {},
        "whisper_final": {}
    }

    for i, item in enumerate(tqdm(data)):
        audio_path = item["audio_path"]
        text = item["text"]
        
        try:
            # A. CosyVoice Text
            text_token, _ = cosy.frontend._extract_text_token(text)
            with torch.no_grad():
                text_token = text_token.to(emb_layer.weight.device).long()
                text_emb = emb_layer(text_token).squeeze(0).cpu() # save as cpu
            
            # B. Whisper Speech
            waveform, _ = load_audio(audio_path)
            mid, final = extract_whisper_encoder_feats(
                waveform, whisper_model, processor, device, args.max_duration
            )
            
            if mid is not None:
                current_chunk["text_emb"][audio_path] = text_emb
                current_chunk["whisper_mid"][audio_path] = mid
                current_chunk["whisper_final"][audio_path] = final
            
        except Exception as e:
            print(f"Error on {audio_path}: {e}")
            continue

        # Save Chunk
        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(data):
            save_name = f"part_{chunk_idx:03d}.pt"
            save_path = os.path.join(args.output_dir, save_name)
            torch.save(current_chunk, save_path)
            print(f"Saved {save_path}")
            
            # Reset & GC
            current_chunk = {"text_emb": {}, "whisper_mid": {}, "whisper_final": {}}
            chunk_idx += 1
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    # 改为输出目录，而非单个文件
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save part_xx.pt")
    parser.add_argument("--max_duration", type=float, default=30.0)
    main(parser.parse_args())