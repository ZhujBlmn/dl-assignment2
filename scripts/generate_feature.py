#!/usr/bin/env python3
import argparse
import json
import time
import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from cosyvoice.cli.cosyvoice import CosyVoice
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """快速加载JSONL文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def fast_load_audio_batch(audio_paths, target_sr=16000, max_workers=4):
    """批量快速加载音频文件（使用多线程）"""
    def load_single_audio(path):
        try:
            # 使用librosa加载，通常比torchaudio快
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            return torch.from_numpy(audio).float(), path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, path
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_single_audio, path): path for path in audio_paths}
        for future in tqdm(as_completed(future_to_path), total=len(audio_paths), desc="Loading audio"):
            result, path = future.result()
            if result is not None:
                results[path] = result
    return results


def extract_whisper_encoder_feats_batch(batch_audio, model, processor, device, max_duration=30.0):
    """批量提取Whisper特征（显著提升速度）"""
    batch_results = {}
    
    # 过滤超长音频
    valid_audio = {}
    for path, waveform in batch_audio.items():
        num_seconds = waveform.numel() / 16000.0
        if num_seconds <= max_duration:
            valid_audio[path] = waveform
    
    if not valid_audio:
        return batch_results
    
    # 批量处理
    audio_paths = list(valid_audio.keys())
    waveforms = list(valid_audio.values())
    
    # 将音频数据转换为numpy数组
    audio_arrays = [waveform.cpu().numpy() for waveform in waveforms]
    
    # 批量处理（Whisper处理器支持批量处理）
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None
    
    with torch.no_grad():
        enc_out = model.model.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    
    hidden_states = enc_out.hidden_states
    mid_idx = len(hidden_states) // 2
    
    # 处理每个样本的结果
    for i, path in enumerate(audio_paths):
        # 根据attention_mask获取有效长度
        if attention_mask is not None:
            valid_length = attention_mask[i].sum().item()
            mid_layer = hidden_states[mid_idx][i, :valid_length].cpu()
            final_layer = hidden_states[-1][i, :valid_length].cpu()
        else:
            mid_layer = hidden_states[mid_idx][i].cpu().squeeze(0)
            final_layer = hidden_states[-1][i].cpu().squeeze(0)
        
        batch_results[path] = (mid_layer, final_layer)
    
    return batch_results


def extract_text_embeddings_batch(texts_with_paths, cosy, device, batch_size=32):
    """批量提取文本嵌入"""
    text_embeddings = {}
    
    # 分批处理文本
    for i in range(0, len(texts_with_paths), batch_size):
        batch_items = texts_with_paths[i:i + batch_size]
        batch_texts = [item[1] for item in batch_items]
        batch_paths = [item[0] for item in batch_items]
        
        # 批量提取文本token
        batch_tokens = []
        batch_token_lens = []
        
        for text in batch_texts:
            text_token, text_token_len = cosy.frontend._extract_text_token(text)
            batch_tokens.append(text_token)
            batch_token_lens.append(text_token_len)
        
        # 堆叠tokens
        max_len = max(token.size(1) for token in batch_tokens)
        padded_tokens = []
        
        for token in batch_tokens:
            pad_size = max_len - token.size(1)
            if pad_size > 0:
                padded_token = torch.cat([
                    token, 
                    torch.zeros((1, pad_size), dtype=token.dtype, device=token.device)
                ], dim=1)
            else:
                padded_token = token
            padded_tokens.append(padded_token)
        
        stacked_tokens = torch.cat(padded_tokens, dim=0).to(device)
        
        # 批量计算嵌入
        with torch.no_grad():
            batch_embeddings = cosy.model.llm.text_embedding(stacked_tokens)
        
        # 保存结果（去除padding）
        for j, (path, original_token) in enumerate(zip(batch_paths, batch_tokens)):
            original_len = original_token.size(1)
            text_embedding = batch_embeddings[j, :original_len].cpu()
            text_embeddings[path] = text_embedding
    
    return text_embeddings


def main(args):
    print("🚀 开始优化处理...")
    start_time = time.time()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 使用设备: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载模型
    print("🔄 加载模型中...")
    model_load_start = time.time()
    
    # 使用更小的Whisper模型以加速（可以改为base/small）
    whisper_model_name = "openai/whisper-base"  # 比large-v3快很多
    if args.fast_mode:
        whisper_model_name = "openai/whisper-small"  # 极速模式
    
    cosy = CosyVoice(args.model_dir)
    processor = AutoProcessor.from_pretrained(whisper_model_name)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name).to(device)
    whisper_model.eval()
    
    # 启用torch2.0编译优化（如果可用）
    if hasattr(torch, 'compile') and device == "cuda":
        print("⚡ 启用Torch编译优化...")
        whisper_model = torch.compile(whisper_model, mode="reduce-overhead")
    
    print(f"✅ 模型加载完成: {time.time() - model_load_start:.2f}s")
    
    # 加载数据
    data = load_jsonl(args.jsonl)
    print(f"📁 加载 {len(data)} 个样本")
    
    # 准备批量数据
    audio_paths = [item["audio_path"] for item in data]
    texts_with_paths = [(item["audio_path"], item["text"]) for item in data]
    
    # 阶段1: 批量加载音频（多线程）
    print("🎵 批量加载音频文件...")
    audio_loading_start = time.time()
    audio_data = fast_load_audio_batch(audio_paths, max_workers=args.num_workers)
    print(f"✅ 音频加载完成: {time.time() - audio_loading_start:.2f}s")
    
    # 阶段2: 批量处理文本嵌入
    print("📝 批量处理文本嵌入...")
    text_embedding_start = time.time()
    utt2text_emb = extract_text_embeddings_batch(
        texts_with_paths, cosy, device, batch_size=args.batch_size
    )
    print(f"✅ 文本嵌入完成: {time.time() - text_embedding_start:.2f}s")
    
    # 阶段3: 批量处理Whisper特征
    print("🎤 批量处理Whisper特征...")
    whisper_start = time.time()
    
    # 分批处理音频以避免OOM
    batch_size = min(args.batch_size, 8)  # Whisper批处理较小以避免内存溢出
    utt2whisper_mid = {}
    utt2whisper_final = {}
    
    audio_items = list(audio_data.items())
    for i in range(0, len(audio_items), batch_size):
        batch_items = audio_items[i:i + batch_size]
        batch_dict = dict(batch_items)
        
        batch_results = extract_whisper_encoder_feats_batch(
            batch_dict, whisper_model, processor, device, args.max_duration
        )
        
        for path, (mid_feat, final_feat) in batch_results.items():
            utt2whisper_mid[path] = mid_feat
            utt2whisper_final[path] = final_feat
    
    print(f"✅ Whisper特征提取完成: {time.time() - whisper_start:.2f}s")
    
    # 保存结果
    print("💾 保存结果...")
    torch.save(utt2text_emb, args.output_text)
    
    whisper_output = {
        "mid": utt2whisper_mid,
        "final": utt2whisper_final,
    }
    torch.save(whisper_output, args.output_whisper)
    
    total_time = time.time() - start_time
    print(f"🎉 处理完成!")
    print(f"⏱️  总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    print(f"📊 处理速度: {len(data)/total_time:.2f} 样本/秒")
    print(f"💾 文本嵌入保存至: {args.output_text}")
    print(f"💾 Whisper特征保存至: {args.output_whisper}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优化版特征提取脚本")
    parser.add_argument("--jsonl", type=str, required=True, help="Input jsonl with audio_path and text")
    parser.add_argument("--model_dir", type=str, required=True, help="CosyVoice1 model dir")
    parser.add_argument("--output_text", type=str, required=True, help="Output .pt for CosyVoice text embeddings")
    parser.add_argument("--output_whisper", type=str, required=True, help="Output .pt for Whisper features")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio length (seconds) to process")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for audio loading")
    parser.add_argument("--fast_mode", action="store_true", help="Use smaller models for maximum speed")
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(args.output_text) if os.path.dirname(args.output_text) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_whisper) if os.path.dirname(args.output_whisper) else ".", exist_ok=True)
    
    main(args)