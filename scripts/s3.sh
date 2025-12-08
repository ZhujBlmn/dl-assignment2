#!/usr/bin/env bash
LIBRISPEECH_DIR="/kaggle/input/origin-data/train-clean-100/LibriSpeech"
OUT_DIR="features"
ONNX_PATH="/root/.cache/modelscope/hub/models/iic/CosyVoice-300M/speech_tokenizer_v1.onnx" # speech_tokenizer_v1.onnx under CosyVoice-300M model dir

mkdir -p "$OUT_DIR"
find "$LIBRISPEECH_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" \) | sort | while read -r f; do
  rel="${f#"$LIBRISPEECH_DIR"/}"
  id="${rel%.*}"; id="${id//\//-}"
  echo "$id $f"
done > "$OUT_DIR/wav.scp"

python3 "/kaggle/working/dl-assignment2/CosyVoice/tools/extract_speech_token_mp.py" \
  --dir "$OUT_DIR" \
  --onnx_path "$ONNX_PATH"