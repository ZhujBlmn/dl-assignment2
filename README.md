# Assignment 2: Text-Aligned Speech Tokenization (TASTE) Implementation

## 1. Project Overview
This project implements a simplified TASTE model. It leverages a cross-attention aggregator to align speech features (from Whisper) with text embeddings (from CosyVoice). The aligned representations are used to predict discrete S3 speech tokens.

**Key Results:**
- **Test Accuracy (Top-1):** ~17% (Evaluated on `test-clean` with unseen speakers).
- **Training Data:** 1,000 samples randomly selected from `train-clean-100`.

---

## 2. Environment Setup

1. **Install Dependencies:**
   Ensure you have Python 3.8+ and PyTorch installed.
```bash
pip install -r requirements.txt

```

2. **Model Preparation:**
Download the **CosyVoice-300M** model and place it in the `models/` directory:
* Path: `models/CosyVoice-300M`



---

## 3. Data Preparation PipelinePlease follow the steps below to reproduce the data processing, training, and evaluation.

### Step 1: Prepare Training Data (train-clean-100)**1.1 Generate JSONL Index:**
Scan the raw LibriSpeech data and generate a file list.

```bash
python scripts/jsonl_prepare.py \
    --librispeech_dir origin_data/LibriSpeech/train-clean-100 \
    --output_jsonl data/librispeech.json

```

**1.2 Extract Features (Text & Whisper):**
Extract text embeddings and Whisper encoder features.

```bash
python scripts/utt2text_and_feature.py \
    --jsonl data/librispeech.json \
    --model_dir models/CosyVoice-300M \
    --output_text features/utt2text.pt \
    --output_whisper features/utt2whisper.pt

```

**1.3 Extract S3 Tokens (Ground Truth):**
Extract discrete speech tokens for training targets.

```bash
# Note: Please verify the arguments for your run_s3.py script
python scripts/run_s3.py

```

---

### Step 2: Prepare Test Data (test-clean)To evaluate on the test set, we process `test-clean` using the same pipeline but save to different filenames.

**2.1 Generate Test JSONL:**

```bash
python scripts/jsonl_prepare.py \
    --librispeech_dir origin_data/LibriSpeech/test-clean \
    --output_jsonl data/librispeech_test.json

```

**2.2 Extract Test Features:**

```bash
python scripts/utt2text_and_feature.py \
    --jsonl data/librispeech_test.json \
    --model_dir models/CosyVoice-300M \
    --output_text features/test_utt2text.pt \
    --output_whisper features/test_utt2whisper.pt

```

**2.3 Extract Test S3 Tokens:**

```bash
python scripts/run_s3.py \

```

---

## 4. TrainingTrain the alignment aggregator. The script defaults to `BATCH_SIZE=1` for memory efficiency.

```bash
python scripts/train.py

```

* **Checkpoints:** Saved to `checkpoints/` (e.g., `epoch_10_acc_0.17xx.pt`).
* **Logs:** Training progress is saved to `training_log.json`.

---

## 5. EvaluationRun the test script to calculate the Top-1 Accuracy on the `test-clean` dataset.

**Important:** Before running, please open `scripts/test.py` and modify the `CKPT_PATH` variable to point to your best checkpoint file.

```bash
python scripts/test.py

```