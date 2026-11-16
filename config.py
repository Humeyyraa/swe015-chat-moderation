"""Central configuration: paths, model names, Hugging Face repo, latency budget."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LEXICON_FILE = ROOT / "data" / "lexicon_tr.txt"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Model sub-folders (created by the training scripts, uploaded to the HF Hub)
TFIDF_DIR = MODELS_DIR / "tfidf_lr"
W2V_DIR = MODELS_DIR / "w2v_ffn"
BERT_FP32_DIR = MODELS_DIR / "berturk_fp32"
ONNX_DIR = MODELS_DIR / "berturk_onnx"
ONNX_FP32 = ONNX_DIR / "model_fp32.onnx"
ONNX_INT8 = ONNX_DIR / "model_int8.onnx"

# Dataset: OffensEval-TR 2020 (Çöltekin, 2020), CC-BY 2.0
DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"

# Pretrained Transformer that we fine-tune
BERT_BASE = "dbmdz/bert-base-turkish-cased"   # BERTurk
MAX_LEN = 128

# Where the trained weights live (weights are NOT stored in the git repo).
# Override with:  export HF_REPO_ID=<your-hf-username>/swe015-chat-moderation
HF_REPO_ID = os.environ.get("HF_REPO_ID", "humeyrakoctemiz/swe015-chat-moderation")

# Latency budget per message on CPU (milliseconds)
LATENCY_BUDGET_MS = 50.0

SEED = 42
LABELS = ["NOT", "OFF"]  # 0 = clean, 1 = offensive
