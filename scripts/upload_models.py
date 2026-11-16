"""Upload ./models to the Hugging Face Hub.

Run:  export HF_REPO_ID=<hf-username>/swe015-chat-moderation
      huggingface-cli login        # or set HF_TOKEN
      python scripts/upload_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.hub import upload_models

CARD = """---
language: tr
license: cc-by-2.0
tags: [text-classification, offensive-language, turkish, onnx, quantization]
datasets: [coltekin/offenseval2020_tr]
---
# SWE015 - Real-time chat moderation (Turkish)

Weights for the homework repository. Trained on OffensEval-TR 2020 (Çöltekin, 2020).

| folder | model |
|---|---|
| `tfidf_lr/` | tf-idf + logistic regression |
| `w2v_ffn/` | own word2vec (skip-gram, negative sampling) + feedforward network |
| `berturk_fp32/` | fine-tuned `dbmdz/bert-base-turkish-cased`, FP32 |
| `berturk_onnx/` | the same model in ONNX Runtime, FP32 and dynamic INT8 |
"""

if __name__ == "__main__":
    (MODELS_DIR / "README.md").write_text(CARD, encoding="utf-8")
    upload_models()
