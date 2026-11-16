"""Model 4 (Area D): export the fine-tuned BERTurk to ONNX and apply dynamic
INT8 quantization with ONNX Runtime.

Dynamic quantization: the weights of every Linear layer (MatMul) are stored as
8-bit integers with a scale factor; activations are quantized on the fly at
inference time. No calibration data and no retraining are needed.

Also exports an FP32 ONNX model, so the results table can separate the effect
of the runtime (PyTorch -> ONNX Runtime) from the effect of quantization
(FP32 -> INT8).

Run:  python scripts/quantize_bert.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BERT_FP32_DIR, ONNX_DIR, ONNX_FP32, ONNX_INT8
from src.data import load_split
from src.metrics import best_threshold, macro_f1


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits


def export_onnx():
    tok = AutoTokenizer.from_pretrained(BERT_FP32_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_FP32_DIR, attn_implementation="eager").eval()
    enc = tok("örnek bir mesaj", return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    axes = {n: {0: "batch", 1: "seq"} for n in ["input_ids", "attention_mask", "token_type_ids"]}
    axes["logits"] = {0: "batch"}
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=list(axes)[:3], output_names=["logits"],
                  dynamic_axes=axes, opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(LogitsOnly(model), args, str(ONNX_FP32), dynamo=False, **kwargs)
    except TypeError:          # older PyTorch without the `dynamo` argument
        torch.onnx.export(LogitsOnly(model), args, str(ONNX_FP32), **kwargs)
    tok.save_pretrained(ONNX_DIR)
    with torch.no_grad():
        ref = model(**enc).logits.numpy()
    return ref


def main():
    ref = export_onnx()
    print(f"[onnx] FP32 model: {ONNX_FP32} ({ONNX_FP32.stat().st_size / 2**20:.1f} MB)")

    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QInt8)
    print(f"[onnx] INT8 model: {ONNX_INT8} ({ONNX_INT8.stat().st_size / 2**20:.1f} MB)")

    # Write a provisional meta.json so the predictors can load, then tune the
    # decision threshold of each ONNX model on the VALIDATION set.
    bert_meta = json.loads((BERT_FP32_DIR / "meta.json").read_text())
    meta = {"threshold_fp32": bert_meta["threshold"], "threshold_int8": bert_meta["threshold"],
            "quantization": "onnxruntime.quantization.quantize_dynamic, weights QInt8"}
    (ONNX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    from src.predictors import OnnxBertPredictor
    fp32, int8 = OnnxBertPredictor(int8=False), OnnxBertPredictor(int8=True)

    # sanity check: ONNX FP32 must give the same logits as PyTorch
    enc = fp32.tok("örnek bir mesaj", return_tensors="np")
    out = fp32.sess.run(None, {k: v.astype(np.int64) for k, v in enc.items() if k in fp32.inputs})[0]
    print(f"[onnx] max |PyTorch - ONNX FP32| logit difference = {np.abs(out - ref).max():.2e}")

    val = load_split("val")
    for key, pred in [("fp32", fp32), ("int8", int8)]:
        p = np.array([pred.predict_proba(t) for t in val["text"]])
        th = best_threshold(val["label"], p)
        meta[f"threshold_{key}"] = th
        meta[f"val_macro_f1_{key}"] = macro_f1(val["label"], (p >= th).astype(int))
        print(f"[onnx] {key}: validation macro-F1={meta[f'val_macro_f1_{key}']:.4f} (threshold {th:.2f})")
    (ONNX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[onnx] done -> {ONNX_DIR}")


if __name__ == "__main__":
    main()
