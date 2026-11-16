"""Evaluate every model on the SAME held-out test set (official OffensEval-TR
test set, 3,528 messages) and build the results table (R4).

* task metric: macro-F1 (plus accuracy and per-class F1)
* size: weights file on disk (MB)
* latency: wall-clock time per message on CPU, batch size 1, end to end
  (normalization + tokenization + model), measured for every test message
  (3,528 runs) after 10 warm-up runs; mean / median / p95 reported.

Run:  python scripts/benchmark.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only, like the target server

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import LATENCY_BUDGET_MS, RESULTS_DIR
from src.data import load_split
from src.metrics import report
from src.predictors import load_all


def hardware() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {"cpu": cpu, "logical_cores": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version()}
    try:
        import torch
        info["torch"], info["torch_threads"] = torch.__version__, torch.get_num_threads()
    except ImportError:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    return info


def main():
    test = load_split("test")
    texts, y = test["text"].tolist(), test["label"].values
    models = load_all(include_onnx_fp32=True, include_torch_fp32=True)
    hw = hardware()
    print("Hardware:", hw)

    rows, preds = [], {"id": test["id"], "label": y}
    for name, m in models.items():
        for t in texts[:10]:                        # warm-up
            m.predict_proba(t)
        probs, times = [], []
        for t in texts:
            t0 = time.perf_counter()
            probs.append(m.predict_proba(t))
            times.append((time.perf_counter() - t0) * 1000)
        probs, times = np.array(probs), np.array(times)
        yhat = (probs >= m.threshold).astype(int)
        r = report(y, yhat)
        row = {"model": name, "technique": m.technique, "precision": m.precision,
               "macro_f1": round(r["macro_f1"], 4), "accuracy": round(r["accuracy"], 4),
               "f1_OFF": round(r["f1_OFF"], 4), "size_mb": round(m.size_mb(), 2),
               "latency_mean_ms": round(float(times.mean()), 2),
               "latency_p50_ms": round(float(np.median(times)), 2),
               "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
               "runs": len(times), "threshold": m.threshold,
               "within_budget": bool(times.mean() <= LATENCY_BUDGET_MS),
               "confusion_matrix": r["confusion_matrix"]}
        rows.append(row)
        preds[name] = probs.round(4)
        print(f"{name:24s} macro-F1={row['macro_f1']:.4f}  size={row['size_mb']:8.2f} MB  "
              f"latency={row['latency_mean_ms']:7.2f} ms (p95 {row['latency_p95_ms']:.2f})")

    # Agreement checks: PyTorch vs ONNX (runtime only) and FP32 vs INT8 (quantization only)
    agreement = []
    for a, b in [("BERTurk (PyTorch)", "BERTurk FP32 (ONNX)"),
                 ("BERTurk FP32 (ONNX)", "BERTurk INT8 (ONNX)")]:
        if a in preds and b in preds:
            pa, pb = np.asarray(preds[a]), np.asarray(preds[b])
            la = pa >= models[a].threshold
            lb = pb >= models[b].threshold
            d = np.abs(pa - pb)
            agreement.append({"a": a, "b": b, "same_label_pct": round(float((la == lb).mean() * 100), 2),
                              "mean_abs_prob_diff": round(float(d.mean()), 4),
                              "max_abs_prob_diff": round(float(d.max()), 4)})
            print(f"[agreement] {a} vs {b}: same label on {agreement[-1]['same_label_pct']}% of test messages, "
                  f"mean |dP|={d.mean():.4f}, max |dP|={d.max():.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(
        {"hardware": hw, "test_size": len(test), "latency_budget_ms": LATENCY_BUDGET_MS,
         "models": rows, "agreement": agreement}, indent=2, ensure_ascii=False))

    df = pd.DataFrame(rows)
    lines = [
        f"# Results on the OffensEval-TR 2020 test set (n = {len(test)})", "",
        f"Hardware: {hw['cpu']}, {hw['logical_cores']} logical cores, CPU only. "
        f"Latency = mean over {len(test)} single-message runs (batch size 1, after 10 warm-up runs), "
        f"end to end. Budget: {LATENCY_BUDGET_MS:.0f} ms / message.", "",
        "| Model | Precision | Macro-F1 | Accuracy | F1 (OFF) | Size (MB) | Latency mean (ms) | p95 (ms) | ≤ budget |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
                     f"{r['f1_OFF']:.4f} | {r['size_mb']:.2f} | {r['latency_mean_ms']:.2f} | "
                     f"{r['latency_p95_ms']:.2f} | {'yes' if r['within_budget'] else 'no'} |")
    if agreement:
        lines += ["", "| Comparison | Same label (%) | Mean abs. P diff | Max abs. P diff |", "|---|---|---|---|"]
        for g in agreement:
            lines.append(f"| {g['a']} vs {g['b']} | {g['same_label_pct']} | "
                         f"{g['mean_abs_prob_diff']} | {g['max_abs_prob_diff']} |")
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # per-message predictions + failure examples for the slides
    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    main_model = "BERTurk INT8 (ONNX)"
    if main_model in models:
        th = models[main_model].threshold
        wrong = test.assign(p_off=pred_df[main_model]).loc[(pred_df[main_model] >= th).astype(int) != y]
        wrong = wrong.assign(error=np.where(wrong["label"] == 1, "missed offensive (FN)",
                                            "false alarm (FP)"))
        cols = ["id", "text", "label", "error", "p_off"]
        others = [c for c in models if c != main_model]
        wrong = wrong.merge(pred_df[["id"] + others], on="id")
        wrong[cols + others].head(40).to_csv(RESULTS_DIR / "failure_examples.csv", index=False)
        print(f"{len(wrong)} test errors for {main_model}; 40 saved to results/failure_examples.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].barh(df["model"], df["macro_f1"], color="#4C72B0")
        ax[0].set_xlabel("macro-F1 (test)")
        ax[0].set_xlim(0, 1)
        ax[1].barh(df["model"], df["latency_mean_ms"], color="#DD8452")
        ax[1].axvline(LATENCY_BUDGET_MS, color="red", ls="--", label=f"budget {LATENCY_BUDGET_MS:.0f} ms")
        ax[1].set_xlabel("latency per message, CPU (ms)")
        ax[1].legend()
        ax[1].set_yticklabels([])
        for a in ax:
            a.invert_yaxis()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "results_chart.png", dpi=160)
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
