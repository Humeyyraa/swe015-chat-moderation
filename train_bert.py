"""Model 3 (Area C): fine-tune BERTurk (dbmdz/bert-base-turkish-cased) with a
2-class classification head on [CLS]. Needs a GPU (Colab T4: ~3-4 min / epoch).

Run:  python scripts/train_bert.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.config import BERT_BASE, BERT_FP32_DIR, MAX_LEN, SEED
from src.data import load_split
from src.metrics import best_threshold, macro_f1, report
from src.normalize import normalize_for_bert

EPOCHS, BATCH, LR, WARMUP = 3, 32, 2e-5, 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batches(tok, texts, labels, shuffle):
    data = list(zip(texts, labels))

    def collate(b):
        enc = tok([t for t, _ in b], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b])
        return enc

    return DataLoader(data, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    torch.manual_seed(SEED)
    if DEVICE == "cpu":
        print("WARNING: no GPU found - this will be very slow. In Colab: Runtime > Change runtime type > T4 GPU")
    train, val = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}
    ).to(DEVICE)

    tr = batches(tok, [normalize_for_bert(t) for t in train["text"]], train["label"].tolist(), True)
    va = batches(tok, [normalize_for_bert(t) for t in val["text"]], val["label"].tolist(), False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    best_f1 = -1.0
    for ep in range(EPOCHS):
        model.train()
        t0, running = time.time(), 0.0
        for step, b in enumerate(tr, 1):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(**b).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"[bert] epoch {ep + 1} step {step}/{len(tr)} loss={running / step:.4f}")
        p = predict_proba(model, va)
        th = best_threshold(val["label"], p)
        f1 = macro_f1(val["label"], (p >= th).astype(int))
        print(f"[bert] epoch {ep + 1}/{EPOCHS} done in {time.time() - t0:.0f}s  "
              f"val macro-F1={f1:.4f} (threshold {th:.2f})")
        if f1 > best_f1:
            best_f1 = f1
            BERT_FP32_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BERT_FP32_DIR)
            tok.save_pretrained(BERT_FP32_DIR)
            meta = {"base_model": BERT_BASE, "epoch": ep + 1, "threshold": th,
                    "max_len": MAX_LEN, "lr": LR, "batch_size": BATCH,
                    "val": report(val["label"], (p >= th).astype(int))}
            (BERT_FP32_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[bert] saved best model -> {BERT_FP32_DIR}")
    print(f"[bert] best validation macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
