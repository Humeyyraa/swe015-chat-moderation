"""Model 2 (Area A): own word2vec + feedforward network trained with SGD.

Run:  python scripts/train_w2v_ffn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.config import RESULTS_DIR, SEED, W2V_DIR
from src.data import load_split
from src.metrics import best_threshold, macro_f1, report
from src.normalize import normalize, tokenize
from src.w2v_ffn import FFN, Featurizer, nearest, train_word2vec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X).to(DEVICE)), -1)[:, 1].cpu().numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")

    # ---- 1. word2vec on the training messages only ----
    tokens = [tokenize(normalize(t)) for t in train["text"]]
    itos, vectors = train_word2vec(tokens, dim=100, epochs=5, device=DEVICE, seed=SEED)
    print(f"[word2vec] vocabulary = {len(itos):,} words, dim = {vectors.shape[1]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "w2v_neighbors.txt", "w", encoding="utf-8") as f:
        for w in ["salak", "aptal", "mac", "guzel", "galatasaray", "hoca", "kanka", "seni"]:
            line = f"{w}: " + ", ".join(f"{n} ({s:.2f})" for n, s in nearest(w, itos, vectors))
            print("[word2vec]", line)
            f.write(line + "\n")

    # ---- 2. features ----
    feat = Featurizer(itos, vectors)
    Xtr, Xva = feat.batch(train["text"]), feat.batch(val["text"])
    ytr = torch.tensor(train["label"].values)

    # ---- 3. feedforward network, SGD + backpropagation ----
    model = FFN(Xtr.shape[1]).to(DEVICE)
    n_pos = int(ytr.sum())
    class_w = torch.tensor([1.0, (len(ytr) - n_pos) / n_pos], dtype=torch.float, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(Xtr)
    best = (-1.0, None, 0.5, 0)
    epochs, bs = 40, 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr[idx].to(DEVICE)
            loss = loss_fn(model(xb), yb)   # forward pass
            opt.zero_grad()
            loss.backward()                 # backpropagation
            opt.step()                      # SGD update
            total += loss.item() * len(idx)
        p = predict_proba(model, Xva)
        th = best_threshold(val["label"], p)
        f1 = macro_f1(val["label"], (p >= th).astype(int))
        print(f"[ffn] epoch {ep + 1:2d}/{epochs} train loss={total / len(perm):.4f} "
              f"val macro-F1={f1:.4f} (threshold {th:.2f})")
        if f1 > best[0]:
            best = (f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, th, ep + 1)

    f1, state, th, ep = best
    model.load_state_dict(state)
    W2V_DIR.mkdir(parents=True, exist_ok=True)
    np.save(W2V_DIR / "vectors.npy", vectors)
    (W2V_DIR / "vocab.json").write_text(json.dumps(itos, ensure_ascii=False))
    torch.save(state, W2V_DIR / "ffn.pt")
    p = predict_proba(model, Xva)
    meta = {"in_dim": int(Xtr.shape[1]), "threshold": th, "best_epoch": ep,
            "val": report(val["label"], (p >= th).astype(int))}
    (W2V_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ffn] best epoch {ep}, validation macro-F1={f1:.4f} -> saved to {W2V_DIR}")


if __name__ == "__main__":
    main()
