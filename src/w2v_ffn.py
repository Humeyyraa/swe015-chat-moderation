"""Area A (+ B embeddings) - our OWN word2vec and a feedforward classifier.

word2vec: skip-gram with negative sampling (Mikolov et al., 2013), written from
scratch in PyTorch and trained only on the training messages.
    loss = -log σ(v_c · u_o) - Σ_k log σ(-v_c · u_k)       (k = negative samples)

Classifier: message vector = [mean ; max] of its word vectors  ->  feedforward
network  ->  2 classes. Trained with mini-batch SGD + backpropagation.
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .normalize import normalize, tokenize


# ------------------------------------------------------------------ word2vec
def build_vocab(token_lists, min_count=2):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = [w for w, c in counts.most_common() if c >= min_count]
    freqs = np.array([counts[w] for w in itos], dtype=np.float64)
    return itos, freqs


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim, sparse=True)   # v (the vectors we keep)
        self.context = nn.Embedding(vocab_size, dim, sparse=True)  # u
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, c, o, neg):
        v = self.center(c)                                    # (B, d)
        pos = (v * self.context(o)).sum(-1)                   # (B,)
        negs = torch.bmm(self.context(neg), v.unsqueeze(2)).squeeze(2)  # (B, K)
        return -(nn.functional.logsigmoid(pos) + nn.functional.logsigmoid(-negs).sum(-1)).mean()


def make_pairs(id_lists, freqs, window, rng, subsample=1e-4):
    """(center, context) pairs with dynamic window and frequent-word subsampling."""
    f = freqs / freqs.sum()
    keep_p = np.minimum(1.0, np.sqrt(subsample / f) + subsample / f)
    cs, os_ = [], []
    for ids in id_lists:
        ids = [i for i in ids if rng.random() < keep_p[i]]
        for pos, c in enumerate(ids):
            w = rng.integers(1, window + 1)
            for j in range(max(0, pos - w), min(len(ids), pos + w + 1)):
                if j != pos:
                    cs.append(c)
                    os_.append(ids[j])
    return np.array(cs, dtype=np.int64), np.array(os_, dtype=np.int64)


def train_word2vec(token_lists, dim=100, window=5, negatives=5, epochs=5,
                   batch_size=1024, lr=0.005, min_count=2, device="cpu", seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    itos, freqs = build_vocab(token_lists, min_count)
    stoi = {w: i for i, w in enumerate(itos)}
    id_lists = [[stoi[t] for t in toks if t in stoi] for toks in token_lists]

    noise = torch.tensor(freqs ** 0.75 / (freqs ** 0.75).sum(), dtype=torch.float, device=device)
    model = SkipGramNS(len(itos), dim).to(device)
    # SparseAdam: Adam for sparse embedding gradients (only the rows used in a batch are updated)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    for ep in range(epochs):
        c, o = make_pairs(id_lists, freqs, window, rng)
        perm = rng.permutation(len(c))
        c, o = torch.from_numpy(c[perm]), torch.from_numpy(o[perm])
        total, steps = 0.0, 0
        n_batches = (len(c) + batch_size - 1) // batch_size
        for b in range(n_batches):
            cb = c[b * batch_size:(b + 1) * batch_size].to(device)
            ob = o[b * batch_size:(b + 1) * batch_size].to(device)
            nb = torch.multinomial(noise, cb.numel() * negatives, replacement=True).view(-1, negatives)
            loss = model(cb, ob, nb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        print(f"[word2vec] epoch {ep + 1}/{epochs}  pairs={len(c):,}  loss={total / steps:.4f}")

    vectors = model.center.weight.detach().cpu().numpy().astype(np.float32)
    return itos, vectors


def nearest(word, itos, vectors, k=8):
    """Cosine-similarity neighbours of a word (for the slides / sanity check)."""
    stoi = {w: i for i, w in enumerate(itos)}
    if word not in stoi:
        return []
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sims = unit @ unit[stoi[word]]
    best = np.argsort(-sims)[1:k + 1]
    return [(itos[i], float(sims[i])) for i in best]


# ------------------------------------------------------------------ classifier
class FFN(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class Featurizer:
    """normalized text -> [mean ; max] of word2vec vectors (zeros if no known word)."""

    def __init__(self, itos, vectors, mean=None, std=None):
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.vectors = vectors
        self.dim = vectors.shape[1]
        self.mean, self.std = mean, std   # feature standardization, fitted on the training set

    def fit_scaler(self, X):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        return (X - self.mean) / self.std

    def __call__(self, text: str) -> np.ndarray:
        ids = [self.stoi[t] for t in tokenize(normalize(text)) if t in self.stoi]
        if not ids:
            x = np.zeros(2 * self.dim, dtype=np.float32)
        else:
            m = self.vectors[ids]
            x = np.concatenate([m.mean(0), m.max(0)])
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x.astype(np.float32)

    def raw_batch(self, texts) -> np.ndarray:
        mean, self.mean = self.mean, None
        X = np.stack([self(t) for t in texts])
        self.mean = mean
        return X

    def batch(self, texts) -> np.ndarray:
        return np.stack([self(t) for t in texts])
