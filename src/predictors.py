"""One common interface for every model, used by BOTH benchmark.py and app.py,
so the numbers in the results table come from exactly the code the app runs.

    p = predictor.predict_proba(text)    # P(offensive) for one raw message
    label = int(p >= predictor.threshold)
"""
import json
from pathlib import Path

import numpy as np

from .config import (BERT_FP32_DIR, LEXICON_FILE, MAX_LEN, ONNX_DIR, ONNX_FP32,
                     ONNX_INT8, TFIDF_DIR, W2V_DIR)
from .lexicon import Lexicon
from .normalize import normalize, normalize_for_bert


def _size_mb(paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        total += sum(f.stat().st_size for f in files)
    return total / 2 ** 20


def _meta(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text())


class LexiconPredictor:
    name, technique, precision = "Lexicon + edit distance", "B: regex + min. edit distance", "rules"

    def __init__(self):
        self.lex = Lexicon()
        self.threshold = 0.5

    def predict_proba(self, text):
        return 1.0 if self.lex.find(text) else 0.0

    def size_mb(self):
        return _size_mb([LEXICON_FILE])


class TfidfPredictor:
    name, technique, precision = "tf-idf + LogReg", "B: tf-idf + logistic regression", "FP64"

    def __init__(self):
        import joblib
        m = joblib.load(TFIDF_DIR / "tfidf_lr.joblib")
        self.word, self.char, self.clf = m["word"], m["char"], m["clf"]
        self.threshold = _meta(TFIDF_DIR)["threshold"]

    def predict_proba(self, text):
        from scipy.sparse import hstack
        t = [normalize(text)]
        x = hstack([self.word.transform(t), self.char.transform(t)]).tocsr()
        return float(self.clf.predict_proba(x)[0, 1])

    def size_mb(self):
        return _size_mb([TFIDF_DIR / "tfidf_lr.joblib"])


class W2VFFNPredictor:
    name, technique, precision = "word2vec + FFN", "A: own word2vec + feedforward NN (SGD)", "FP32"

    def __init__(self):
        import torch
        from .w2v_ffn import FFN, Featurizer
        self.torch = torch
        meta = _meta(W2V_DIR)
        itos = json.loads((W2V_DIR / "vocab.json").read_text(encoding="utf-8"))
        self.feat = Featurizer(itos, np.load(W2V_DIR / "vectors.npy"))
        self.model = FFN(meta["in_dim"])
        self.model.load_state_dict(torch.load(W2V_DIR / "ffn.pt", map_location="cpu"))
        self.model.eval()
        self.threshold = meta["threshold"]

    def predict_proba(self, text):
        x = self.torch.from_numpy(self.feat(text)).unsqueeze(0)
        with self.torch.no_grad():
            return float(self.torch.softmax(self.model(x), -1)[0, 1])

    def size_mb(self):
        return _size_mb([W2V_DIR / "vectors.npy", W2V_DIR / "vocab.json", W2V_DIR / "ffn.pt"])


class BertTorchPredictor:
    name, technique, precision = "BERTurk (PyTorch)", "C: fine-tuned BERTurk", "FP32"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BERT_FP32_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_FP32_DIR).eval()
        self.threshold = _meta(BERT_FP32_DIR)["threshold"]

    def predict_proba(self, text):
        enc = self.tok(normalize_for_bert(text), truncation=True, max_length=MAX_LEN,
                       return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        return float(self.torch.softmax(logits, -1)[0, 1])

    def size_mb(self):
        return _size_mb([BERT_FP32_DIR / "model.safetensors"])


class OnnxBertPredictor:
    technique = "C + D: BERTurk on ONNX Runtime"

    def __init__(self, int8: bool = True):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.path = ONNX_INT8 if int8 else ONNX_FP32
        self.precision = "INT8 (dynamic)" if int8 else "FP32"
        self.name = "BERTurk INT8 (ONNX)" if int8 else "BERTurk FP32 (ONNX)"
        self.technique = "D: dynamic INT8 quantization" if int8 else "C: BERTurk exported to ONNX"
        self.tok = AutoTokenizer.from_pretrained(ONNX_DIR)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        meta = json.loads((ONNX_DIR / "meta.json").read_text())
        self.threshold = meta["threshold_int8" if int8 else "threshold_fp32"]

    def predict_proba(self, text):
        enc = self.tok(normalize_for_bert(text), truncation=True, max_length=MAX_LEN,
                       return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
        logits = self.sess.run(None, feed)[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def size_mb(self):
        return _size_mb([self.path])


def load_all(include_onnx_fp32: bool = False, include_torch_fp32: bool = True) -> dict:
    """Load every model that exists on disk. Keys are display names."""
    builders = [LexiconPredictor, TfidfPredictor, W2VFFNPredictor]
    if include_torch_fp32:
        builders.append(BertTorchPredictor)
    if include_onnx_fp32:
        builders.append(lambda: OnnxBertPredictor(int8=False))
    builders.append(lambda: OnnxBertPredictor(int8=True))
    out = {}
    for b in builders:
        try:
            p = b()
            out[p.name] = p
        except Exception as e:   # model not trained / downloaded yet
            print(f"[predictors] skipped a model: {type(e).__name__}: {e}")
    return out
