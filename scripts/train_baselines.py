"""Model 1 (classical baselines, Area B).

1a. Lexicon + edit-distance rule matcher (no training; evaluated on validation).
1b. tf-idf (word 1-2 grams + character 2-5 grams) + logistic regression.

Run:  python scripts/train_baselines.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import TFIDF_DIR
from src.data import load_split
from src.lexicon import Lexicon
from src.metrics import best_threshold, macro_f1, report
from src.normalize import normalize


def main():
    train, val = load_split("train"), load_split("val")
    Xtr = [normalize(t) for t in train["text"]]
    Xva = [normalize(t) for t in val["text"]]

    # ---- 1a. lexicon rule baseline ----
    lex = Lexicon()
    pred = [int(bool(lex.find(t))) for t in val["text"]]
    print(f"[lexicon] validation macro-F1 = {macro_f1(val['label'], pred):.4f}")

    # ---- 1b. tf-idf + logistic regression ----
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=200_000)
    Ftr = hstack([word.fit_transform(Xtr), char.fit_transform(Xtr)]).tocsr()
    Fva = hstack([word.transform(Xva), char.transform(Xva)]).tocsr()

    best = None
    for C in [0.5, 1.0, 2.0, 4.0, 8.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(Ftr, train["label"])
        p = clf.predict_proba(Fva)[:, 1]
        th = best_threshold(val["label"], p)
        f1 = macro_f1(val["label"], (p >= th).astype(int))
        print(f"[tfidf] C={C:<4} threshold={th:.2f} validation macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, th, clf)

    f1, C, th, clf = best
    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"word": word, "char": char, "clf": clf}, TFIDF_DIR / "tfidf_lr.joblib",
                compress=3)
    p = clf.predict_proba(Fva)[:, 1]
    meta = {"C": C, "threshold": th, "val": report(val["label"], (p >= th).astype(int))}
    (TFIDF_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[tfidf] best C={C}, threshold={th}, validation macro-F1={f1:.4f} -> saved to {TFIDF_DIR}")


if __name__ == "__main__":
    main()
