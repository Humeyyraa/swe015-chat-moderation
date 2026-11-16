"""Download OffensEval-TR 2020 and build the train / validation / test splits.

* train + validation: the official training file (31,756 tweets), split 90/10
  stratified with a fixed seed.
* test: the official test set (3,528 tweets) with its gold labels. This SAME test
  set is used for every model in the results table.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import DATA_PROCESSED, DATA_RAW, DATA_URL, SEED

csv.field_size_limit(sys.maxsize)


def _find(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def _read_tsv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return pd.DataFrame(rows)


def download() -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_RAW / "offenseval2020-turkish.zip"
    if not zip_path.exists():
        print(f"Downloading {DATA_URL}")
        r = requests.get(DATA_URL, timeout=120)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_RAW)
    return DATA_RAW


def load_raw():
    root = download()
    train = _read_tsv(_find(root, "offenseval-tr-training-v1.tsv"))
    test = _read_tsv(_find(root, "offenseval-tr-testset-v1.tsv"))

    # Gold test labels: "id,label" per line, no header.
    labels_path = _find(root, "offenseval-tr-labela-v1.tsv")
    with open(labels_path, encoding="utf-8") as f:
        lab = dict(line.strip().split(",")[:2] for line in f if line.strip())
    test["subtask_a"] = test["id"].map(lab)
    if test["subtask_a"].isna().any():
        raise ValueError("Some test ids have no gold label")

    for df in (train, test):
        df.rename(columns={"tweet": "text"}, inplace=True)
        df["label"] = (df["subtask_a"].str.strip() == "OFF").astype(int)
    cols = ["id", "text", "label"]
    return train[cols], test[cols]


def prepare():
    train_full, test = load_raw()
    train, val = train_test_split(
        train_full, test_size=0.10, stratify=train_full["label"], random_state=SEED
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(df):6d} messages, offensive = {df['label'].mean():.1%}")


def load_split(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{name}.csv"
    if not path.exists():
        prepare()
    df = pd.read_csv(path, keep_default_na=False)
    df["text"] = df["text"].astype(str)
    return df
