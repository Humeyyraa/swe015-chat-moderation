"""Evaluation helpers shared by every model."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def best_threshold(y_true, p_off) -> float:
    """Decision threshold on P(offensive) that maximizes macro-F1.
    Tuned on the VALIDATION set only, then frozen and used on the test set."""
    y_true, p_off = np.asarray(y_true), np.asarray(p_off)
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [macro_f1(y_true, (p_off >= t).astype(int)) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def report(y_true, y_pred) -> dict:
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_NOT": float(f1s[0]),
        "f1_OFF": float(f1s[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
