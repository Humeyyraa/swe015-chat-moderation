"""Download OffensEval-TR 2020 and write data/processed/{train,val,test}.csv.

Run:  python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import prepare

if __name__ == "__main__":
    prepare()
