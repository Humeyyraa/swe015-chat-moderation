"""Area B - rule baseline: lexicon matcher with minimum edit distance.

A message is OFFENSIVE if any of its normalized tokens matches the lexicon:
  * exact entries ("=got"): token must be identical,
  * prefix entries ("salak"): token starts with the root (Turkish suffixes),
  * fuzzy: for entries with >= 6 letters, a token (or its prefix of the same
    length) within Levenshtein distance 1 also matches ("serefsz" -> "serefsiz").
"""
from pathlib import Path

from .config import LEXICON_FILE
from .normalize import normalize, tokenize

FUZZY_MIN_LEN = 6
FUZZY_MAX_DIST = 1


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein: insert, delete, substitute = cost 1),
    computed with dynamic programming, keeping only two rows of the table."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,               # deletion
                           cur[j - 1] + 1,            # insertion
                           prev[j - 1] + (ca != cb))) # substitution
        prev = cur
    return prev[-1]


class Lexicon:
    def __init__(self, path: Path = LEXICON_FILE):
        self.exact, self.prefix = set(), set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("="):
                self.exact.add(line[1:])
            else:
                self.prefix.add(line)
        self.fuzzy = [w for w in self.exact | self.prefix if len(w) >= FUZZY_MIN_LEN]

    def match_token(self, tok: str):
        """Return the lexicon entry that `tok` matches, or None."""
        if tok in self.exact:
            return tok
        for root in self.prefix:
            if tok.startswith(root):
                return root
        if len(tok) >= FUZZY_MIN_LEN - 1:
            for w in self.fuzzy:
                if abs(len(tok) - len(w)) <= FUZZY_MAX_DIST and edit_distance(tok, w) <= FUZZY_MAX_DIST:
                    return w
                if w in self.prefix and len(tok) > len(w):
                    if edit_distance(tok[: len(w)], w) <= FUZZY_MAX_DIST:
                        return w
        return None

    def find(self, text: str):
        """List of (token, matched_entry) in the normalized message."""
        hits = []
        for tok in tokenize(normalize(text)):
            m = self.match_token(tok)
            if m:
                hits.append((tok, m))
        return hits

    def mask(self, text: str) -> str:
        """Star-mask the words of the ORIGINAL message whose normalized form matches."""
        out = []
        for word in text.split(" "):
            norm = tokenize(normalize(word))
            if norm and any(self.match_token(t) for t in norm):
                out.append(word[:1] + "*" * max(len(word) - 1, 2))
            else:
                out.append(word)
        return " ".join(out)
