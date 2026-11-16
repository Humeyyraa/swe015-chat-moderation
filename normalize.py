"""Area B - regex normalization of noisy chat text.

Handles the tricks people use to slip past filters:
  * leetspeak:            s4l4k -> salak,  @ptal -> aptal
  * separated letters:    a.p.t.a.l -> aptal,  s-a-l-a-k -> salak,  a m k -> amk
  * repeated letters:     salaaaaak -> salak   (3+ repeats -> 1)
  * Turkish casing:       I -> ı, İ -> i  (Python's .lower() gets this wrong)
  * Turkish characters:   optional ASCII folding ş->s, ı->i, ... so that
                          "serefsiz" and "şerefsiz" become the same token
  * dataset placeholders: @USER and URL are removed
"""
import re

_LEET = str.maketrans({"4": "a", "@": "a", "3": "e", "1": "i",
                       "0": "o", "5": "s", "$": "s", "7": "t"})
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")

_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)")
_USER = re.compile(r"@USER\b")
# a token that mixes letters with leet characters, e.g. "s4l4k", "@ptal"
_LEET_TOKEN = re.compile(r"(?=\S*[a-zçğıöşü])(?=\S*[4@3105$7])[\w@$]+", re.I)
# single letters joined by . - _ * : "a.p.t.a.l"
_SPACED = re.compile(r"\b(?:\w[.\-_*]){2,}\w\b")
# 3+ single letters separated by spaces: "a m k" -> "amk"
_SPACED_WS = re.compile(r"(?<!\S)(?:\w ){2,}\w(?!\S)")
_REPEAT = re.compile(r"(\w)\1{2,}")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zçğıöşüâîû0-9]+", re.I)


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def fold_turkish(text: str) -> str:
    return text.translate(_FOLD)


def normalize(text: str, lower: bool = True, fold: bool = True) -> str:
    """Full normalization for the classical models and the lexicon.

    lower/fold are switched off for BERTurk, which is a *cased* model with
    Turkish characters in its vocabulary.
    """
    t = _URL.sub(" ", text)
    t = _USER.sub(" ", t)
    t = _SPACED.sub(lambda m: re.sub(r"[.\-_*]", "", m.group(0)), t)
    t = _SPACED_WS.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = _LEET_TOKEN.sub(lambda m: m.group(0).translate(_LEET), t)
    t = _REPEAT.sub(r"\1", t)
    if lower:
        t = turkish_lower(t)
    if fold:
        t = fold_turkish(t)
    return _WS.sub(" ", t).strip()


def normalize_for_bert(text: str) -> str:
    return normalize(text, lower=False, fold=False)


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer used after normalize()."""
    return _TOKEN.findall(text)
