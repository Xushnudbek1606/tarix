"""Uzbek Latin <-> Cyrillic transliteration and text helpers.

The whole system stores and transmits Uzbek text in canonical Latin
(using U+2018 for o‘ / g‘ and U+2019 for the tutuq belgisi in ma’lumot).
Cyrillic is produced on demand, per client, via :func:`to_script`.
"""
from __future__ import annotations

import re
import unicodedata

# Every apostrophe-like character that people type for o‘, g‘ and ’.
_APOS_CHARS = "'`´ʻʼ‘’′"
G_APOS = "‘"   # ‘  used after o / g
TUTUQ = "’"    # ’  tutuq belgisi (ma’lumot)

_LAT_VOWELS = set("aeiouAEIOU")
_CYR_VOWELS = set("аеёиоуўэюяАЕЁИОУЎЭЮЯ")


def normalize_latin(text: str) -> str:
    """Canonicalise apostrophes and Unicode form of Uzbek Latin text."""
    text = unicodedata.normalize("NFC", text)
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _APOS_CHARS:
            prev = out[-1] if out else ""
            if prev in ("o", "O", "g", "G"):
                out.append(G_APOS)
            else:
                out.append(TUTUQ)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Latin -> Cyrillic
# --------------------------------------------------------------------------- #
_L2C_SINGLE = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х",
    "y": "й", "z": "з", "c": "к", "w": "в",
}


def _match_case(src: str, dst: str) -> str:
    """Give *dst* the capitalisation pattern of *src* (first char decides)."""
    if src.isupper() and len(src) >= 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def latin_to_cyrillic(text: str) -> str:
    text = normalize_latin(text)
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        low = ch.lower()
        nxt = text[i + 1].lower() if i + 1 < n else ""
        nxt2 = text[i + 2] if i + 2 < n else ""
        at_word_start = i == 0 or (not text[i - 1].isalpha() and text[i - 1] not in (G_APOS, TUTUQ))

        # o‘ / g‘
        if low in ("o", "g") and i + 1 < n and text[i + 1] == G_APOS:
            out.append(_match_case(ch, "ў" if low == "o" else "ғ"))
            i += 2
            continue
        # tutuq belgisi
        if ch == TUTUQ:
            out.append("ъ")
            i += 1
            continue
        # digraphs
        if low == "s" and nxt == "h":
            out.append(_match_case(ch, "ш")); i += 2; continue
        if low == "c" and nxt == "h":
            out.append(_match_case(ch, "ч")); i += 2; continue
        if low == "n" and nxt == "g":
            out.append(_match_case(ch, "нг")); i += 2; continue
        if low == "y" and nxt in ("o", "u", "a", "e"):
            # "yo‘q" is й + ў, not ё
            if nxt == "o" and nxt2 == G_APOS:
                out.append(_match_case(ch, "й")); i += 1; continue
            mapping = {"o": "ё", "u": "ю", "a": "я", "e": "е"}
            out.append(_match_case(ch, mapping[nxt])); i += 2; continue
        if low == "e":
            out.append(_match_case(ch, "э" if at_word_start else "е"))
            i += 1
            continue
        if low in _L2C_SINGLE:
            out.append(_match_case(ch, _L2C_SINGLE[low]))
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Cyrillic -> Latin
# --------------------------------------------------------------------------- #
_C2L = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ц": "ts", "ч": "ch", "ш": "sh", "ъ": TUTUQ, "ь": "", "ы": "i",
    "э": "e", "ю": "yu", "я": "ya", "ё": "yo", "ў": "o" + G_APOS,
    "қ": "q", "ғ": "g" + G_APOS, "ҳ": "h",
}


def cyrillic_to_latin(text: str) -> str:
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        if low == "е":
            prev = text[i - 1] if i > 0 else ""
            lat = "ye" if (i == 0 or not prev.isalpha() or prev in _CYR_VOWELS or prev in "ъЪ") else "e"
            out.append(_match_case(ch, lat))
        elif low in _C2L:
            lat = _C2L[low]
            out.append(_match_case(ch, lat) if lat else "")
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
_CYR_RE = re.compile(r"[Ѐ-ӿ]")


def detect_script(text: str) -> str:
    """Return 'cyrillic' if the text contains Cyrillic letters else 'latin'."""
    return "cyrillic" if _CYR_RE.search(text or "") else "latin"


def to_script(text: str, script: str) -> str:
    """Render *text* (any script) in the requested script ('latin'|'cyrillic')."""
    script = (script or "latin").lower()
    if script.startswith("cyr"):
        return latin_to_cyrillic(cyrillic_to_latin(text) if detect_script(text) == "cyrillic" else text)
    return normalize_latin(cyrillic_to_latin(text)) if detect_script(text) == "cyrillic" else normalize_latin(text)


def capitalize_sentence(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


_RUSSIAN_ONLY = set("ыщьЫЩЬ")


def is_probably_uzbek(text: str) -> bool:
    """False for text that contains letters which exist only in Russian (ы, щ, ь).

    Such messages are passed through unchanged instead of being transliterated,
    so Russian replies are never corrupted by the Uzbek converter.
    """
    return not any(ch in _RUSSIAN_ONLY for ch in text or "")


def canonical_text(text: str) -> str:
    """Storage form: Latin text gets canonical apostrophes, everything else is kept as typed."""
    if is_probably_uzbek(text) and detect_script(text) == "latin":
        return normalize_latin(text)
    return text
