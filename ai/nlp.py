"""Sign-gloss sequence -> natural Uzbek sentence (rule-based, data-driven, replaceable).

Uzbek Sign Language word order differs from spoken Uzbek, so glosses are turned into a
sentence by templates plus a few morphology rules. Unknown combinations fall back to a
literal gloss join, which is shown (never silently "corrected") so the employee can judge.
Add new templates to TEMPLATES (or later swap this module for a seq2seq model).
"""
from __future__ import annotations

import json
from pathlib import Path

from common.uzbek_text import capitalize_sentence, normalize_latin

# gloss tuple (after dedup) -> sentence. Longest match wins.
TEMPLATES: dict[tuple[str, ...], str] = {
    ("MEN", "PASPORT", "OLISH", "KELDIM"): "Men pasport olish uchun keldim.",
    ("PASPORT", "OLISH", "KELDIM"): "Men pasport olish uchun keldim.",
    ("MEN", "HUJJAT", "OLISH", "KELDIM"): "Men hujjat olish uchun keldim.",
    ("MEN", "ARIZA", "BERISH", "KELDIM"): "Men ariza topshirish uchun keldim.",
    ("TUGILGANLIK", "GUVOHNOMA", "OLISH", "KELDIM"): "Men tug‘ilganlik guvohnomasini olish uchun keldim.",
    ("MEN", "TUGILGANLIK", "GUVOHNOMA", "OLISH", "KELDIM"): "Men tug‘ilganlik guvohnomasini olish uchun keldim.",
    ("TUGILGANLIK", "GUVOHNOMA"): "Tug‘ilganlik guvohnomasi.",
    ("MEN", "YORDAM"): "Menga yordam kerak.",
    ("YORDAM",): "Menga yordam kerak.",
    ("MALUMOT", "YORDAM"): "Menga ma’lumot kerak.",
    ("MEN", "MALUMOT"): "Menga ma’lumot kerak.",
    ("SALOM",): "Salom.",
    ("RAHMAT",): "Rahmat.",
    ("HA",): "Ha.",
    ("YOQ",): "Yo‘q.",
    ("KUTING",): "Kutaman.",
    ("TAYYOR",): "Tayyor.",
    ("MANA",): "Mana.",
    ("MANA", "PASPORT"): "Mana, pasportim.",
    ("MANA", "HUJJAT"): "Mana, hujjatim.",
}

class SentenceBuilder:
    def __init__(self, signs_file: Path | None = None) -> None:
        self.words: dict[str, str] = {}
        self.pos: dict[str, str] = {}
        if signs_file and Path(signs_file).exists():
            for s in json.loads(Path(signs_file).read_text(encoding="utf-8"))["signs"]:
                self.words[s["name"]] = normalize_latin(s["uz_latin"])
                self.pos[s["name"]] = s.get("pos", "noun")

    def reload(self, signs_file: Path) -> None:
        self.__init__(signs_file)

    @staticmethod
    def dedupe(glosses: list[str]) -> list[str]:
        out: list[str] = []
        for g in glosses:
            if not out or out[-1] != g:
                out.append(g)
        return out

    def build(self, glosses: list[str]) -> str:
        g = tuple(self.dedupe([x.upper() for x in glosses]))
        if not g:
            return ""
        # longest template contained in the sequence, at the start
        for n in range(len(g), 0, -1):
            hit = TEMPLATES.get(g[:n])
            if hit and n == len(g):
                return hit
        # sub-sequence match (e.g. greeting + request)
        parts: list[str] = []
        i = 0
        while i < len(g):
            for n in range(len(g) - i, 0, -1):
                hit = TEMPLATES.get(g[i:i + n])
                if hit:
                    parts.append(hit.rstrip("."))
                    i += n
                    break
            else:
                parts.append(self.words.get(g[i], g[i].lower()))
                i += 1
        parts = [parts[0]] + [x[:1].lower() + x[1:] for x in parts[1:]]
        text = ", ".join(parts)
        return capitalize_sentence(text) + "."
