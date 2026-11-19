"""Letters and digits in one token: 5G, 4K, 1080p, PS5, H100, x86, IPv6, 2FA, 5W1H, K3, A4, G20, S1.

How such a token is said is a habit of the field, not something its letters show: "5G" is English
("five G"), "K3" is Indonesian ("ka tiga"), "G20" Indonesian in an Indonesian sentence. So the habit
is data (alnum_data.csv), and the mechanics are code:

    style   EN    letters with English names, numbers in English, in both languages
            ID    Indonesian letter names and numbers in an Indonesian sentence, English ones in English
            WORD  the letters as a word, the number in the sentence's language ("Covid sembilan belas")
    number  words ("twenty-four"), pairs ("1080" -> "ten eighty"), digits ("9001" -> "nine zero zero one")

A token missing from the table takes the style acronym_clf.py gives its letters (two or more), or the
sentence's language for a single letter. In an Indonesian sentence English letter names are written the
way the voice says them ("pi-es five"), like EN_ACRONYM_STYLE=spell; the respelling search
(train/respell/) can measure and replace any of them.

What this module writes is final: it is protected from the rules that run after it (the capitals rule
would turn "PS" back into "pé-és"). speech_text.normalize calls reset() first and restore() last.
"""

from __future__ import annotations

import csv
import re
from contextvars import ContextVar
from pathlib import Path
from string import ascii_uppercase

from num2words import num2words

import acronym_clf

ID_LETTERS = dict(zip(ascii_uppercase, "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))
EN_LETTERS = dict(zip(ascii_uppercase, "ei bi si di i ef ji eich ai jei kei el em en ou pi kyu ar es ti yu vi dabelyu eks wai zi".split()))

# ---------------------------------------------------------------- protection

_kept: ContextVar[list[str]] = ContextVar("alnum_kept", default=[])
_SLOT = re.compile("([-])")


def reset() -> None:
    _kept.set([])


def protect(text: str) -> str:
    """A placeholder with no letters or digits, so no later rule can touch what it stands for."""
    kept = _kept.get()
    if not kept:
        kept = []
        _kept.set(kept)
    kept.append(text)
    return f"{chr(0xE100 + len(kept) - 1)}"


def restore(text: str) -> str:
    kept = _kept.get()
    return _SLOT.sub(lambda m: kept[ord(m.group(1)) - 0xE100], text)


# ---------------------------------------------------------------- the table

def _load() -> dict[str, dict]:
    path = Path(__file__).resolve().parent / "alnum_data.csv"
    with path.open(encoding="utf-8") as f:
        return {r["token"]: r for r in csv.DictReader(f) if r.get("token")}


TABLE = _load()
# Letters and digits mixed, at least one of each; "5W1H", "IPv6", "x86", "1080p".
TOKEN = re.compile(r"(?<![\w.-])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{2,10}(?![\w-]|\.\d)")


def words(n: int, lang: str) -> str:
    return num2words(n, lang="en" if lang == "en" else "id")


def say_number(digits: str, mode: str, lang: str) -> str:
    if mode == "digits":
        return " ".join(words(int(d), lang) for d in digits)
    if mode == "pairs" and len(digits) in (3, 4):
        head, tail = digits[:-2], digits[-2:]
        if tail == "00":
            return f"{words(int(head), lang)} {'hundred' if lang == 'en' else 'ratus'}"
        tail_said = words(int(tail), lang) if tail[0] != "0" else f"{'oh' if lang == 'en' else 'nol'} {words(int(tail[1]), lang)}"
        return f"{words(int(head), lang)} {tail_said}"
    return words(int(digits), lang)


def say_letters(letters: str, style: str, lang: str) -> str:
    up = letters.upper()
    if style == "WORD":
        return letters.capitalize()
    if lang == "en":
        return " ".join(up)
    names = EN_LETTERS if style == "EN" else ID_LETTERS
    return "-".join(names[c] for c in up)


def guess_style(token: str, lang: str) -> str:
    letters = "".join(re.findall(r"[A-Za-z]", token)).upper()
    if len(letters) >= 2:
        label = acronym_clf.known(letters) or acronym_clf.classify(letters)
        if label:
            return {"SPELL_EN": "EN", "SPELL_ID": "ID", "WORD": "WORD"}[label]
    return "EN" if lang == "en" else "ID"


def say(token: str, lang: str) -> str:
    row = TABLE.get(token) or TABLE.get(token.upper())
    custom = row and (row.get("en") if lang == "en" else row.get("id"))
    if custom:
        return custom
    style = (row or {}).get("style") or guess_style(token, lang)
    mode = (row or {}).get("number") or "words"
    number_lang = "en" if (style == "EN" or lang == "en") else "id"
    parts = re.findall(r"[A-Za-z]+|\d+", token)
    said = [say_number(p, mode, number_lang) if p.isdigit() else say_letters(p, style, lang) for p in parts]
    return " ".join(said)


# ---------------------------------------------------------------- versions, standards, pairs of letters

VERSIONED = {"USB": "EN", "Bluetooth": "EN", "Wi-Fi": "EN", "WiFi": "EN", "PCIe": "EN", "DDR": "EN", "Android": "EN",
             "iOS": "EN", "Python": "EN", "Windows": "EN", "macOS": "EN", "version": "EN", "HDMI": "ID", "versi": "ID"}
_VERSION = re.compile(r"\b(" + "|".join(map(re.escape, VERSIONED)) + r")\s?(\d+(?:\.\d+)*)\b(?!\.\d)")


def _version(m: re.Match, lang: str) -> str:
    style = VERSIONED[m.group(1)]
    n_lang = "en" if style == "EN" or lang == "en" else "id"
    point = " point " if n_lang == "en" else " titik "
    pieces = m.group(2).split(".")
    said = point.join("oh" if (p == "0" and i and n_lang == "en") else words(int(p), n_lang) for i, p in enumerate(pieces))
    word = m.group(1)
    if word == "HDMI" and lang != "en":
        word = "ha-dé-ém-i"
    return f"{word} {protect(said)}"


def _ieee(m: re.Match, lang: str) -> str:
    """'802.11ax' -> 'eight oh two dot eleven ei-eks' (English: a standard's name)."""
    tail = m.group(2)
    letters = (" " + say_letters(tail, "EN", lang)) if tail else ""
    return protect(f"eight oh two dot {words(int(m.group(1)), 'en')}{letters}")


def _letter_pair(m: re.Match, lang: str, joiner: str) -> str:
    """'R&D' -> 'ar en di', 'Q&A' -> 'kyu en ei', 'A/B' -> 'ei bi'; English: 'R and D', 'A B'."""
    a, b = m.group(1), m.group(2)
    if lang == "en":
        return protect(f"{a} and {b}" if joiner == "&" else f"{a} {b}")
    mid = " en " if joiner == "&" else " "
    return protect(f"{EN_LETTERS[a]}{mid}{EN_LETTERS[b]}")


def rules(lang: str) -> list:
    return [
        (re.compile(r"\b802\.(\d+)([a-z]{0,2})\b"), lambda m: _ieee(m, lang)),
        (_VERSION, lambda m: _version(m, lang)),
        (re.compile(r"\b([A-Z])&([A-Z])\b"), lambda m: _letter_pair(m, lang, "&")),
        (re.compile(r"\b([A-Z])/([A-Z])\b"), lambda m: _letter_pair(m, lang, "/")),
        # "COVID-19", "Sea Games-2": a word, then its number
        (re.compile(r"\b(?!ke-)([A-Za-z]{2,})-(\d+)\b"), lambda m: f"{m.group(1)} {m.group(2)}"),
        # "3x", "1920x1080" are times and sizes (everyday.py), not names
        (TOKEN, lambda m: m.group(0) if re.fullmatch(r"\d+[xX]\d*", m.group(0)) else protect(say(m.group(0), lang))),
    ]
