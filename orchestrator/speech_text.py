"""Text rewrites applied only to what the robot says (Indonesian), registered as TTS text_transforms.

Pipecat's own number, currency, date and percent transforms are English-only
(num2words lang="en", "dollars", "January"), so Indonesian gets its own here.
Indonesian writing: "." groups thousands (150.000) and "," is the decimal mark (3,5).

Decisions for ambiguous input:
- "14.30" or "08.00" with two digits after the dot is a time, with or without "pukul":
  in Indonesian a dot never marks a decimal and thousands groups have three digits.
- "14:30" is always a time.
- A number that starts with 0 or +62 and has at least 9 digits is a phone number and is
  read digit by digit; other codes are left as numbers.
- Uppercase words of 2 to 6 letters (ITS, UMKM, WIB, FTEIC) are spelled with Indonesian
  letter names, except the few in READ_AS_WORD that are said as a word (RAISA).
"""

import re

import numpy as np
from num2words import num2words

from pipecat.utils.text.transforms.strip_markdown import strip_markdown


def words(n: int) -> str:
    return num2words(n, lang="id")


def digits(s: str) -> str:
    return " ".join(words(int(d)) for d in s if d.isdigit())


def number(whole: str, frac: str | None = None) -> str:
    """'150.000' -> seratus lima puluh ribu; with frac '5' -> ... koma lima (frac read per digit)."""
    text = words(int(whole.replace(".", "")))
    return f"{text} koma {digits(frac)}" if frac else text


def clock(h: str, m: str) -> str:
    hour, minute = int(h), int(m)
    if minute == 0:
        return words(hour)
    if minute < 10:
        return f"{words(hour)} lewat {words(minute)} menit"
    return f"{words(hour)} {words(minute)}"


LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   "a be ce de e ef ge ha i je ka el em en o pe ki er es te u fe we eks ye zet".split()))
ABBREVIATIONS = {"dll": "dan lain-lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
                 "tsb": "tersebut", "yg": "yang", "dgn": "dengan", "utk": "untuk"}
UNITS = {"km": "kilometer", "m": "meter", "cm": "sentimeter", "mm": "milimeter", "kg": "kilogram",
         "g": "gram", "l": "liter", "ml": "mililiter", "km/jam": "kilometer per jam"}
# Robot and team names written in capitals but said as a word, not spelled.
READ_AS_WORD = {"RAISA", "NABBIT", "MAGE", "IRIS", "BIMA"}
SCALES = r"(?:\s?(ribu|juta|miliar|triliun))?"
NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?"  # Indonesian: 1.500.000,75

RULES = [
    # Table rows and their |---| separators: tables go to the screen, never to the voice.
    (re.compile(r"^\s*\|.*\|\s*$\n?", re.M), lambda m: ""),
    (re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]"), lambda m: ""),
    (re.compile(r"(\+62|\b0)[\d\- ]{8,}\d"), lambda m: ("plus enam dua " if m.group(1) == "+62" else "nol ")
     + digits(m.group(0)[len(m.group(1)):])),
    (re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b(?![.,]?\d)"), lambda m: clock(m.group(1), m.group(2))),
    (re.compile(rf"\bRp\.?\s?{NUM}{SCALES}(?:,-|,00)?"),
     lambda m: " ".join(filter(None, [number(m.group(1), m.group(2)), m.group(3), "rupiah"]))),
    (re.compile(rf"{NUM}\s?%"), lambda m: f"{number(m.group(1), m.group(2))} persen"),
    (re.compile(rf"\b{NUM}\s?(km/jam|km|cm|mm|kg|ml|m|g|l)\b"),
     lambda m: f"{number(m.group(1), m.group(2))} {UNITS[m.group(3)]}"),
    (re.compile(r"\bke-(\d+)\b"), lambda m: "ke" + words(int(m.group(1)))),
    (re.compile(rf"\b{NUM}\b"), lambda m: number(m.group(1), m.group(2))),
    # The dot after "dll." stays: it usually also ends the sentence, which gives the pause.
    (re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b"), lambda m: ABBREVIATIONS[m.group(1)]),
    (re.compile(r"\b[A-Z]{2,6}\b"),
     lambda m: m.group(0) if m.group(0) in READ_AS_WORD else " ".join(LETTERS[c] for c in m.group(0))),
]


def normalize(text: str) -> str:
    for pattern, replace in RULES:
        text = pattern.sub(replace, text)
    return text


async def screen_text(text: str) -> str:
    """What the robot says, as shown to the visitor: no markdown, no table rows, digits kept."""
    return RULES[0][0].sub("", await strip_markdown(text, "*")).strip()


# Markdown goes before run_tts; the Indonesian rewrite happens inside run_tts (normalize), after
# the sentence's screen text is sent, so the screen keeps "Rp150.000" while the voice says the words.
SPEECH_TRANSFORMS = [("*", strip_markdown)]

# Extra syllables of silence a word's trailing mark adds before the next word.
PAUSES = {",": 1.0, ";": 1.5, ":": 1.5, ".": 2.0, "?": 2.0, "!": 2.0}


def word_times(shown: list[str], audio: bytes, sample_rate: int) -> list[float]:
    """Start time (s) of each on-screen word in its sentence's PCM16 audio.

    Words are spread over the voiced span in proportion to the syllables of their spoken
    form (so "Rp150.000" gets the time of "seratus lima puluh ribu rupiah") plus the pause
    their punctuation adds.
    shortcut: each word is normalized alone, so a rule spanning words ("pukul 08.00 WIB")
    weighs slightly off; real word timestamps from the TTS would replace this estimate.
    """
    level = np.abs(np.frombuffer(audio, "<i2").astype(np.int32))
    voiced = np.flatnonzero(level > max(500, int(level.max(initial=0) * 0.05)))
    if not len(voiced):
        return [0.0] * len(shown)
    start, end = voiced[0] / sample_rate, voiced[-1] / sample_rate
    weights = [max(1, len(re.findall(r"[aiueo]+", normalize(w).lower()))) + PAUSES.get(w[-1], 0.0) for w in shown]
    total, before, times = sum(weights), 0.0, []
    for w in weights:
        times.append(float(start + (end - start) * before / total))
        before += w
    return times

