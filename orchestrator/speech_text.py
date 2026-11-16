"""Text rewrites applied only to what the robot says, Indonesian or English per sentence.

Pipecat's own number, currency, date and percent transforms are English-only
(num2words lang="en", "dollars", "January"), so Indonesian gets its own here.
Indonesian writing: "." groups thousands (150.000) and "," is the decimal mark (3,5).

Decisions for ambiguous input:
- "14.30" or "08.00" with two digits after the dot is a time, with or without "pukul":
  in Indonesian a dot never marks a decimal and thousands groups have three digits.
- "14:30" is always a time.
- A number that starts with 0 or +62 and has at least 9 digits is a phone number and is
  read digit by digit; other codes are left as numbers.
- Acronyms (ITS, USB, HDMI, GPU) go to the voice as written: LoRA v2 was trained on raw
  acronyms and reads them best that way. READ_AS_WORD and capitals longer than 6 letters
  (BANDUNG) are said as a Title Case word.
- FIXED overrides where the raw form reads worse (QRIS "kyuris", units kWh/GHz/kHz) hold in
  both languages; SPELLED acronyms (WIB, WIT) take the sentence language's letter names.
- A sentence is English when it has English function words and no Indonesian ones; mixed
  sentences ("silakan login pakai email") stay Indonesian. English sentences use "," for
  thousands and "." for decimals.
The LoRA v2 "hybrid" training transcripts are this output with QRIS, WIB and WIT left raw.
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
# Said the same way in both languages; picked by ear over the raw acronym (LoRA v2 A/B).
# "colokkan" is read with a k; "ch" gave the Indonesian c 3/3 (other c-words were already right).
FIXED = {"QRIS": "kyuris", "kWh": "kilowatt hour", "GHz": "gigahertz", "kHz": "kilohertz",
         "colokkan": "cholokkan", "Colokkan": "Cholokkan"}
# Spelled with the sentence language's letter names ("we i be"); every other acronym stays raw.
SPELLED = {"WIB", "WIT"}
ID_WORDS = set("""yang dan di ke dari ini itu untuk dengan saya anda kamu ada tidak akan bisa sudah
    silakan mohon ya adalah pada atau juga kami kita apa mana berapa sampai pukul jam dalam
    selamat terima kasih tolong sedang belum lagi""".split())
EN_WORDS = set("""the is are you your to of and in on at for with please this that it we our my
    will can be have has what where how thank welcome yes here there from by me i an a""".split())
ABBREVIATIONS = {"dll": "dan lain-lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
                 "tsb": "tersebut", "yg": "yang", "dgn": "dengan", "utk": "untuk"}
UNITS = {"km": "kilometer", "m": "meter", "cm": "sentimeter", "mm": "milimeter", "kg": "kilogram",
         "g": "gram", "l": "liter", "ml": "mililiter", "km/jam": "kilometer per jam"}
# Written in capitals but said as a word, not spelled: robot and team names, words, places, people.
READ_AS_WORD = set("""RAISA NABBIT MAGE IRIS BIMA LUNA WITA BRIN SIM RAM LAN SKEM EXPO SPARK UNAIR
    HIMATEKTRO TENDON GARUDA NUSANTARA ANDI ARIF BALI BANDUNG DEWI DINA FARHAN JAKARTA JOGJA KARA
    MALANG MEDAN NORA PUTRA RARA RENDI SATRIA SEMARANG SENO SINTA SURABAYA TASYA YOGI ZARA""".split())
SCALES = r"(?:\s?(ribu|juta|miliar|triliun))?"
NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?"  # Indonesian: 1.500.000,75

FRONT = [
    # Table rows and their |---| separators: tables go to the screen, never to the voice.
    (re.compile(r"^\s*\|.*\|\s*$\n?", re.M), lambda m: ""),
    (re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]"), lambda m: ""),
    (re.compile(r"\b(" + "|".join(FIXED) + r")\b"), lambda m: FIXED[m.group(1)]),
]


def say_capitals(word: str, letters) -> str:
    if word in READ_AS_WORD or len(word) > 6:
        return word.capitalize()
    return " ".join(letters(c) for c in word) if word in SPELLED else word


RULES = [
    (re.compile(r"\s?°C"), lambda m: " derajat Celsius"),
    (re.compile(r"\s?°"), lambda m: " derajat"),
    (re.compile(r"\s?m²"), lambda m: " meter persegi"),
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
    (re.compile(r"\b[A-Z]{2,}\b"), lambda m: say_capitals(m.group(0), LETTERS.get)),
]


def en_words(n: int) -> str:
    return num2words(n, lang="en")


def en_number(whole: str, frac: str | None = None) -> str:
    text = en_words(int(whole.replace(",", "")))
    return f"{text} point {' '.join(en_words(int(d)) for d in frac)}" if frac else text


EN_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"  # English: 1,500,000.75
EN_UNITS = {"km": "kilometer", "m": "meter", "cm": "centimeter", "kg": "kilogram", "g": "gram",
            "l": "liter", "ml": "milliliter", "km/h": "kilometers per hour", "km/jam": "kilometers per hour"}


def en_quantity(m: re.Match) -> str:
    unit = EN_UNITS[m.group(3)]
    plural = "" if unit.endswith("hour") or (m.group(1) == "1" and not m.group(2)) else "s"
    return f"{en_number(m.group(1), m.group(2))} {unit}{plural}"


EN_RULES = [
    (re.compile(r"\s?°C"), lambda m: " degrees Celsius"),
    (re.compile(r"\s?°"), lambda m: " degrees"),
    (re.compile(r"\s?m²"), lambda m: " square meters"),
    (re.compile(rf"\b(?:Rp\.?|IDR)\s?{EN_NUM}"), lambda m: f"{en_number(m.group(1), m.group(2))} rupiah"),
    (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b"),
     lambda m: en_words(int(m.group(1))) + ("" if m.group(2) == "00" else
               (" oh " if m.group(2)[0] == "0" else " ") + en_words(int(m.group(2))))),
    (re.compile(rf"{EN_NUM}\s?%"), lambda m: f"{en_number(m.group(1), m.group(2))} percent"),
    (re.compile(rf"\b{EN_NUM}\s?(km/h|km/jam|km|cm|kg|ml|m|g|l)(?!\w)"), en_quantity),
    (re.compile(rf"\b{EN_NUM}\b"), lambda m: en_number(m.group(1), m.group(2))),
    (re.compile(r"\b[A-Z]{2,}\b"), lambda m: say_capitals(m.group(0), lambda c: c)),
]


def language(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    english = any(t in EN_WORDS for t in tokens) and not any(t in ID_WORDS for t in tokens)
    return "en" if english else "id"


def normalize(text: str, lang: str | None = None) -> str:
    """Spoken form of one sentence; `lang` ("id"/"en") overrides detection, e.g. for single words."""
    rules = EN_RULES if (lang or language(text)) == "en" else RULES
    for pattern, replace in FRONT + rules:
        text = pattern.sub(replace, text)
    return text


async def screen_text(text: str) -> str:
    """What the robot says, as shown to the visitor: no markdown, no table rows, digits kept."""
    return FRONT[0][0].sub("", await strip_markdown(text, "*")).strip()


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
    lang = language(" ".join(shown))
    weights = [max(1, len(re.findall(r"[aiueo]+", normalize(w, lang).lower()))) + PAUSES.get(w[-1], 0.0) for w in shown]
    total, before, times = sum(weights), 0.0, []
    for w in weights:
        times.append(float(start + (end - start) * before / total))
        before += w
    return times

