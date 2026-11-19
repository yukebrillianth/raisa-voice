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
- Acronyms in an Indonesian sentence are written out as Indonesian letter names with the taling
  e marked and joined by "-" ("WIB" -> "wé-i-bé", "HDMI" -> "ha-dé-ém-i"): one generic rule, no
  per-acronym list. Measured on the served voice, 8 samples x 2 judges: WIB 8/8 (raw 3/8, "we i be"
  0/8), HDMI 8/8 (raw 1/8), NIK 8/8 (raw 5/8), the others as good as raw (train/v3/v3_report.md).
  Exceptions stay raw: EN_ACRONYMS (English letter names by the user's rule: USB, GPU, PDF...).
  READ_AS_WORD and capitals longer than 6 letters (BANDUNG) are said as a Title Case word.
  English sentences keep acronyms raw.
- FIXED overrides where the raw form reads worse (QRIS "kyuris", units kWh/GHz/kHz) hold in
  both languages.
- Academic titles are said in full ("Prof. Dr. Ir. Budi, S.T., M.Sc." -> "Profesor Doktor
  Insinyur Budi, Sarjana Teknik, Master of Science"). Their dots are dropped so they give no
  sentence-end pause; a degree after the name keeps one dot when it also ends the sentence.
  "Dr." is Doktor and "dr." is Dokter. TITLE_DOTS keeps the sentence splitter off them (bot.py).
  titles.py reads them from a lexicon plus a grammar (composed S./M./A.Md./Sp. degrees, consultant
  parts, sloppy writing, English names in English sentences); see train/titles/README.md.
- A sentence is English when it has English function words and no Indonesian or Javanese ones;
  mixed sentences ("silakan login pakai email") stay Indonesian. A sentence with no function words
  at all ("Specs: 2 GPU, 64 GB RAM.") takes the language of the sentence before it in the same reply
  (voxcpm_tts keeps it). English sentences use "," for thousands and "." for decimals; Rp amounts
  written the Indonesian way ("Rp 150.000") are still thousands.
- Capitals nobody listed go to acronym_clf.py, a small model trained on acronym_data.csv
  (WORD / SPELL_ID / SPELL_EN, 80% on unseen tokens in grouped cross-validation against 43% for the
  old "longer than 6 letters is a word" rule). Below its confidence they are spelled in Indonesian.
- EN_ACRONYMS stay raw in Indonesian sentences, as served and trained. EN_ACRONYM_STYLE=spell writes
  them with English letter names instead ("GPU" -> "ji-pi-yu"); not measured yet, while raw GPU, API,
  PDF are known wrong (0-2/8, CLAUDE.md section 8). Measure before switching it on.
- Ranges "2019-2024", "08.00-10.00", "5-10 jam" are said with "sampai" ("to" in English). A score
  like "3-1" is read as a range too. Dates "19-11-2024" or "19/11/2024" are day-month-year:
  "sembilan belas November dua ribu dua puluh empat", "November nineteenth, twenty twenty-four".
- Model numbers: "RTX 5090" -> "RTX lima puluh sembilan puluh" (pairs, as people say GPU names),
  "ISO 9001" -> digit by digit, letters glued to digits ("ESP32") get a space first.
- SPEECH_LEXICON_PATH points to an optional JSON file with tenant words (read_as_word, en_acronyms,
  id_acronyms, fixed, titles) so a hospital or a telco can add its own names without a code change.
The LoRA v2 "hybrid" training transcripts are this output with QRIS and every acronym left raw.
"""

import json
import logging
import os
import re
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from num2words import num2words

from pipecat.utils.text.transforms.strip_markdown import strip_markdown

import acronym_clf
import alnum
import everyday
import text_tagger
from langid import function_words

log = logging.getLogger(__name__)
EN_ACRONYM_STYLE = os.environ.get("EN_ACRONYM_STYLE", "raw")   # raw | spell
TAGGER_CONF = float(os.environ.get("TEXT_TAGGER_CONF", "0.6"))
# Readings the text tagger gave the capitals of the sentence being normalized.
_CAPS_HINTS: ContextVar[dict] = ContextVar("caps_hints", default={})


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


# Indonesian letter names; é marks the taling e, so "we i be" is not read as English "wi ai bi".
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))
# Said the same way in both languages; picked by ear over the raw acronym (LoRA v2 A/B).
# colok-words read with a k, where "ch" gives the Indonesian c (phoneme judge, 8 samples): colokkan 3/3,
# dicolokkan 0/8 -> 7/8, colokan 5/8 -> 7/8. "colok" and "mencolok" are right as written (8/8).
FIXED = {"QRIS": "kyuris", "kWh": "kilowatt hour", "GHz": "gigahertz", "kHz": "kilohertz",
         "colokkan": "cholokkan", "Colokkan": "Cholokkan", "dicolokkan": "dicholokkan",
         "Dicolokkan": "Dicholokkan", "colokan": "cholokan", "Colokan": "Cholokan",
         # One e was dropped 3/8 written plain, 1/8 as "se-enaknya" (Gemini judge). "seekor" and "keenam" lose an
         # e however they are written (needs training data); "keempat", "keesokan" are right (8/8).
         "seenaknya": "se-enaknya", "Seenaknya": "Se-enaknya"}
# English letter names for EN_ACRONYM_STYLE=spell, written so VoxCPM reads them in an Indonesian sentence.
EN_LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "ei bi si di i ef ji eich ai jei kei el em en ou pi kyu ar es ti yu vi dabelyu eks wai zi".split()))
# Said with English letter names in both languages (the user's rule): sent raw, as the LoRA was trained.
EN_ACRONYMS = set("USB TV VIP NFC GB MB KB TB IT ID QR API GPU CPU SSD AI LED PC PDF".split())
# English sentences: these are spelled with English letter names ("W I B"); other acronyms stay raw.
SPELLED_EN = {"WIB", "WIT"}
ABBREVIATIONS = {"dll": "dan lain-lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
                 "tsb": "tersebut", "yg": "yang", "dgn": "dengan", "utk": "untuk"}
UNITS = {"km": "kilometer", "m": "meter", "cm": "sentimeter", "mm": "milimeter", "kg": "kilogram",
         "g": "gram", "l": "liter", "ml": "mililiter", "km/jam": "kilometer per jam", "ms": "milidetik"}
# Written in capitals but said as a word, not spelled: robot and team names, words, places, people.
READ_AS_WORD = set("""OK RAISA NABBIT MAGE IRIS BIMA LUNA WITA BRIN SIM RAM LAN SKEM EXPO SPARK UNAIR
    HIMATEKTRO TENDON GARUDA NUSANTARA ANDI ARIF BALI BANDUNG DEWI DINA FARHAN JAKARTA JOGJA KARA
    MALANG MEDAN NORA PUTRA RARA RENDI SATRIA SEMARANG SENO SINTA SURABAYA TASYA YOGI ZARA
    UNESA UINSA IKIP PENS UNDIP UNPAD UNHAS UNSOED BPOM INDOSAT MAKASSAR PALEMBANG""".split())
ID_ACRONYMS: set[str] = set()   # always Indonesian letters, even where the model guesses otherwise


def _load_tenant_lexicon() -> None:
    """Words of one deployment (SPEECH_LEXICON_PATH, JSON): read_as_word, en_acronyms, id_acronyms, fixed."""
    path = os.environ.get("SPEECH_LEXICON_PATH", "")
    if not path:
        return
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Ignoring SPEECH_LEXICON_PATH %s: %s", path, exc)
        return
    READ_AS_WORD.update(w.upper() for w in data.get("read_as_word", []))
    EN_ACRONYMS.update(w.upper() for w in data.get("en_acronyms", []))
    ID_ACRONYMS.update(w.upper() for w in data.get("id_acronyms", []))
    FIXED.update(data.get("fixed", {}))


_load_tenant_lexicon()


def _load_respell() -> tuple[dict[str, str], dict[str, str]]:
    """Spellings the served voice picked itself (train/respell/respell_search.py): measured, so they win
    over every rule, in Indonesian sentences only. Missing file: nothing changes."""
    path = Path(os.environ.get("RESPELL_PATH", Path(__file__).resolve().parent / "respell.json"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    return data.get("capitals_id", {}), data.get("words_id", {})


RESPELL_CAPS, RESPELL_WORDS = _load_respell()
_RESPELL_WORD = re.compile(r"\b(" + "|".join(map(re.escape, sorted(RESPELL_WORDS, key=len, reverse=True))) + r")\b",
                           re.I) if RESPELL_WORDS else None
# Titles around names ("Prof. Dr. Ir. Budi, S.T., Ph.D.") are read by titles.py. TITLE_DOTS is
# re-exported for bot.py, which keeps the sentence splitter off a title's dots.
from titles import TITLE_DOTS as _TITLE_DOTS, say_titles  # noqa: E402

# Dots that never end a sentence, for bot.TitleDots: titles, "Jl."/"No. 5"/"e.g.", list numbers.
TITLE_DOTS = re.compile(f"{_TITLE_DOTS.pattern}|{everyday.ABBREV_DOTS.pattern}|{everyday.LIST_NUMBER.pattern}")


SCALES = r"(?:\s?(ribu|juta|miliar|triliun))?"
NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?"  # Indonesian: 1.500.000,75

FRONT = [
    # Table rows and their |---| separators: tables go to the screen, never to the voice.
    (re.compile(r"^\s*\|.*\|\s*$\n?", re.M), lambda m: ""),
    (re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]"), lambda m: ""),
    # Markdown the splitter cut in half ("**Kol." in one sentence, "...**" in the next).
    (re.compile(r"\*\*|__|`"), lambda m: ""),
    # Longest first, so a tenant's "IM3 Ooredoo" wins over "IM3".
    (re.compile(r"\b(" + "|".join(map(re.escape, sorted(FIXED, key=len, reverse=True))) + r")\b"),
     lambda m: FIXED[m.group(1)]),
    # "ESP32", "RTX5090": a space first, then the number and the capitals are read as usual.
    (re.compile(r"\b([A-Z]{2,})(\d{2,})\b"),
     lambda m: m.group(0) if m.group(0) in alnum.TABLE else f"{m.group(1)} {m.group(2)}"),
]


def spell_id(word: str) -> str:
    return "-".join(LETTERS[c] for c in word)


def spell_en(word: str) -> str:
    return "-".join(EN_LETTERS[c] for c in word)


def say_capitals(word: str, lang: str) -> str:
    if word in READ_AS_WORD:
        return word.capitalize()
    if lang == "en":
        return " ".join(word) if word in SPELLED_EN else word
    if word in RESPELL_CAPS:
        return RESPELL_CAPS[word]
    if word in EN_ACRONYMS:
        return spell_en(word) if EN_ACRONYM_STYLE == "spell" else word
    if word in ID_ACRONYMS:
        return spell_id(word)
    guess = acronym_clf.known(word)
    if guess is None:
        tagged, conf = _CAPS_HINTS.get().get(word, (None, 0.0))
        guess = tagged if conf >= TAGGER_CONF else acronym_clf.classify(word)
    if guess == "WORD" or (guess is None and len(word) > 6):
        return word.capitalize()
    if guess == "SPELL_EN":
        return spell_en(word) if EN_ACRONYM_STYLE == "spell" else word
    return spell_id(word)


def say_capitals_match(m: re.Match, lang: str) -> str:
    """'KTPnya' -> 'ka-té-pé-nya': the suffix stays a separate part of the spelling."""
    said = say_capitals(m.group(1), lang)
    return said + ("-" if m.group(2) and said != m.group(1) and said[-1].islower() and "-" in said else "") + (m.group(2) or "")


def model_pairs(d: str, w) -> str:
    """'5090' -> 'lima puluh sembilan puluh', '7900' -> 'tujuh puluh sembilan ratus'."""
    hundred = "ratus" if w is words else "hundred"
    tail = hundred if d[2:] == "00" else w(int(d[2:])) if d[2] != "0" else f"{'nol' if w is words else 'oh'} {w(int(d[3]))}"
    return f"{w(int(d[:2]))} {tail}"


def year_en(n: int) -> str:
    return num2words(n, lang="en", to="year")


def is_year(s: str) -> bool:
    return bool(re.fullmatch(r"1[1-9]\d\d|20\d\d", s))


# A score, not a range: "menang 3-1", "skor akhir 2-2", "won 3-1".
SCORE_WORDS = re.compile(r"\b(skor|skornya|menang|kalah|unggul|tertinggal|imbang|hasil akhir|score|won|lost|beat|beats|"
                         r"leads?|drew|draw)\b[^.!?]{0,25}$", re.I)


def say_range(m: re.Match, lang: str) -> str:
    a, b = m.group(1), m.group(2)
    if SCORE_WORDS.search(m.string[:m.start()]) and len(a) <= 2 and len(b) <= 2:
        return f"{a} to {b}" if lang == "en" else f"{a} lawan {b}"
    if lang == "en":
        if is_year(a) and is_year(b):
            return f"{year_en(int(a))} to {year_en(int(b))}"
        return f"{a} to {b}"
    return f"{a} sampai {b}"


BULAN = "Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember".split()
EN_MONTHS = "January February March April May June July August September October November December".split()
# "19-11-2024" or "19/11/2024": day, month, year.
DATE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](\d{4})\b")


def say_date(m: re.Match, lang: str) -> str:
    day, mon, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if lang == "en":
        return f"{EN_MONTHS[mon - 1]} {num2words(day, lang='en', to='ordinal')}, {year_en(year)}"
    return f"{words(day)} {BULAN[mon - 1]} {words(year)}"


# "2019-2024", "08.00-10.00", "5-10": a dash between two numbers. Not dates ("19-11-2024"), not "ke-12".
RANGE = re.compile(r"(?<![\w.,:-])(\d[\d.,:]*\d|\d)\s?[–-]\s?(\d[\d.,:]*\d|\d)(?![\w-])")
MODEL_PAIRS = re.compile(r"\b(RTX|GTX|RX)\s(\d{4})\b")
MODEL_DIGITS = re.compile(r"\b(ISO|IEC|SNI|IEEE)\s(\d{3,})\b")
CAPITALS = re.compile(r"\b([A-Z]{2,})(nya)?\b")


RULES = [
    (re.compile(r"\s?°C"), lambda m: " derajat Celsius"),
    (re.compile(r"\s?°"), lambda m: " derajat"),
    (re.compile(r"\s?m²"), lambda m: " meter persegi"),
    (re.compile(r"(\+62|\b0)[\d\- ]{8,}\d"), lambda m: ("plus enam dua " if m.group(1) == "+62" else "nol ")
     + digits(m.group(0)[len(m.group(1)):])),
    (DATE, lambda m: say_date(m, "id")),
    (RANGE, lambda m: say_range(m, "id")),
    (re.compile(r"\s?\+\s?(?=\d)"), lambda m: " plus "),   # "UTC+7"; phone numbers are read above
    (re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b(?![.,]?\d)"), lambda m: clock(m.group(1), m.group(2))),
    (MODEL_PAIRS, lambda m: f"{m.group(1)} {model_pairs(m.group(2), words)}"),
    (MODEL_DIGITS, lambda m: f"{m.group(1)} {digits(m.group(2))}"),
    (re.compile(rf"\bRp\.?\s?{NUM}{SCALES}(?:,-|,00)?"),
     lambda m: " ".join(filter(None, [number(m.group(1), m.group(2)), m.group(3), "rupiah"]))),
    (re.compile(rf"{NUM}\s?%"), lambda m: f"{number(m.group(1), m.group(2))} persen"),
    (re.compile(rf"\b{NUM}\s?(km/jam|km|cm|mm|kg|ml|ms|m|g|l)\b"),
     lambda m: f"{number(m.group(1), m.group(2))} {UNITS[m.group(3)]}"),
    (re.compile(r"\bke-(\d+)\b"), lambda m: "ke" + words(int(m.group(1)))),
    (re.compile(rf"\b{NUM}\b"), lambda m: number(m.group(1), m.group(2))),
    # The dot after "dll." stays: it usually also ends the sentence, which gives the pause.
    (re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b"), lambda m: ABBREVIATIONS[m.group(1)]),
    (CAPITALS, lambda m: say_capitals_match(m, "id")),
]
RULES = everyday.early("id") + RULES[:-1] + everyday.late("id") + RULES[-1:]


def en_words(n: int) -> str:
    return num2words(n, lang="en")


def en_number(whole: str, frac: str | None = None) -> str:
    text = en_words(int(whole.replace(",", "")))
    return f"{text} point {' '.join(en_words(int(d)) for d in frac)}" if frac else text


EN_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"  # English: 1,500,000.75
EN_UNITS = {"km": "kilometer", "m": "meter", "cm": "centimeter", "kg": "kilogram", "g": "gram",
            "l": "liter", "ml": "milliliter", "km/h": "kilometers per hour", "km/jam": "kilometers per hour",
            "ms": "millisecond"}


def en_quantity(m: re.Match) -> str:
    unit = EN_UNITS[m.group(3)]
    plural = "" if unit.endswith("hour") or (m.group(1) == "1" and not m.group(2)) else "s"
    return f"{en_number(m.group(1), m.group(2))} {unit}{plural}"


MONTHS = ("January February March April May June July August September October November December "
          "Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec").split()
MONTH = r"(" + "|".join(MONTHS) + r")\.?"
FULL_MONTH = dict(zip("Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec".split(),
                      "January February March April June July August September September October November December".split()))


def month(m: str) -> str:
    return FULL_MONTH.get(m, m)


def ordinal(n: str) -> str:
    return num2words(int(n), lang="en", to="ordinal")


EN_RULES = [
    (re.compile(r"\s?°C"), lambda m: " degrees Celsius"),
    (re.compile(r"\s?°"), lambda m: " degrees"),
    (re.compile(r"\s?m²"), lambda m: " square meters"),
    # Rupiah written the Indonesian way inside an English sentence: "Rp 150.000" is thousands.
    (re.compile(r"\b(?:Rp\.?|IDR)\s?(\d{1,3}(?:\.\d{3})+)(?:,(\d+))?(?:,-)?"),
     lambda m: f"{en_number(m.group(1).replace('.', ''), m.group(2))} rupiah"),
    (re.compile(rf"\b(?:Rp\.?|IDR)\s?{EN_NUM}"), lambda m: f"{en_number(m.group(1), m.group(2))} rupiah"),
    (re.compile(r"\b\d{1,3}(?:\.\d{3}){2,}\b"), lambda m: en_words(int(m.group(0).replace(".", "")))),
    # "April 30, 2024" -> "April thirtieth, twenty twenty-four"; "30 April 2024" -> "the thirtieth of April, ..."
    (re.compile(rf"\b{MONTH}\s(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s(\d{{4}}))?\b"),
     lambda m: f"{month(m.group(1))} {ordinal(m.group(2))}" + (f", {year_en(int(m.group(3)))}" if m.group(3) else "")),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s{MONTH}(?:,?\s(\d{{4}}))?\b"),
     lambda m: f"the {ordinal(m.group(1))} of {month(m.group(2))}" + (f", {year_en(int(m.group(3)))}" if m.group(3) else "")),
    (DATE, lambda m: say_date(m, "en")),
    (RANGE, lambda m: say_range(m, "en")),
    (re.compile(r"\s?\+\s?(?=\d)"), lambda m: " plus "),
    (re.compile(r"\b(from|between)\s(1[1-9]\d\d|20\d\d)\s(to|and)\s(1[1-9]\d\d|20\d\d)\b", re.I),
     lambda m: f"{m.group(1)} {year_en(int(m.group(2)))} {m.group(3)} {year_en(int(m.group(4)))}"),
    (re.compile(r"\b(in|since|from|until|till|by|year|before|after|between|around|of)\s(1[1-9]\d\d|20\d\d)\b", re.I),
     lambda m: f"{m.group(1)} {year_en(int(m.group(2)))}"),
    (MODEL_PAIRS, lambda m: f"{m.group(1)} {model_pairs(m.group(2), en_words)}"),
    (MODEL_DIGITS, lambda m: f"{m.group(1)} {' '.join(en_words(int(d)) for d in m.group(2))}"),
    (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b"),
     lambda m: en_words(int(m.group(1))) + ("" if m.group(2) == "00" else
               (" oh " if m.group(2)[0] == "0" else " ") + en_words(int(m.group(2))))),
    (re.compile(rf"{EN_NUM}\s?%"), lambda m: f"{en_number(m.group(1), m.group(2))} percent"),
    (re.compile(rf"\b{EN_NUM}\s?(km/h|km/jam|km|cm|kg|ml|ms|m|g|l)(?!\w)"), en_quantity),
    (re.compile(rf"\b{EN_NUM}\b"), lambda m: en_number(m.group(1), m.group(2))),
    (re.compile(r"\b[A-Z]{2,}\b"), lambda m: say_capitals(m.group(0), "en")),
]
EN_RULES = everyday.early("en") + EN_RULES[:-1] + everyday.late("en") + EN_RULES[-1:]


def language_evidence(text: str) -> str | None:
    """'id' or 'en', None when nothing in the sentence says ("Specs: 2 GPU.").

    Function words of one language only ("the", "is" / "yang", "ya") decide: they are almost never
    wrong. The text tagger decides the rest, sentences with none ("Great choice!") or with both
    ("I'm at the lobby ya"), so a weak model can never overturn the easy cases."""
    local, english = function_words(text)
    if local != english:
        return "id" if local else "en"
    tagged = text_tagger.analyse(text)
    if tagged is not None and tagged[0]:
        return tagged[0]
    return "id" if local else None


def language(text: str, before: str | None = None) -> str:
    """Language of a sentence; `before` is the previous sentence's, used when this one has no clue."""
    return language_evidence(text) or before or "id"


def normalize(text: str, lang: str | None = None, tagger: bool = True) -> str:
    """Spoken form of one sentence; `lang` ("id"/"en") overrides detection, e.g. for single words.
    `tagger=False` skips the text tagger (word_times normalizes word by word, after the audio)."""
    alnum.reset()
    tagged = text_tagger.analyse(text) if tagger else None
    _CAPS_HINTS.set(tagged[1] if tagged else {})
    lang = lang or language(text)
    rules = EN_RULES if lang == "en" else RULES
    for pattern, replace in FRONT:
        text = pattern.sub(replace, text)
    text = say_titles(text, lang)
    for pattern, replace in rules:
        text = pattern.sub(replace, text)
    if _RESPELL_WORD and lang != "en":   # last, so no rule touches a "{S AH0 ...}" respelling
        text = _RESPELL_WORD.sub(lambda m: RESPELL_WORDS.get(m.group(1).lower(), m.group(1)), text)
    return alnum.restore(re.sub(r"[ \t]{2,}", " ", text))


async def screen_text(text: str) -> str:
    """What the robot says, as shown to the visitor: no markdown, no table rows, digits kept."""
    return FRONT[0][0].sub("", await strip_markdown(text, "*")).strip()


# Markdown goes before run_tts; the Indonesian rewrite happens inside run_tts (normalize), after
# the sentence's screen text is sent, so the screen keeps "Rp150.000" while the voice says the words.
SPEECH_TRANSFORMS = [("*", strip_markdown)]

# Extra syllables of silence a word's trailing mark adds before the next word.
PAUSES = {",": 1.0, ";": 1.5, ":": 1.5, ".": 2.0, "?": 2.0, "!": 2.0}


def word_times(shown: list[str], audio: bytes, sample_rate: int, lang: str | None = None) -> list[float]:
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
    lang = lang or language(" ".join(shown))
    said = [normalize(w, lang, tagger=False) for w in shown]
    # The pause comes from the spoken form: "Prof." is said "Profesor", with no stop after it.
    weights = [max(1, len(re.findall(r"[aiueoéè]+", w.lower()))) + PAUSES.get(w[-1:], 0.0) for w in said]
    total, before, times = sum(weights), 0.0, []
    for w in weights:
        times.append(float(start + (end - start) * before / total))
        before += w
    return times

