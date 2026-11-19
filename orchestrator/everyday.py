"""The small things an LLM writes that a voice cannot read as they are.

URLs and e-mail addresses, "Jl. Raya ITS No. 5", "2.5 GHz" in an Indonesian sentence, "3x sehari",
"1990-an", "S1/S2/D3", "5G", "$20", fractions, Roman numerals after "Bab" or "Perang Dunia",
"±", "&", "rektor/direktur", "e.g." and "vs." in English, list numbers at the start of a line.

speech_text.normalize runs EARLY before its number rules and LATE right before the capitals rule.
ABBREV_DOTS and LIST_NUMBER tell bot.TitleDots which dots never end a sentence.
"""

from __future__ import annotations

import re
from string import ascii_uppercase

from num2words import num2words

ID_LETTERS = dict(zip(ascii_uppercase, "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))


def w(n: int, lang: str) -> str:
    return num2words(n, lang="en" if lang == "en" else "id")


# ---------------------------------------------------------------- abbreviations with a dot

# written -> (Indonesian, English, may end a sentence). Case matters: "No." before a number only.
ABBREV = {
    "Jl.": ("Jalan", "Jalan", False), "Jln.": ("Jalan", "Jalan", False), "Gg.": ("Gang", "Gang", False),
    "Kec.": ("Kecamatan", "District", False), "Kab.": ("Kabupaten", "Regency", False),
    "Kel.": ("Kelurahan", "Kelurahan", False), "Prov.": ("Provinsi", "Province", False),
    "Telp.": ("telepon", "phone", False), "Tlp.": ("telepon", "phone", False),
    "Hlm.": ("halaman", "page", False), "hlm.": ("halaman", "page", False),
    "s.d.": ("sampai dengan", "until", False), "s/d": ("sampai dengan", "until", False),
    "a.n.": ("atas nama", "on behalf of", False), "u.p.": ("untuk perhatian", "attention", False),
    "dkk.": ("dan kawan-kawan", "and colleagues", True), "vs.": ("versus", "versus", False),
    "e.g.": ("misalnya", "for example", False), "i.e.": ("yaitu", "that is", False),
    "etc.": ("dan lain-lain", "et cetera", True), "approx.": ("sekitar", "approximately", False),
    "Dept.": ("Departemen", "Department", False), "Ltd.": ("Limited", "Limited", True),
}
_ABBREV = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, sorted(ABBREV, key=len, reverse=True))) + r")(?![\w])")
_NUMBER_NO = re.compile(r"\b[Nn]o\.\s?(?=\d)")
ABBREV_DOTS = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, ABBREV)) + r")|\b[Nn]o\.(?=\s?\d)")
# "5." at the start of a line is a list number, not the end of a sentence.
LIST_NUMBER = re.compile(r"(?:^|\n)[ \t]*[*_#>•-]*[ \t]*\d{1,2}[.)](?=[ \t*_]|\n|$)")


def _abbrev(m: re.Match, lang: str) -> str:
    id_, en, ends = ABBREV[m.group(1)]
    said = en if lang == "en" else id_
    rest = m.string[m.end():]
    # "... dkk." at the end of a sentence keeps its dot as the sentence's own
    if ends and m.group(1).endswith(".") and (not rest.strip() or re.match(r"\s+[A-Z]", rest)):
        said += "."
    return said


# ---------------------------------------------------------------- web

TLD_WORDS = {"com", "net", "org", "edu", "gov", "app", "dev", "info", "io", "ai", "co"}
DOMAIN = r"(?:[a-z0-9-]+\.)+(?:id|com|net|org|edu|gov|io|ai|app|dev|info|co|uk|sg|my|au|jp|go|ac|sch|or|web)\b"
_URL = re.compile(rf"\b(?:https?://)?(?:www\.)?({DOMAIN})(?:/[^\s)\]]*)?", re.I)
_EMAIL = re.compile(rf"\b([A-Za-z0-9._%+-]+)@({DOMAIN})", re.I)


def say_domain(domain: str, lang: str) -> str:
    """'its.ac.id' -> 'i-té-és titik a-cé titik i-dé'; English: 'its dot ac dot id'."""
    parts = domain.lower().split(".")
    if lang == "en":
        return " dot ".join(parts)
    said = [p if p in TLD_WORDS or len(p) > 3 or not p.isalpha() else "-".join(ID_LETTERS[c.upper()] for c in p)
            for p in parts]
    return " titik ".join(said)


def _email(m: re.Match, lang: str) -> str:
    user = m.group(1).replace(".", " dot " if lang == "en" else " titik ")
    return f"{user} {'at' if lang == 'en' else 'et'} {say_domain(m.group(2), lang)}"


# ---------------------------------------------------------------- numbers in odd places

def _fraction(m: re.Match, lang: str) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    if (a, b) == (24, 7):
        return "twenty-four seven" if lang == "en" else "dua puluh empat jam setiap hari"
    if lang == "en":
        special = {(1, 2): "one half", (1, 4): "one quarter", (3, 4): "three quarters"}
        return special.get((a, b), f"{w(a, lang)} over {w(b, lang)}")
    special = {(1, 2): "setengah", (1, 4): "seperempat"}
    return special.get((a, b), f"{w(a, lang)} per {w(b, lang)}")


def _digit_letter(m: re.Match, lang: str) -> str:
    """'5G' -> 'lima ji', '4K' -> 'empat ka', '3D' -> 'tiga dé'."""
    n, letter = int(m.group(1)), m.group(2)
    if lang == "en":
        return f"{w(n, lang)} {letter}"
    return f"{w(n, lang)} {'ji' if letter == 'G' else ID_LETTERS[letter]}"


ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman(s: str) -> int | None:
    if not s or not ROMAN.match(s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = VALUES[a]
        total += -v if b != " " and VALUES[b] > v else v
    return total


ROMAN_ID_COUNT = r"Bab|Pasal|Jilid|Tahap|Sesi|Gelombang|Angkatan|Kelas|Semester|Lantai|[Aa]bad|Bagian|Seri|Golongan|Pelita|Kodam"
ROMAN_ID_ORDER = r"Perang Dunia|Hamengkubuwono|Hamengku Buwono|Pakubuwono|Paku Buwono|Pakualam|Paku Alam|Mangkunegara|Paus [A-Z][a-z]+"
ROMAN_EN = r"World War|Chapter|Part|Section|Phase|Grade|Level|Season|Volume|Book|Act|Article|Class|Stage"


def _roman(m: re.Match, lang: str, order: bool) -> str:
    n = roman(m.group(2))
    if n is None:
        return m.group(0)
    if order and lang != "en":
        return f"{m.group(1)} {'pertama' if n == 1 else 'ke' + w(n, lang)}"
    return f"{m.group(1)} {w(n, lang)}"


# ---------------------------------------------------------------- rule lists

def early(lang: str) -> list:
    en = lang == "en"
    rules = [
        # "1. Universitas Airlangga" at the start of a line: "satu, Universitas Airlangga"
        (re.compile(r"^(\s*)(\d{1,2})[.)]\s+", re.M), lambda m: f"{m.group(1)}{w(int(m.group(2)), lang)}, "),
        (re.compile(r"\s*•\s*"), lambda m: ", "),
        (re.compile(r"\s+[—–]\s+|\s*—\s*"), lambda m: ", "),
        (_EMAIL, lambda m: _email(m, lang)),
        (_URL, lambda m: say_domain(m.group(1), lang)),   # the path is for the screen, not the voice
        (_NUMBER_NO, lambda m: "number " if en else "nomor "),
        (_ABBREV, lambda m: _abbrev(m, lang)),
        # versions: "3.1.4" -> "tiga titik satu titik empat"
        (re.compile(r"\bv?(\d+)\.(\d+)\.(\d+)\b"),
         lambda m: (" point " if en else " titik ").join(w(int(x), lang) for x in m.groups())),
        (re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])"), lambda m: _fraction(m, lang)),
        # "$20", "US$ 5", "USD 10", "€15": the number stays digits for the number rules after
        (re.compile(r"(?:US\$|\$|USD\s?)(\d[\d.,]*)(\s?(?:ribu|juta|miliar|triliun|thousand|million|billion))?"),
         lambda m: f"{m.group(1)}{m.group(2) or ''} {'dollars' if en else 'dolar'}"),
        (re.compile(r"(?:€|EUR\s?)(\d[\d.,]*)(\s?(?:ribu|juta|miliar|thousand|million|billion))?"),
         lambda m: f"{m.group(1)}{m.group(2) or ''} euro"),
        (re.compile(r"\s?±\s?"), lambda m: " plus or minus " if en else " kurang lebih "),
        (re.compile(r"[~≈]\s?(?=\d)"), lambda m: "about " if en else "sekitar "),
        (re.compile(r"\s&\s"), lambda m: " and " if en else " dan "),
        (re.compile(r"\b(\d+)\s?[x×]\s?(\d+)\b"), lambda m: f"{m.group(1)} {'by' if en else 'kali'} {m.group(2)}"),
        (re.compile(r"\b(\d+)\s?[x×](?=[\s.,;!?)]|$)"), lambda m: f"{m.group(1)} {'times' if en else 'kali'}"),
        (re.compile(r"\b(\d)([A-Z])\b"), lambda m: _digit_letter(m, lang)),
        (re.compile(rf"\b({ROMAN_EN})\s+([IVXLCDM]{{1,7}})\b"), lambda m: _roman(m, "en", False)) if en else
        (re.compile(rf"\b({ROMAN_ID_ORDER})\s+([IVXLCDM]{{1,7}})\b"), lambda m: _roman(m, lang, True)),
    ]
    if not en:
        rules += [
            (re.compile(rf"\b({ROMAN_ID_COUNT})\s+([IVXLCDM]{{1,7}})\b"), lambda m: _roman(m, lang, False)),
            # "1990-an" -> "seribu sembilan ratus sembilan puluhan"
            (re.compile(r"\b(\d+)-an\b"), lambda m: w(int(m.group(1)), lang) + "an"),
            # "S1", "S2", "D3": degree levels
            (re.compile(r"\b([SD])([1-3])\b"), lambda m: f"{ID_LETTERS[m.group(1)]} {w(int(m.group(2)), lang)}"),
            # "2.5 GHz" written the English way in an Indonesian sentence: a decimal, not 2500.
            (re.compile(r"(?<![\d.,])(\d+)\.(\d)(?![\d.,])"), lambda m: f"{m.group(1)},{m.group(2)}"),
            (re.compile(r"(?<![\d.,])(\d+)\.(\d{2})(?=\s?(?:GHz|MHz|kHz|Hz|GB|MB|TB|%|inci|inch|kg|km|cm|mm|m)\b)"),
             lambda m: f"{m.group(1)},{m.group(2)}"),
        ]
    return rules


def late(lang: str) -> list:
    """After numbers and units ("km/jam" is gone by then), before capitals."""
    en = lang == "en"
    return [
        (re.compile(r"\bdan/atau\b"), lambda m: "dan atau"),
        (re.compile(r"\band/or\b"), lambda m: "and or"),
        (re.compile(r"\b([A-Za-z]{2,})/([A-Za-z]{2,})\b"), lambda m: f"{m.group(1)} {'or' if en else 'atau'} {m.group(2)}"),
    ]
