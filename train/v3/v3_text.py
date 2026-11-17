"""LoRA v3 transcripts = the text the robot sends to a v3 voice: the generic normalizer rules only.

Numbers, Rp, times, units, academic titles and Title Case of capitalised names are spelled out as
orchestrator/speech_text.py does; every acronym stays raw (no QRIS -> "kyuris", no WIB -> "we i be")
and there is no word lexicon (colokkan, seenaknya). v3 learns those readings from data.
Language is decided per sentence, as the robot's TTS gets it.
"""
import re

import speech_text as st

UNITS = {k: st.FIXED[k] for k in ("kWh", "GHz", "kHz")}
# speech_text.FRONT = [table rows, emoji, FIXED overrides, titles]: the overrides are swapped for units only.
FRONT = st.FRONT[:2] + [(re.compile(r"\b(kWh|GHz|kHz)\b"), lambda m: UNITS[m.group(1)]), st.FRONT[3]]


def keep_acronym(m: re.Match) -> str:
    w = m.group(0)
    return w.capitalize() if w in st.READ_AS_WORD or len(w) > 6 else w


def one(text: str, lang: str | None = None) -> str:
    rules = st.EN_RULES if (lang or st.language(text)) == "en" else st.RULES
    for pattern, replace in FRONT + rules[:-1] + [(rules[-1][0], keep_acronym)]:
        text = pattern.sub(replace, text)
    return text


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def v3(text: str) -> str:
    return " ".join(one(s) for s in sentences(text))


def served_v2(text: str) -> str:
    """What the served v2 path sends for the same text (speech_text.normalize, one sentence at a time)."""
    return " ".join(st.normalize(s) for s in sentences(text))


assert v3("Bayar pakai QRIS Rp 45.000 pukul 09.00 WIB di ITS, kak ANDI.") == \
    "Bayar pakai QRIS empat puluh lima ribu rupiah pukul sembilan WIB di ITS, kak Andi."
assert v3("Colokkan kabel seenaknya, 3 kWh. Welcome to ITS at 07:15.") == \
    "Colokkan kabel seenaknya, tiga kilowatt hour. Welcome to ITS at seven fifteen."
assert v3("Tim MAGE dan IRIS, WITA.") == "Tim Mage dan Iris, Wita."
assert served_v2("Bayar pakai QRIS pukul 09.00 WIB.") == "Bayar pakai kyuris pukul sembilan wé-i-bé."
