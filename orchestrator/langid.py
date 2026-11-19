"""Sentence language from function words: the fallback when the text tagger model is not there.

Kept apart from speech_text.py so the tagger's training script can score this baseline without
importing Pipecat.
"""

import re

ID_WORDS = set("""yang dan di ke dari ini itu untuk dengan saya anda kamu ada tidak akan bisa sudah
    silakan mohon ya adalah pada atau juga kami kita apa mana berapa sampai pukul jam dalam
    selamat terima kasih tolong sedang belum lagi""".split())
JV_WORDS = set("""aku kowe sampeyan panjenengan saka karo ning neng ora wis durung piye opo apa sapa
    iki iku kuwi monggo matur nuwun sugeng rawuh enggih nggih mboten""".split())
EN_WORDS = set("""the is are you your to of and in on at for with please this that it we our my
    will can be have has what where how thank welcome yes here there from by me i an a
    sure okay ok sorry certainly alright they he she his her them their was were been do does did not but
    or if than then when who which would should could about into after before because there's it's
    i'm you're we're let's won lost visited""".split())


def function_words(text: str) -> tuple[bool, bool]:
    """(has Indonesian or Javanese function words, has English ones)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return (any(t in ID_WORDS or t in JV_WORDS for t in tokens), any(t in EN_WORDS for t in tokens))


def function_word_language(text: str) -> str | None:
    """'id' or 'en', None when the sentence has no function word at all. Both kinds: Indonesian
    (a code-switched sentence is said as Indonesian)."""
    local, english = function_words(text)
    return "id" if local else "en" if english else None
