"""How speech_text.py uses the text tagger, with a fake tagger standing in for the model."""

import pytest

import speech_text
import text_tagger
from speech_text import language, normalize


@pytest.fixture
def fake(monkeypatch):
    answers = {}
    monkeypatch.setattr(text_tagger, "analyse", lambda text: answers.get(text, (None, {})))
    return answers


def test_tagger_decides_when_there_are_no_function_words(fake):
    fake["Great choice: 2 GPU, 64 GB RAM."] = ("en", {})
    assert language("Great choice: 2 GPU, 64 GB RAM.") == "en"
    assert normalize("Great choice: 2 GPU, 64 GB RAM.") == "Great choice: two GPU, sixty-four GB Ram."   # RAM is read as a word (user rule)


def test_function_words_of_one_language_beat_the_tagger(fake):
    fake["It is 25% off."] = ("id", {})
    assert language("It is 25% off.") == "en"
    fake["Ini 2 GPU ya."] = ("en", {})
    assert language("Ini 2 GPU ya.") == "id"


def test_mixed_function_words_go_to_the_tagger(fake):
    text = "I'm at the lobby ya."
    fake[text] = ("en", {})
    assert language(text) == "en"
    fake[text] = ("id", {})
    assert language(text) == "id"


def test_capital_reading_from_context(fake):
    text = "Kampus PNJ ada di Depok."
    fake[text] = ("id", {"PNJ": ("WORD", 0.9)})
    assert normalize(text) == "Kampus Pnj ada di Depok."
    fake[text] = ("id", {"PNJ": ("SPELL_ID", 0.9)})
    assert normalize(text) == "Kampus pé-én-jé ada di Depok."


def test_unsure_tagger_falls_back(fake, monkeypatch):
    text = "Kampus PNJ ada di Depok."
    fake[text] = ("id", {"PNJ": ("WORD", 0.3)})
    monkeypatch.setattr(speech_text.acronym_clf, "classify", lambda w: None)
    assert normalize(text) == "Kampus pé-én-jé ada di Depok."


def test_labelled_tokens_and_user_lists_beat_the_tagger(fake, monkeypatch):
    monkeypatch.setattr(speech_text.acronym_clf, "known", lambda w: {"NIK": "SPELL_ID"}.get(w))
    text = "Isi NIK dan GPU di sini."
    fake[text] = ("id", {"NIK": ("WORD", 0.99), "GPU": ("SPELL_ID", 0.99)})
    assert normalize(text) == "Isi én-i-ka dan GPU di sini."


def test_words_split_like_the_generator():
    assert text_tagger.split_words("GPU-nya RTX5090, Rp150.000!") == \
        ["GPU", "-", "nya", "RTX", "5090", ",", "Rp", "150", ".", "000", "!"]


def test_sentence_language_counts_words():
    assert text_tagger.sentence_language(["ID", "EN", "X"]) == "id"     # code-switched: Indonesian
    assert text_tagger.sentence_language(["EN", "EN", "ID"]) == "en"
    assert text_tagger.sentence_language(["NAME", "X", "SPELL_EN"]) is None
