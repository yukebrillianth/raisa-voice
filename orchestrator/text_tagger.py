"""Tag every word of a sentence with its language and every capitalised token with its reading.

    ID, EN, JV, NAME        language of a word (NAME: people, places, brands)
    WORD, SPELL_ID, SPELL_EN how a capitalised token is said ("Unesa", "u-gé-ém", "GPU")
    X                        digits and punctuation

One multilingual encoder (XLM-R family, fine-tuned by train/tagger/train_tagger.py) reads the whole
sentence, so "PNJ" in a sentence about polytechnics and "GPU" in one about laptops get different
readings, and "download aplikasinya ya" is still an Indonesian sentence. It runs on the CPU as an
int8 ONNX model, so it takes no VRAM from the LLM and the TTS.

The model directory (TEXT_TAGGER_DIR, default /models/text_tagger) holds model.onnx,
tokenizer.json and labels.json. Without it everything here returns None and speech_text.py uses
its rules: function words for the language, acronym_clf.py for capitals.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("TEXT_TAGGER_DIR", "/models/text_tagger"))
# Words as the tagger sees them. Must match WORD in train/tagger/generate_sentences.mjs.
WORD = re.compile(r"[A-Za-zÀ-ÿ]+|\d+|[^\sA-Za-zÀ-ÿ\d]")
CAPS = re.compile(r"[A-Z]{2,}")
CAP_TAGS = ("WORD", "SPELL_ID", "SPELL_EN")


def split_words(text: str) -> list[str]:
    return WORD.findall(text)


def sentence_language(tags: list[str]) -> str | None:
    """English only when English words outnumber Indonesian and Javanese ones: a code-switched
    sentence ("download aplikasinya ya") is said as Indonesian. None when no word has a language."""
    local = sum(t in ("ID", "JV") for t in tags)
    english = sum(t == "EN" for t in tags)
    if not local and not english:
        return None
    return "en" if english > local else "id"


class Tagger:
    def __init__(self, model_dir: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        meta = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.labels: list[str] = meta["labels"]
        self.max_len = int(meta.get("max_len", 128))
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(self.max_len)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("TEXT_TAGGER_THREADS", "1"))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def tag(self, words: list[str]) -> list[tuple[str, float]]:
        """One (tag, confidence) per word; words cut off by truncation get ("X", 0.0)."""
        if not words:
            return []
        enc = self.tokenizer.encode(words, is_pretokenized=True)
        feed = {"input_ids": self.np.array([enc.ids], dtype=self.np.int64),
                "attention_mask": self.np.array([enc.attention_mask], dtype=self.np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = self.np.zeros_like(feed["input_ids"])
        logits = self.session.run(None, feed)[0][0]
        probs = self.np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        out: list[tuple[str, float]] = [("X", 0.0)] * len(words)
        seen = set()
        for pos, wid in enumerate(enc.word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)   # first sub-word of a word carries its tag, as in training
            best = int(probs[pos].argmax())
            out[wid] = (self.labels[best], float(probs[pos, best]))
        return out


_tagger: Tagger | None = None
_tried = False


def get() -> Tagger | None:
    global _tagger, _tried
    if not _tried:
        _tried = True
        if (MODEL_DIR / "model.onnx").exists():
            try:
                _tagger = Tagger(MODEL_DIR)
                log.info("Text tagger loaded from %s", MODEL_DIR)
            except Exception as exc:  # noqa: BLE001 - no tagger means the rules, never a dead voice
                log.warning("Text tagger not loaded (%s): %s", MODEL_DIR, exc)
    return _tagger


@lru_cache(maxsize=1024)
def analyse(text: str) -> tuple[str | None, dict[str, tuple[str, float]]] | None:
    """(sentence language, {capitalised token: (reading, confidence)}), or None without a model."""
    tagger = get()
    if tagger is None:
        return None
    words = split_words(text)
    tags = tagger.tag(words)
    caps = {}
    for w, (t, c) in zip(words, tags):
        if CAPS.fullmatch(w) and t in CAP_TAGS and c > caps.get(w, ("", -1.0))[1]:
            caps[w] = (t, c)
    return sentence_language([t for t, _ in tags]), caps
