#!/usr/bin/env python3
"""Pipeline check without an LLM or a download: template sentences and a 2-layer model.

    python smoke.py --out /tmp/tagger-smoke
    python train_tagger.py --data /tmp/tagger-smoke/data.jsonl --base /tmp/tagger-smoke/tiny-base \\
        --out /tmp/tagger-smoke/model --epochs 3 --batch 64

The numbers it prints mean nothing about the real model: the templates are easy and few. It only
proves that data -> training -> int8 ONNX -> orchestrator/text_tagger.py works end to end.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "orchestrator"))
from text_tagger import split_words  # noqa: E402

TEMPLATES = [
    ("Silakan bawa {CAP} ke lantai {N} ya .", "ID ID CAP ID ID X ID X"),
    ("Kantor {CAP} buka pukul {N} sampai {N} .", "ID CAP ID ID X ID X X"),
    ("Kamu bisa download {CAP} - nya di sini .", "ID ID EN CAP X ID ID ID X"),
    ("Please bring your {CAP} to floor {N} .", "EN EN EN CAP EN EN X X"),
    ("The {CAP} office opens at {N} .", "EN CAP EN EN EN X X"),
    ("Monggo pinarak , {CAP} wonten ing lantai {N} nggih .", "JV JV X CAP JV JV ID X JV X"),
    ("Meeting sama {NAME} soal {CAP} jadi besok ya .", "EN ID NAME ID CAP ID ID ID X"),
    ("{NAME} dari {CAP} akan datang .", "NAME ID CAP ID ID X"),
    ("{NAME} from {CAP} will join us .", "NAME EN CAP EN EN EN X"),
]
NAMES = "Budi Siti Bambang Dewi Surabaya Malang Raisa Joko Andi".split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    with (ROOT / "orchestrator" / "acronym_data.csv").open(encoding="utf-8") as f:
        caps = [(r["token"], r["label"]) for r in csv.DictReader(f)]

    rows = []
    for _ in range(args.n):
        text, tags = rng.choice(TEMPLATES)
        token, reading = rng.choice(caps)
        words, out = [], []
        for w, t in zip(text.split(), tags.split()):
            if w == "{CAP}":
                words.append(token); out.append(reading)
            elif w == "{NAME}":
                words.append(rng.choice(NAMES)); out.append("NAME")
            elif w == "{N}":
                words.append(str(rng.randint(1, 30))); out.append("X")
            else:
                words.append(w); out.append(t)
        assert split_words(" ".join(words)) == words, words
        rows.append({"text": " ".join(words), "words": words, "tags": out})
    with (args.out / "data.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator((r["text"] for r in rows), trainers.WordPieceTrainer(vocab_size=2000, special_tokens=special))
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]", model_max_length=128)
    base = args.out / "tiny-base"
    fast.save_pretrained(base)
    BertModel(BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128, max_position_embeddings=160)).save_pretrained(base)
    print(f"{len(rows)} sentences -> {args.out / 'data.jsonl'}, tiny model -> {base}")


if __name__ == "__main__":
    main()
