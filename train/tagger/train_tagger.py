#!/usr/bin/env python3
"""Fine-tune the text tagger, score it against the rules it replaces, export it for the CPU.

    python train_tagger.py --data out/train.jsonl --out out/model
    python train_tagger.py --data out/train.jsonl --base FacebookAI/xlm-roberta-large --out out/model-large

Data: generate_sentences.mjs export (one {"words": [...], "tags": [...]} per line).

Splits (seeded, written to the report so a run can be repeated):
- unseen: every sentence with one of 15% of the capitalised tokens, held out whole. This is the
  number that matters: the robot meets acronyms nobody labelled.
- random: 5% of the other sentences (tokens the model saw, new sentences).
- val: another 5%, for picking the epoch.

The report compares, on the same sentences:
- rules: function words for the language, acronym_clf.py (table + model) for capitals, as served now;
- tagger: this model alone;
- served order: user lists and the acronym table first, then the tagger, then acronym_clf's guess,
  exactly what speech_text.py does with both installed.

Output dir: model.onnx (int8), tokenizer.json, labels.json, report.json, report.md.
Copy it to the server as models/text_tagger (compose mounts ./models on /models).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "orchestrator"))
import acronym_clf  # noqa: E402
import langid  # noqa: E402
from text_tagger import CAP_TAGS, CAPS, sentence_language  # noqa: E402

LABELS = ["X", "ID", "EN", "JV", "NAME", "WORD", "SPELL_ID", "SPELL_EN"]
L2I = {l: i for i, l in enumerate(LABELS)}


def load(paths: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = " ".join(r["words"])
            if key in seen or len(r["words"]) != len(r["tags"]) or not all(t in L2I for t in r["tags"]):
                continue
            seen.add(key)
            rows.append(r)
    return rows


def split(rows: list[dict], heldout: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    tokens = sorted({w for r in rows for w in r["words"] if CAPS.fullmatch(w)})
    rng.shuffle(tokens)
    held = set(tokens[: int(len(tokens) * heldout)])
    unseen = [r for r in rows if any(w in held for w in r["words"])]
    rest = [r for r in rows if not any(w in held for w in r["words"])]
    rng.shuffle(rest)
    n = max(1, len(rest) // 20)
    return {"unseen": unseen, "random": rest[:n], "val": rest[n:2 * n], "train": rest[2 * n:], "held_tokens": sorted(held)}


# ---------------------------------------------------------------- model

def encode(tokenizer, rows, max_len):
    enc = tokenizer([r["words"] for r in rows], is_split_into_words=True, truncation=True, max_length=max_len)
    labels = []
    for i, r in enumerate(rows):
        prev, lab = None, []
        for wid in enc.word_ids(i):
            lab.append(-100 if wid is None or wid == prev else L2I[r["tags"][wid]])
            prev = wid
        labels.append(lab)
    return enc, labels


def batches(enc, labels, size, shuffle, rng):
    import torch
    order = list(range(len(labels)))
    if shuffle:
        rng.shuffle(order)
    for s in range(0, len(order), size):
        idx = order[s:s + size]
        width = max(len(enc["input_ids"][i]) for i in idx)
        pad = lambda seq, v: seq + [v] * (width - len(seq))  # noqa: E731
        yield idx, {
            "input_ids": torch.tensor([pad(enc["input_ids"][i], 0) for i in idx]),   # pad id set by the caller
            "attention_mask": torch.tensor([pad(enc["attention_mask"][i], 0) for i in idx]),
            "labels": torch.tensor([pad(labels[i], -100) for i in idx]),
        }


def predict(model, tokenizer, rows, max_len, device, size=64):
    import torch
    enc, _ = encode(tokenizer, rows, max_len)
    out = [None] * len(rows)
    model.eval()
    with torch.no_grad():
        for idx, b in batches(enc, [[0]] * len(rows), size, False, None):
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            b["input_ids"][b["attention_mask"] == 0] = tokenizer.pad_token_id
            logits = model(**b).logits.float().cpu()
            for j, i in enumerate(idx):
                tags, prev = ["X"] * len(rows[i]["words"]), None
                for pos, wid in enumerate(enc.word_ids(i)):
                    if wid is not None and wid != prev:
                        tags[wid] = LABELS[int(logits[j, pos].argmax())]
                    prev = wid
                out[i] = tags
    return out


# ---------------------------------------------------------------- scoring

def caps_reading_rules(word: str) -> str:
    p = acronym_clf.predict(word)
    return p[0] if p else ("WORD" if len(word) > 6 else "SPELL_ID")


def caps_reading_served(word: str, tagged: str) -> str:
    known = acronym_clf.known(word)
    if known:
        return known
    if tagged in CAP_TAGS:
        return tagged
    return caps_reading_rules(word)


def score(rows: list[dict], pred: list[list[str]]) -> dict:
    c = Counter()
    for r, p in zip(rows, pred):
        gold_lang = sentence_language(r["tags"])
        if gold_lang:
            text = " ".join(r["words"])
            c["lang_n"] += 1
            c["lang_rules"] += (langid.function_word_language(text) or "id") == gold_lang
            c["lang_tagger"] += (sentence_language(p) or "id") == gold_lang
        for w, g, t in zip(r["words"], r["tags"], p):
            if g in ("ID", "EN", "JV", "NAME"):
                c["word_n"] += 1
                c["word_tagger"] += t == g
            if CAPS.fullmatch(w) and g in CAP_TAGS:
                c["caps_n"] += 1
                c["caps_rules"] += caps_reading_rules(w) == g
                c["caps_tagger"] += t == g
                c["caps_served"] += caps_reading_served(w, t) == g
    pct = lambda k, n: round(c[k] / c[n], 4) if c[n] else None  # noqa: E731
    return {"sentences": len(rows), "sentence_language": {"n": c["lang_n"], "rules": pct("lang_rules", "lang_n"), "tagger": pct("lang_tagger", "lang_n")},
            "word_language": {"n": c["word_n"], "tagger": pct("word_tagger", "word_n")},
            "capitals": {"n": c["caps_n"], "rules": pct("caps_rules", "caps_n"), "tagger": pct("caps_tagger", "caps_n"),
                         "served": pct("caps_served", "caps_n")}}


# ---------------------------------------------------------------- export

def export(model, tokenizer, out: Path, max_len: int) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = model.float().cpu().eval()
    sample = tokenizer([["Halo", "GPU", "nya", "di", "ITS", "."]], is_split_into_words=True, return_tensors="pt")
    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(
        model, (sample["input_ids"], sample["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink()
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, action="append", required=True)
    ap.add_argument("--base", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--heldout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = load(args.data)
    parts = split(rows, args.heldout, args.seed)
    print(f"{len(rows)} sentences: train {len(parts['train'])}, val {len(parts['val'])}, random test {len(parts['random'])}, "
          f"unseen-capitals test {len(parts['unseen'])} ({len(parts['held_tokens'])} tokens held out)")
    print("tags:", dict(Counter(t for r in parts["train"] for t in r["tags"])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(device)

    enc, labels = encode(tokenizer, parts["train"], args.max_len)
    steps = args.epochs * math.ceil(len(labels) / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    best, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _, b in batches(enc, labels, args.batch, True, rng):
            b = {k: v.to(device) for k, v in b.items()}
            b["input_ids"][b["attention_mask"] == 0] = tokenizer.pad_token_id
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        val = score(parts["val"], predict(model, tokenizer, parts["val"], args.max_len, device))
        key = (val["capitals"]["tagger"] or 0) + (val["word_language"]["tagger"] or 0)
        history.append({"epoch": epoch + 1, "loss": round(total, 3), "val": val})
        print(f"epoch {epoch + 1}: loss {total:.2f}, val capitals {val['capitals']['tagger']}, words {val['word_language']['tagger']}")
        if key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_minutes = round((time.time() - t0) / 60, 1)

    result = {"base": args.base, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
              "train_minutes": train_minutes, "sizes": {k: len(v) for k, v in parts.items() if k != "held_tokens"},
              "held_tokens": parts["held_tokens"], "history": history}
    for name in ("unseen", "random"):
        result[name] = score(parts[name], predict(model, tokenizer, parts[name], args.max_len, device))

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_export:
        export(model, tokenizer, args.out, args.max_len)
        (args.out / "labels.json").write_text(json.dumps({"labels": LABELS, "max_len": args.max_len, "base": args.base}, indent=1))
        # The int8 model must say what the trained one says, and fast enough for the voice.
        import text_tagger
        tagger = text_tagger.Tagger(args.out)
        test = parts["unseen"][:300] + parts["random"][:300]
        torch_tags = predict(model, tokenizer, test, args.max_len, device)
        same = sum(a == b for r, t in zip(test, torch_tags) for a, b in zip(t, [x for x, _ in tagger.tag(r["words"])]))
        words = sum(len(r["words"]) for r in test)
        t1 = time.perf_counter()
        for r in test:
            tagger.tag(r["words"])
        result["onnx_int8"] = {"same_tags_as_trained": round(same / max(words, 1), 4),
                               "ms_per_sentence_1_thread": round((time.perf_counter() - t1) * 1000 / max(len(test), 1), 2),
                               "size_mb": round((args.out / "model.onnx").stat().st_size / 2**20, 1)}

    (args.out / "report.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Text tagger: {args.base}", "",
          f"{len(rows)} sentences, seed {args.seed}, {args.epochs} epochs, lr {args.lr}, batch {args.batch}, {train_minutes} min on {device}.", "",
          "| test set | measure | rules (served now) | tagger | served order |", "| --- | --- | --- | --- | --- |"]
    for name in ("unseen", "random"):
        s = result[name]
        md.append(f"| {name} ({s['sentences']}) | sentence language ({s['sentence_language']['n']}) | "
                  f"{s['sentence_language']['rules']} | {s['sentence_language']['tagger']} | {s['sentence_language']['tagger']} |")
        md.append(f"| {name} | capital readings ({s['capitals']['n']}) | {s['capitals']['rules']} | "
                  f"{s['capitals']['tagger']} | {s['capitals']['served']} |")
        md.append(f"| {name} | word language ({s['word_language']['n']}) | - | {s['word_language']['tagger']} | - |")
    if "onnx_int8" in result:
        o = result["onnx_int8"]
        md += ["", f"int8 ONNX: {o['size_mb']} MB, {o['ms_per_sentence_1_thread']} ms per sentence on one CPU thread, "
               f"{o['same_tags_as_trained']:.2%} of tags equal to the trained model."]
    md += ["", "Text-level only: how the voice says the result is a separate listening test (CLAUDE.md section 6)."]
    (args.out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
