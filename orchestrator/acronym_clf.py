"""Guess how an unseen ALL-CAPS token is read: WORD ("Unesa"), SPELL_ID ("u-gé-ém") or SPELL_EN ("ji-pi-yu").

Only for tokens no list knows. speech_text.say_capitals asks READ_AS_WORD, EN_ACRONYMS and the
tenant lexicon first; this model comes next, and below its confidence threshold the token falls
back to Indonesian letter names (the measured rule).

A model fits here, unlike for titles: there is no lookup answer for a token nobody listed, and
its shape (vowels, clusters, length, letters like Q X V Z) carries most of the reading.

    python acronym_clf.py train                                  # acronym_data.csv -> acronym_clf.joblib
    python acronym_clf.py train --extra ../train/acronyms/out/accepted.csv --report ../train/acronyms/out
    python acronym_clf.py predict BAPENDA GRPC UNESA

The Dockerfile runs `train` at build time, so the model file is never committed (CLAUDE.md: no
weights in git). Training is deterministic and takes about a second.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "acronym_data.csv"
MODEL_PATH = Path(os.environ.get("ACRONYM_CLF_PATH", HERE / "acronym_clf.joblib"))
THRESHOLD = float(os.environ.get("ACRONYM_CLF_CONF", "0.5"))
LOG_PATH = os.environ.get("ACRONYM_LOG_PATH", "")
LABELS = ("WORD", "SPELL_ID", "SPELL_EN")

VOWELS = set("AEIOU")
EN_HEAVY = set("QXVZFWYC")
ID_CLUSTERS = ("NG", "NY", "KH", "SY", "PR", "TR", "KR", "BR", "DR", "GR", "ST", "SP", "SK")


def shape(token: str) -> list[str]:
    """Pronounceability tags. Module level, so the pickled pipeline finds it by name."""
    t = token.upper()
    n = max(len(t), 1)
    vowels = sum(c in VOWELS for c in t)
    longest_run = max((len(m) for m in re.findall(r"[^AEIOU]+", t)), default=0)
    switches = sum((t[i] in VOWELS) != (t[i + 1] in VOWELS) for i in range(n - 1)) / max(n - 1, 1)
    return [
        f"len{min(n, 8)}",
        "short" if n <= 3 else "mid" if n <= 5 else "long",
        f"vowels{min(vowels, 4)}",
        f"vratio{round(vowels / n * 4)}",
        f"run{min(longest_run, 5)}",
        f"switch{round(switches * 4)}",
        f"en{min(sum(c in EN_HEAVY for c in t), 3)}",
        f"idclust{int(any(c in t for c in ID_CLUSTERS))}",
        f"startv{int(t[0] in VOWELS)}",
        f"endv{int(t[-1] in VOWELS)}",
        f"sayable{int(n >= 4 and vowels >= 1 and longest_run <= 2)}",
    ]


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline([
        ("features", FeatureUnion([
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), lowercase=False, sublinear_tf=True)),
            ("shape", TfidfVectorizer(analyzer=shape, lowercase=False, use_idf=False, norm=None)),
        ])),
        ("clf", LogisticRegression(C=2.5, max_iter=2000, class_weight="balanced", random_state=42)),
    ])


# ---------------------------------------------------------------- runtime

_model = None
_known: dict[str, str] = {}   # every labelled token: looked up, never guessed
_loaded = False
_logged: set[str] = set()


def _load():
    global _model, _known, _loaded
    if not _loaded:
        _loaded = True
        if MODEL_PATH.exists():
            try:
                import joblib
                saved = joblib.load(MODEL_PATH)
                _model, _known = saved["model"], saved["known"]
            except Exception as exc:  # noqa: BLE001 - no model means the safe fallback, not a crash
                log.warning("Acronym model not loaded (%s): %s", MODEL_PATH, exc)
    return _model


@lru_cache(maxsize=4096)
def predict(token: str) -> tuple[str, float] | None:
    model = _load()
    if model is None:
        return None
    if token in _known:
        return _known[token], 1.0
    probs = model.predict_proba([token])[0]
    best = int(probs.argmax())
    label, conf = str(model.classes_[best]), float(probs[best])
    if LOG_PATH and token not in _logged:   # what the LLM really writes: input for the next data round
        _logged.add(token)
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{token}\t{label}\t{conf:.3f}\n")
        except OSError:
            pass
    return label, conf


def classify(token: str, threshold: float = THRESHOLD) -> str | None:
    """WORD / SPELL_ID / SPELL_EN, or None when the model is missing or unsure."""
    p = predict(token)
    return p[0] if p and p[1] >= threshold else None


# ---------------------------------------------------------------- training

def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("token")]


def merge(curated: list[dict], extra: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """Curated rows always win. A generated row only adds a token nobody labelled."""
    data, conflicts = {}, []
    for r in curated:
        data[r["token"].strip().upper()] = {**r, "label": r["label"].strip().upper()}
    for r in extra:
        tok, label = r["token"].strip().upper(), r["label"].strip().upper()
        if not re.fullmatch(r"[A-Z]{2,14}", tok) or label not in LABELS:
            continue
        if tok in data:
            if data[tok]["label"] != label:
                conflicts.append({"token": tok, "kept": data[tok]["label"], "dropped": label})
            continue
        data[tok] = {"token": tok, "label": label, "source": r.get("source", "generated")}
    return data, conflicts


def evaluate(tokens: list[str], labels: list[str], threshold: float) -> dict:
    """5-fold CV grouped by the first 3 letters, so KEMENKES and KEMENDAG never sit on both sides."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupKFold

    groups = [t[:3] for t in tokens]
    y = np.array(labels)
    pred = np.empty(len(tokens), dtype=object)
    conf = np.zeros(len(tokens))
    for train, test in GroupKFold(n_splits=5).split(tokens, y, groups):
        m = clone(build_pipeline()).fit([tokens[i] for i in train], y[train])
        p = m.predict_proba([tokens[i] for i in test])
        pred[test] = m.classes_[p.argmax(1)]
        conf[test] = p.max(1)
    sure = conf >= threshold
    report = classification_report(y, pred, digits=3, output_dict=True)
    print(classification_report(y, pred, digits=3))
    print(f"threshold {threshold}: answers {sure.mean():.1%} of unseen tokens, "
          f"{(pred[sure] == y[sure]).mean():.1%} of those right; the rest fall back to Indonesian letters")
    return {"report": report, "coverage": float(sure.mean()), "accuracy_when_sure": float((pred[sure] == y[sure]).mean()),
            "accuracy_all": float((pred == y).mean())}


def train(args) -> None:
    import joblib

    import acronym_clf  # the pickled pipeline must point at acronym_clf.shape, not __main__.shape

    curated = read_rows(args.data)
    extra = [r for path in args.extra for r in read_rows(path)]
    data, conflicts = merge(curated, extra)
    tokens = sorted(data)
    labels = [data[t]["label"] for t in tokens]
    print(f"{len(tokens)} tokens ({len(curated)} curated, {len(tokens) - len(curated)} generated): "
          + ", ".join(f"{lab} {labels.count(lab)}" for lab in LABELS))
    for c in conflicts[:20]:
        print(f"  conflict {c['token']}: kept {c['kept']}, generated said {c['dropped']}")

    result = {} if args.quiet else evaluate(tokens, labels, args.threshold)
    model = acronym_clf.build_pipeline().fit(tokens, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "known": dict(zip(tokens, labels))}, args.out, compress=3)
    print(f"saved {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

    if not args.quiet:
        probe = [t for t in "BAPENDA DISDUKCAPIL GRPC NVLINK SIPENMARU KOMINFO UNESA UINSA IKIP PNJ PENS TKDN".split()
                 if t not in data]
        for t in probe:
            p = model.predict_proba([t])[0]
            print(f"  {t:<12} {model.classes_[p.argmax()]:<9} {p.max():.2f}")
        t0 = time.perf_counter()
        for i in range(500):
            model.predict_proba([probe[i % len(probe)] if probe else "ABCD"])
        print(f"{(time.perf_counter() - t0) * 2:.3f} ms per token, uncached")
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        (args.report / "acronym_clf_report.json").write_text(
            json.dumps({**result, "tokens": len(tokens), "conflicts": conflicts}, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=DATA_PATH)
    t.add_argument("--extra", type=Path, action="append", default=[], help="Generated token,label CSV (verified).")
    t.add_argument("--out", type=Path, default=MODEL_PATH)
    t.add_argument("--threshold", type=float, default=THRESHOLD)
    t.add_argument("--report", type=Path, default=None)
    t.add_argument("--quiet", action="store_true", help="Fit and save only (Docker build).")
    p = sub.add_parser("predict")
    p.add_argument("tokens", nargs="+")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        sys.path.insert(0, str(HERE))
        for tok in args.tokens:
            print(tok, predict(tok.upper()))


if __name__ == "__main__":
    main()
