#!/usr/bin/env python3
"""Build orchestrator/title_lexicon.json and check the title reader against it.

This is the "training" step for titles. There are no weights: the seed plus the LLM-verified
entries become the lexicon, then the reader is scored on synthetic cases. The build fails when
the lexicon has bad entries, two explicit titles collide, or accuracy drops below --min-acc.

    python build_title_lexicon.py                                  # seed only
    python build_title_lexicon.py --accepted out/accepted.jsonl    # plus generate_title_dataset.mjs output
    python build_title_lexicon.py --report out/build               # failures and cases written there

accepted.jsonl rows (one per line):
    {"type": "field"|"spesialis"|"konsultan", "code": "Kom", "id": "Komputer", "en": "Computer Science"}
    {"type": "post"|"pre", "surface": "M.Sc.", "id": "Master of Science", "en": "...", "letters": "en"}
The seed wins every conflict. Conflicts are listed so you can fix the seed or the generated row.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")
TABLES = ("field", "spesialis", "konsultan")


def norm(text: str | None) -> str:
    return re.sub(r"[^a-z]+", " ", (text or "").lower()).strip()


def merge(seed: dict, accepted_files: list[Path]) -> tuple[dict, list[dict]]:
    data = json.loads(json.dumps(seed))
    data.pop("_comment", None)
    conflicts = []
    index = {t: {re.sub(r"[^A-Za-z]", "", e["code"]): e for e in data["tables"].get(t, [])} for t in TABLES}
    index["post"] = {re.sub(r"[^A-Za-z]", "", e["surface"]): e for e in data["post"]}
    index["pre"] = {re.sub(r"[^A-Za-z]", "", e["surface"]): e for e in data["pre"]}

    for path in accepted_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("type")
            if kind not in index:
                continue
            name = row.get("code") if kind in TABLES else row.get("surface")
            if not name:
                continue
            k = re.sub(r"[^A-Za-z]", "", name)
            old = index[kind].get(k)
            if old:
                if norm(old.get("id")) != norm(row.get("id")):
                    conflicts.append({"type": kind, "key": k, "kept": old.get("id"), "dropped": row.get("id"),
                                      "source": str(path)})
                continue
            entry = {f: row[f] for f in ("code", "surface", "id", "en", "letters", "short", "short_en", "initial",
                                         "dot_required", "chain_only_dotless") if row.get(f) not in (None, "", False)}
            entry.setdefault("id", row.get("id"))
            index[kind][k] = entry
            (data["tables"].setdefault(kind, []) if kind in TABLES else data[kind]).append(entry)
    return data, conflicts


def validate(data: dict) -> list[str]:
    errors = []

    def check(where: str, e: dict, name_field: str) -> None:
        name = e.get(name_field, "")
        if not re.search(r"[A-Za-z]", name):
            errors.append(f"{where}: no letters in {name!r}")
        for lang in ("id", "en"):
            text = e.get(lang)
            if text and ALLCAPS.search(text):
                errors.append(f"{where} {name}: {lang} {text!r} has capitals the acronym rule would spell out")
        if not e.get("id") and not e.get("short") and where != "post":
            errors.append(f"{where} {name}: no reading")

    for lv in data["levels"]:
        check("level", lv, "code")
        if lv.get("table", "field") not in data["tables"]:
            errors.append(f"level {lv['code']}: unknown table {lv.get('table')}")
    for t in TABLES:
        for e in data["tables"].get(t, []):
            check(t, e, "code")
    for e in data["post"]:
        check("post", e, "surface")
    for e in data["pre"]:
        check("pre", e, "surface")
    return errors


def evaluate(lexicon_path: Path, n: int, seed: int, report: Path | None) -> tuple[float, bool, dict]:
    os.environ["TITLE_LEXICON_PATH"] = str(lexicon_path)
    sys.path.insert(0, str(ROOT / "orchestrator"))
    sys.path.insert(0, str(HERE))
    titles = importlib.import_module("titles")
    cases_mod = importlib.import_module("gen_title_cases")

    data = json.loads(lexicon_path.read_text(encoding="utf-8"))
    cases = cases_mod.generate(data, n, seed)
    per_kind = collections.defaultdict(lambda: [0, 0])
    failures = []
    for c in cases:
        got = titles.say_titles(c["written"], c["lang"], c["mode"])
        ok = got == c["spoken"]
        for k in c["kinds"]:
            per_kind[k][0] += ok
            per_kind[k][1] += 1
        if not ok:
            failures.append({**c, "got": got})

    total_ok = sum(1 for c in cases) - len(failures)
    acc = total_ok / len(cases)
    negatives_ok = per_kind["negative"][0] == per_kind["negative"][1]
    print(f"\n{'kind':<28} {'ok':>6} {'n':>6}  acc")
    for k, (ok, cnt) in sorted(per_kind.items()):
        print(f"{k:<28} {ok:>6} {cnt:>6}  {ok / cnt:.4f}")
    print(f"{'ALL':<28} {total_ok:>6} {len(cases):>6}  {acc:.4f}")
    for f in failures[:15]:
        print(f"\n  FAIL [{f['lang']}/{f['mode']}] {f['written']}\n   want {f['spoken']}\n   got  {f['got']}")

    if titles.READER.collisions:
        print("\nTitles that share a spelling (first one kept):")
        for k, a, b in titles.READER.collisions:
            print(f"  {k}: {a} vs {b}")

    if report:
        report.mkdir(parents=True, exist_ok=True)
        with (report / "cases.jsonl").open("w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with (report / "failures.jsonl").open("w", encoding="utf-8") as f:
            for c in failures:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return acc, negatives_ok, {k: v for k, v in per_kind.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-file", type=Path, default=HERE / "title_seed.json")
    ap.add_argument("--accepted", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / "orchestrator" / "title_lexicon.json")
    ap.add_argument("--n", type=int, default=4000, help="Synthetic cases to score.")
    ap.add_argument("--random-seed", type=int, default=13)
    ap.add_argument("--min-acc", type=float, default=0.999)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    seed = json.loads(args.seed_file.read_text(encoding="utf-8"))
    data, conflicts = merge(seed, args.accepted)
    errors = validate(data)
    sizes = {t: len(data["tables"].get(t, [])) for t in TABLES} | {"post": len(data["post"]), "pre": len(data["pre"])}
    print(f"lexicon: {sizes}, levels: {len(data['levels'])}")
    if conflicts:
        print(f"{len(conflicts)} generated rows disagree with the seed (seed kept):")
        for c in conflicts[:20]:
            print(f"  {c['type']} {c['key']}: kept {c['kept']!r}, dropped {c['dropped']!r}")
        if args.report:
            args.report.mkdir(parents=True, exist_ok=True)
            (args.report / "conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=1), encoding="utf-8")
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} bad entries, lexicon not written")

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.no_eval:
        return
    acc, negatives_ok, _ = evaluate(args.out, args.n, args.random_seed, args.report)
    if acc < args.min_acc or not negatives_ok:
        sys.exit(f"accuracy {acc:.4f} (min {args.min_acc}), negatives {'ok' if negatives_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
