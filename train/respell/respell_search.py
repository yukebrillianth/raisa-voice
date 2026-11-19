#!/usr/bin/env python3
"""Let the served voice choose the spelling: search written forms until it says the word right.

Every text rule so far was a spelling a person picked, then measured ("wé-i-bé" won 8/8, "W I B" lost
0/8). This turns it around. For each word the voice gets wrong (GPU "gepiu", API "apii", PDF "pedeef",
"seekor", "keenam") it writes many candidate forms, renders each through the running TTS service
(the same voice and path the robot uses), judges every clip with the phoneme recogniser, and keeps the
form the voice reads right most often.

Candidate families, per letter or per syllable:
- id:     Indonesian spelling of the reading       "ji pi yu", "ji-pi-yu", "jipiyu"
- en:     English look-alike spelling              "jee pee you", "jee-pee-you"
- cmu:    CMUdict phonemes in braces               "{JH IY1} {P IY1} {Y UW1}"  VoxCPM's base model read
          ARPAbet in braces during training (CLAUDE.md section 4), so it may follow it where it guesses
          at letters.
- raw / é-hyphen / spaced capitals, as before.
Round 2 mixes the two best families letter by letter: "{JH IY1}-pi-yu" and so on.

    # on the server, stack running; renders go through the tts service, the judge needs a GPU or patience
    docker compose run --rm --no-deps --entrypoint python -e PYTHONPATH=/train/v3:/train/respell \
        -v ./train:/train -v ./out:/out tts /train/respell/respell_search.py all /out/respell
    # step by step: render <dir> [--round 2] / judge <dir> / report <dir> / lexicon <dir>

Output <dir>: wav/ (every clip, kept for the thesis), manifest.jsonl, heard.jsonl, report.md,
respell.json (winners: >= --win of the samples right and clearly better than the served form).
Copy respell.json to orchestrator/respell.json; speech_text uses it in Indonesian sentences only.

Cost: 2 carriers x 8 samples per candidate; about 30 candidates per word in round 1, so ~500 short
renders (2-4 minutes on the 5090 at RTF 0.085) plus the judge. Runs beside live robots: nano-vllm
batches them with real requests, so do it off-hours.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "v3"))
from acro import CARRIERS, g2p, grade, heard, region  # noqa: E402

TTS_URL = os.environ.get("TTS_URL", "http://tts:8000")

# ---------------------------------------------------------------- what to say

# Letter names: Indonesian spelling of the English name, English look-alike, CMUdict phones.
LETTER = {
    "A": ("ei", "ay", "EY1"), "B": ("bi", "bee", "B IY1"), "C": ("si", "see", "S IY1"), "D": ("di", "dee", "D IY1"),
    "E": ("i", "ee", "IY1"), "F": ("ef", "eff", "EH1 F"), "G": ("ji", "jee", "JH IY1"), "H": ("eich", "aitch", "EY1 CH"),
    "I": ("ai", "eye", "AY1"), "J": ("jei", "jay", "JH EY1"), "K": ("kei", "kay", "K EY1"), "L": ("el", "ell", "EH1 L"),
    "M": ("em", "em", "EH1 M"), "N": ("en", "en", "EH1 N"), "O": ("ou", "oh", "OW1"), "P": ("pi", "pee", "P IY1"),
    "Q": ("kyu", "cue", "K Y UW1"), "R": ("ar", "are", "AA1 R"), "S": ("es", "ess", "EH1 S"), "T": ("ti", "tee", "T IY1"),
    "U": ("yu", "you", "Y UW1"), "V": ("vi", "vee", "V IY1"), "W": ("dabelyu", "double-you", "D AH1 B AH0 L Y UW0"),
    "X": ("eks", "ex", "EH1 K S"), "Y": ("wai", "why", "W AY1"), "Z": ("zi", "zee", "Z IY1"),
}
ID_LETTER = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))

# token -> how it must sound (acro spelling: the judge's target), which family of letters, carriers.
TARGETS = {
    "GPU": {"reading": "ji pi yu", "letters": "en"},
    "GB": {"reading": "ji bi", "letters": "en"},
    "API": {"reading": "ei pi ai", "letters": "en"},
    "CPU": {"reading": "si pi yu", "letters": "en", "carriers": ["Server ini memakai {} generasi terbaru.", "Harga {} sekarang sedang naik."]},
    "HD": {"reading": "eich di", "letters": "en", "carriers": ["Videonya tersedia dalam kualitas {} di situs kami.", "Layar {} ada di ruang tunggu."]},
    "PDF": {"reading": "pi di ef", "letters": "en"},
    "WIT": {"reading": "we i te", "letters": "id"},
    "SKCK": {"reading": "es ka ce ka", "letters": "id", "carriers": ["Silakan urus {} di kantor polisi terdekat.", "Fotokopi {} dikumpulkan hari ini."]},
    "seekor": {"reading": "se ekor", "word": True, "cmu": ["S AH0 EH1 K AO0 R", "S EH0 EH1 K OW0 R"],
               "carriers": ["Di taman ada {} kucing yang lucu.", "Saya melihat {} burung di atas pohon."]},
    "keenam": {"reading": "ke enam", "word": True, "cmu": ["K AH0 EH1 N AA0 M", "K EH0 EH1 N AA0 M"],
               "carriers": ["Ruangan {} ada di ujung lorong.", "Ini adalah kunjungan {} kami ke sini."]},
}


def carriers(token: str) -> list[str]:
    return TARGETS[token].get("carriers") or CARRIERS[token]


def letter_forms(token: str, family: str) -> list[str]:
    """One written piece per letter."""
    out = []
    for c in token:
        if family == "id_name":
            out.append(ID_LETTER[c])
        else:
            i = {"id": 0, "en": 1, "cmu": 2}[family]
            out.append("{" + LETTER[c][2] + "}" if family == "cmu" else LETTER[c][i])
    return out


def round_one(token: str) -> dict[str, str]:
    """name -> written form."""
    t = TARGETS[token]
    if t.get("word"):
        base = token
        syl = t["reading"]                        # "se ekor"
        a, b = syl.split(" ", 1)
        forms = {"raw": base, "space": syl, "hyphen": f"{a}-{b}", "apostrophe": f"{a}'{b}",
                 "accent": f"{a}-é{b[1:]}" if b.startswith("e") else f"{a}-{b}",
                 "accent_apos": f"{a}'é{b[1:]}" if b.startswith("e") else f"{a}'{b}",
                 "comma": f"{a}, {b}", "double": f"{a}-e{b}"}
        for k, cmu in enumerate(t.get("cmu", [])):
            forms[f"cmu{k}"] = "{" + cmu + "}"
        return forms
    forms = {"raw": token, "spaced": " ".join(token), "ehyph": "-".join(letter_forms(token, "id_name"))}
    for fam in ("id", "en", "cmu"):
        parts = letter_forms(token, fam)
        forms[f"{fam}_space"] = " ".join(parts)
        if fam != "cmu":
            forms[f"{fam}_hyph"] = "-".join(parts)
            forms[f"{fam}_join"] = "".join(parts)
            forms[f"{fam}_title"] = " ".join(p.capitalize() for p in parts)
        else:
            forms["cmu_one"] = "{" + " ".join(LETTER[c][2] for c in token) + "}"
            forms["cmu_flat"] = "{" + " ".join(re.sub(r"\d", "0", LETTER[c][2]) for c in token) + "}"
    return forms


def round_two(token: str, scores: dict[str, float]) -> dict[str, str]:
    """Mix the two best letter families, letter by letter, hyphen and space joins."""
    if TARGETS[token].get("word"):
        return {}
    fams = []
    for name, _ in sorted(scores.items(), key=lambda kv: -kv[1]):
        fam = name.split("_")[0]
        if fam in ("id", "en", "cmu") and fam not in fams:
            fams.append(fam)
    if len(fams) < 2:
        return {}
    per = {f: letter_forms(token, f) for f in fams[:2]}
    out = {}
    for choice in itertools.product(fams[:2], repeat=len(token)):
        if len(set(choice)) == 1:
            continue
        parts = [per[f][i] for i, f in enumerate(choice)]
        out["mix_" + "".join(f[0] for f in choice) + "_h"] = "-".join(parts)
        out["mix_" + "".join(f[0] for f in choice) + "_s"] = " ".join(parts)
    return out


# ---------------------------------------------------------------- render

async def render(out: Path, tokens: list[str], samples: int, rnd: int, parallel: int) -> None:
    import aiohttp

    (out / "wav").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = {json.loads(l)["file"] for l in manifest.read_text().splitlines()} if manifest.exists() else set()
    jobs = []
    for token in tokens:
        if rnd == 1:
            forms = round_one(token)
        else:
            forms = round_two(token, scores_for(out, token))
            if not forms:
                continue
        for name, written in forms.items():
            for n, template in enumerate(carriers(token)):
                for s in range(samples):
                    f = f"wav/{token}__r{rnd}__{name}__c{n}__s{s}.wav"
                    if f not in done:
                        jobs.append({"file": f, "token": token, "round": rnd, "form": name, "written": written,
                                     "carrier": n, "sample": s, "text": template.format(written)})
    print(f"{len(jobs)} clips to render")
    sem = asyncio.Semaphore(parallel)
    async with aiohttp.ClientSession() as session:
        async def one(job):
            async with sem:
                async with session.post(f"{TTS_URL}/v1/tts", json={"text": job["text"]}) as r:
                    r.raise_for_status()
                    rate, pcm = int(r.headers["X-Sample-Rate"]), await r.read()
            with wave.open(str(out / job["file"]), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            job["secs"] = round(len(pcm) / 2 / rate, 2)
            with manifest.open("a") as m:
                m.write(json.dumps(job, ensure_ascii=False) + "\n")
        for i in range(0, len(jobs), 200):
            await asyncio.gather(*(one(j) for j in jobs[i:i + 200]))
            print(f"  {min(i + 200, len(jobs))}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- judge

def judge(out: Path) -> None:
    from phones import Phones

    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    heard_path = out / "heard.jsonl"
    done = {json.loads(l)["file"] for l in heard_path.read_text().splitlines()} if heard_path.exists() else set()
    p = Phones()
    with heard_path.open("a") as h:
        for i, r in enumerate(rows):
            if r["file"] in done:
                continue
            h.write(json.dumps({"file": r["file"], "tokens": p(str(out / r["file"]))}, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"  judged {i}/{len(rows)}", flush=True)


def graded(out: Path) -> list[dict]:
    rows = {json.loads(l)["file"]: json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()}
    out_rows = []
    for line in (out / "heard.jsonl").read_text().splitlines():
        h = json.loads(line)
        r = rows.get(h["file"])
        if not r:
            continue
        template = carriers(r["token"])[r["carrier"]]
        left, right = (g2p(x) for x in template.split("{}"))
        target = g2p(TARGETS[r["token"]]["reading"])
        got = region(heard(h["tokens"]), left, target, right)
        r["heard"] = "".join(got)
        r["ok"] = r["secs"] <= 15 and grade(got, target, left, right, "id") != "wrong"
        out_rows.append(r)
    return out_rows


def scores_for(out: Path, token: str) -> dict[str, float]:
    by = defaultdict(list)
    for r in graded(out):
        if r["token"] == token:
            by[r["form"]].append(r["ok"])
    return {k: sum(v) / len(v) for k, v in by.items()}


# ---------------------------------------------------------------- report and lexicon

def report(out: Path, win: float) -> dict:
    rows = graded(out)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["token"], r["form"], r["written"])].append(r)
    winners, lines = {}, ["# Spelling search: the served voice picks the spelling", "",
                          "Each cell: clips read right by the phoneme judge (`phones.py` + `acro.grade`) / clips "
                          "(carriers x samples), then the most common wrong reading. Ordered best first per word.", ""]
    for token in dict.fromkeys(k[0] for k in cells):
        forms = sorted(((k[1], k[2], v) for k, v in cells.items() if k[0] == token),
                       key=lambda x: (-sum(r["ok"] for r in x[2]) / len(x[2]), x[0]))
        served = next((v for name, _, v in forms if name in ("raw", "ehyph")), None)
        served_rate = sum(r["ok"] for r in served) / len(served) if served else 0.0
        lines += [f"## {token} (target \"{TARGETS[token]['reading']}\", served form {served_rate:.0%})", "",
                  "| form | written | right | most common wrong |", "|---|---|---|---|"]
        for name, written, rs in forms:
            ok = sum(r["ok"] for r in rs)
            bad = Counter(r["heard"] for r in rs if not r["ok"]).most_common(1)
            lines.append(f"| {name} | `{written}` | {ok}/{len(rs)} | {f'`{bad[0][0]}` x{bad[0][1]}' if bad else ''} |")
        best_name, best_written, best = forms[0]
        rate = sum(r["ok"] for r in best) / len(best)
        # clearly better: above the bar and at least 3 clips of 16 over the served form (the judge moves ~5 points)
        if best_name not in ("raw", "ehyph") and rate >= win and (rate - served_rate) * len(best) >= 3:
            winners[token] = {"written": best_written, "form": best_name, "right": rate, "served": served_rate,
                              "clips": len(best)}
        lines.append("")
    lines += ["## Winners written to respell.json", ""]
    lines += [f"- {t}: `{w['written']}` {w['right']:.0%} (served {w['served']:.0%}, {w['clips']} clips)" for t, w in winners.items()] or ["- none"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(winners) - 2:]))
    return winners


def lexicon(out: Path, win: float) -> None:
    winners = report(out, win)
    words = {t: w["written"] for t, w in winners.items() if TARGETS[t].get("word")}
    caps = {t: w["written"] for t, w in winners.items() if not TARGETS[t].get("word")}
    data = {"_source": f"train/respell/respell_search.py, {out}", "capitals_id": caps, "words_id": words,
            "evidence": winners}
    (out / "respell.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'respell.json'}: {len(caps)} capitals, {len(words)} words")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "judge", "report", "lexicon", "all", "forms"])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tokens", default=",".join(TARGETS))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--win", type=float, default=0.875, help="Share of clips right a winner needs (7/8).")
    a = ap.parse_args()
    tokens = [t for t in a.tokens.split(",") if t]
    if a.cmd == "forms":
        for t in tokens:
            print(t, json.dumps(round_one(t), ensure_ascii=False, indent=1))
    elif a.cmd == "render":
        asyncio.run(render(a.dir, tokens, a.samples, a.round, a.parallel))
    elif a.cmd == "judge":
        judge(a.dir)
    elif a.cmd == "report":
        report(a.dir, a.win)
    elif a.cmd == "lexicon":
        lexicon(a.dir, a.win)
    else:
        asyncio.run(render(a.dir, tokens, a.samples, 1, a.parallel))
        judge(a.dir)
        asyncio.run(render(a.dir, tokens, a.samples, 2, a.parallel))
        judge(a.dir)
        lexicon(a.dir, a.win)


if __name__ == "__main__":
    main()
