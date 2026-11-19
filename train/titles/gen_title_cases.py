#!/usr/bin/env python3
"""Synthetic evaluation cases for orchestrator/titles.py, built from the lexicon itself.

Each case is a written sentence with titles around a name and the expected spoken text. The gold
text is assembled from the lexicon entries the generator picked, NOT by running the reader, so the
cases test what can actually go wrong at runtime: finding the titles, splitting a chain, sloppy
writing (dotless, spaced, missing final dot), sentence-final dots, Indonesian / English / code-switched
carriers, the three post modes, consultant suffixes, unknown degrees, and look-alikes that must stay
untouched (initials, acronym lists, US states).

    python gen_title_cases.py --lexicon ../../orchestrator/title_lexicon.json --n 4000 --out cases.jsonl
    python gen_title_cases.py ... --tts-out tts_title_sentences.txt   # spoken sentences for voice data
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "orchestrator"))
from titles import short_form  # noqa: E402  (pure formatting helper, no lexicon state)

FIRST = ("Bambang Budi Siti Sri Agus Dewi Rina Andi Joko Hadi Wahyu Ratna Nur Eko Yuni Fajar Putri Rizky "
         "Dian Ayu Hendra Lestari Arif Ahmad Muhammad Made Ketut Yohanes Maria Christian Bagus Intan "
         "Teguh Wulan Yoga Citra Gilang Fitri Rudy Soekarno").split()
LAST = ("Pramujati Santoso Wibowo Hartono Susanto Wijaya Kusuma Rahmawati Nugroho Setiawan Hidayat "
        "Saputra Pratama Sulistyo Purnomo Gunawan Halim Siregar Nasution Simanjuntak Pangaribuan "
        "Tanjung Wulandari Anggraini Hestiavin Firdausi Rahardjo Sutanto Kurniawan").split()

CARRIERS = {
    "id": ["Selamat datang, {X}.", "Pembicara hari ini adalah {X} dari ITS.", "{X} akan membuka acara.",
           "Silakan temui {X} di lantai dua.", "Mari kita sambut {X}.", "Sambutan disampaikan oleh {X}, lalu sesi tanya jawab.",
           "Dosen pembimbingnya {X}. Beliau sangat ramah.", "Ruangan ini milik {X}"],
    "en": ["Please welcome {X}.", "Our speaker today is {X} from ITS.", "{X} will open the event.",
           "You can meet {X} on the second floor.", "This lab is led by {X}. The team has ten members.",
           "The keynote is by {X}, followed by a panel."],
    "mix": ["Next speaker kita adalah {X}, beliau expert di bidang AI.", "Let's welcome {X} ya.",
            "Meeting nanti sama {X} di ruang rapat.", "Project ini di-supervise oleh {X}. Keren banget."],
}

NEGATIVES = [
    ("Kampus ITS, UGM, dan UI sudah buka.", "id"), ("Motor seri ST, MT, dan SE tersedia.", "id"),
    ("Bambang H. Santoso hadir hari ini.", "id"), ("Bapak S.H. Wibowo sudah datang.", "id"),
    ("Airnya habis. Irigasi di Dr.X.", "id"), ("Vitamin C, D, dan E penting.", "id"),
    ("Ukuran S, M, dan L tersedia.", "id"), ("Dia tinggal di Jakarta, Indonesia.", "id"),
    ("Tim Budi, Ani, dan Sinta menang.", "id"), ("Saya ketemu S. H. Wibowo kemarin.", "id"),
    ("The office is in San Jose, CA.", "en"), ("Baltimore, MD is in the US.", "en"),
    ("Pilih paket A, B, atau C.", "id"), ("Dosen kami ramah, sabar, dan pintar.", "id"),
    ("Robot Raisa, Nabbit, dan Iris dipamerkan.", "id"), ("Silakan isi KTP, KK, dan NPWP.", "id"),
    ("Bu Siti, Pak Joko, dan Mas Andi sudah hadir.", "id"), ("Coba restart PC, TV, dan HP kamu.", "id"),
    ("Welcome to Surabaya, East Java.", "en"), ("It costs more in Boston, MA than here.", "en"),
]


def composed(data: dict) -> list[dict]:
    """Level x field titles, minus those an explicit entry overrides (M.S. is Magister Sains, not Sastra)."""
    explicit = {re.sub(r"[^A-Za-z]", "", e["surface"]) for e in data["post"]}
    out = []
    for lv in data["levels"]:
        for f in data["tables"][lv.get("table", "field")]:
            surface = f"{lv['code']}.{f['code']}."
            if re.sub(r"[^A-Za-z]", "", surface) in explicit:
                continue
            out.append({"surface": surface, "id": f"{lv['id']} {f['id']}",
                        "en": f"{lv['en']} {f['en']}" if lv.get("en") and f.get("en") else f"{lv['id']} {f['id']}",
                        "letters": "id", "table": lv.get("table", "field"), "level": lv["code"]})
    return out


def spoken(e: dict, lang: str, mode: str, pre: bool = False) -> str:
    if mode == "short" and not pre:
        if lang == "en":
            return e.get("short_en") or short_form(e["surface"], e.get("letters", "id"), "en")
        return e.get("short") or short_form(e["surface"], e.get("letters", "id"), "id")
    full = (e.get("en") or e.get("id")) if lang == "en" else e.get("id")
    return full or (e.get("short_en") if lang == "en" else e.get("short")) or short_form(
        e["surface"], e.get("letters", "id"), "en" if lang == "en" else "id")


def variant(e: dict, rng: random.Random, first: bool, pre: bool = False) -> tuple[str, str]:
    """A way of writing the title, and its kind."""
    s = e["surface"]
    if pre:
        if not e.get("dot_required") and s.count(".") == 1 and s.endswith(".") and rng.random() < 0.2:
            return s[:-1], "pre_nodot"
        return s, "canonical"
    if "." not in s or " " in s.strip("."):
        return s, "canonical"
    r = rng.random()
    if r < 0.15 and not (e.get("chain_only_dotless") and first):
        return s.replace(".", "").replace(" ", ""), "dotless"
    if r < 0.25:
        return re.sub(r"\.(?=\S)", ". ", s), "spaced"
    if r < 0.35:
        return s[:-1] if s.endswith(".") else s, "no_final_dot"
    return s, "canonical"


def fill(template: str, written: str, said: str) -> tuple[str, str]:
    """Put a chain in a carrier. A chain ending in '.' shares that dot with the sentence."""
    if not written.endswith("."):
        return template.replace("{X}", written), template.replace("{X}", said)
    if "{X}." in template:
        return template.replace("{X}.", written), template.replace("{X}.", said + ".")
    if template.endswith("{X}"):
        return template.replace("{X}", written), template.replace("{X}", said) + "."
    # mid-sentence: the title's own dot is not spoken
    return template.replace("{X}", written), template.replace("{X}", said)


def generate(data: dict, n: int = 4000, seed: int = 13, modes=("expand", "short", "drop")) -> list[dict]:
    rng = random.Random(seed)
    pre = data["pre"]
    pre_academic = [e for e in pre if e["surface"] in {"Prof.", "Dr.", "Dr.Eng.", "Dr.-Ing.", "Dr.rer.nat."}]
    pre_prof = [e for e in pre if e["surface"] in {"Ir.", "dr.", "drg.", "drh.", "Drs.", "Dra.", "apt.", "Ns."}]
    pre_relig = [e for e in pre if e["surface"] in {"H.", "Hj.", "K.H.", "Ust.", "Pdt."}]
    comp = composed(data)
    degrees = [e for e in comp if e["table"] == "field"]
    bach = [e for e in degrees if e["level"] in {"S", "S.Tr", "A.Md"}] + [e for e in data["post"] if e["surface"].startswith("B")]
    mast = [e for e in degrees if e["level"] in {"M", "M.Tr"}] + [e for e in data["post"] if e["surface"].startswith("M") and e.get("letters") == "en"]
    doct = [e for e in data["post"] if e["surface"] in {"Ph.D.", "D.Eng.", "D.Sc.", "Ed.D.", "DBA", "D.Phil."}]
    prof = [e for e in data["post"] if e["surface"] in {"IPU", "IPM", "IPP", "ASEAN Eng.", "APEC Eng.", "Apt.", "Psikolog",
                                                       "Ak.", "CPA", "CFA", "PMP", "MARS", "Ns."}]
    spes = [e for e in comp if e["table"] == "spesialis"]
    kons = data["tables"].get("konsultan", [])
    cases: list[dict] = []

    for i in range(n):
        lang_key = rng.choice(["id", "id", "en", "mix"])
        lang = "en" if lang_key == "en" else "id"
        mode = "expand" if rng.random() < 0.7 else rng.choice([m for m in modes if m != "expand"] or ["expand"])
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        kinds: set[str] = set()

        medical = rng.random() < 0.15
        chain_pre: list[dict] = []
        if medical:
            chain_pre = ([rng.choice(pre_academic)] if rng.random() < 0.3 else []) + [next(e for e in pre if e["surface"] == "dr.")]
        else:
            if rng.random() < 0.5:
                chain_pre.append(next(e for e in pre if e["surface"] == "Prof."))
            if rng.random() < 0.6:
                chain_pre.append(rng.choice(pre_academic[1:] or pre_academic))
            if rng.random() < 0.4:
                chain_pre.append(rng.choice(pre_prof))
            if rng.random() < 0.15:
                chain_pre.append(rng.choice(pre_relig))
        hc = bool(chain_pre) and chain_pre[0]["surface"] in {"Prof.", "Dr."} and rng.random() < 0.15
        if hc and not any(e["surface"] == "Dr." for e in chain_pre):
            hc = False

        written_pre, said_pre = [], []
        for e in chain_pre:
            w, k = variant(e, rng, True, pre=True)
            kinds.add(k)
            written_pre.append(w)
            said_pre.append(spoken(e, lang, mode, pre=True))
            if hc and e["surface"] == "Dr.":
                written_pre.append(rng.choice(["(H.C.)", "(HC)", "(h.c.)"]))
                said_pre.append("Honoris Causa")
                kinds.add("honoris_causa")

        posts: list[tuple[str, str]] = []
        if medical:
            sp = rng.choice(spes)
            w, said = sp["surface"], spoken(sp, lang, mode)
            r = rng.random()
            if sp["surface"] == "Sp.PD." and r < 0.6 and kons:
                k = rng.choice(kons)
                w = f"Sp.PD-K{k['code']}"
                said = short_form(w, "id", lang) if mode == "short" else (
                    f"{sp['en']}, Consultant in {k['en']}" if lang == "en" else f"{sp['id']} Konsultan {k['id']}")
                kinds.add("konsultan")
            elif r < 0.2:
                w = sp["surface"].rstrip(".") + "(K)"
                said = short_form(w, "id", lang) if mode == "short" else (
                    f"{sp['en']}, Consultant" if lang == "en" else f"{sp['id']} Konsultan")
                kinds.add("konsultan")
            posts.append((w, said))
            if rng.random() < 0.3:
                e = next((x for x in data["post"] if x["surface"] == "FINASIM"), None)
                if e:
                    posts.append((e["surface"], spoken(e, lang, mode)))
        else:
            pools = [(bach, 0.7), (mast, 0.6), (doct, 0.35), (prof, 0.3)]
            for pool, p in pools:
                if pool and rng.random() < p:
                    e = rng.choice(pool)
                    w, k = variant(e, rng, not posts)
                    kinds.add(k)
                    posts.append((w, spoken(e, lang, mode)))
            if posts and rng.random() < 0.05:   # unknown degree inside a chain: spelled out
                fake = f"M.{rng.choice(['Qwx', 'Zrb', 'Xkt'])}."
                posts.append((fake, short_form(fake, "id", lang)))
                kinds.add("unknown_degree")
        if not chain_pre and not posts:
            posts.append(("S.T.", spoken({"surface": "S.T.", "id": "Sarjana Teknik", "en": "Bachelor of Engineering"}, lang, mode)))

        written = " ".join(written_pre + [name])
        said = " ".join(said_pre + [name])
        if posts:
            written += "".join(f", {w}" for w, _ in posts)
            if mode != "drop":
                said += "".join(f", {s}" for _, s in posts)
            kinds.add(f"mode_{mode}")
        template = rng.choice(CARRIERS[lang_key])
        w, s = fill(template, written, said)
        kinds.add(f"lang_{lang_key}")
        if posts and posts[-1][0].endswith(".") and re.search(r"\{X\}\. [A-Z]", template):
            kinds.add("sentence_end_then_sentence")
        cases.append({"written": w, "lang": lang, "mode": mode, "spoken": s, "kinds": sorted(kinds)})

    for text, lang in NEGATIVES:
        for mode in ("expand", "drop"):
            cases.append({"written": text, "lang": lang, "mode": mode, "spoken": text, "kinds": ["negative"]})
    for e in data["post"][:40] + composed(data)[:40] + data["pre"]:
        for lang in ("id", "en"):
            pre_ = e in data["pre"]
            cases.append({"written": e["surface"] + ",", "lang": lang, "mode": "expand",
                          "spoken": spoken(e, lang, "expand", pre=pre_) + ",", "kinds": ["standalone"]})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=Path(__file__).resolve().parents[2] / "orchestrator" / "title_lexicon.json")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=Path("cases.jsonl"))
    ap.add_argument("--tts-out", type=Path, default=None, help="Spoken sentences (expand mode) for TTS training text.")
    args = ap.parse_args()
    data = json.loads(args.lexicon.read_text(encoding="utf-8"))
    cases = generate(data, args.n, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases -> {args.out}")
    if args.tts_out:
        lines = sorted({c["spoken"] for c in cases if c["mode"] == "expand" and "negative" not in c["kinds"]
                        and "standalone" not in c["kinds"]})
        args.tts_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(lines)} spoken sentences -> {args.tts_out}")


if __name__ == "__main__":
    main()
