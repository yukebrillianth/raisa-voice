"""LoRA v2 data (one-off): excluded rows, the shared holdout, norm/hybrid manifests, eval sentences.

    docker run --rm -v ~/yuke/raisa-voice/orchestrator:/orchestrator:ro -v ~/yuke/raisa-voice-data600:/d \
      -v ~/yuke/raisa-voice-train/v2:/v2:ro raisa-voice-orchestrator python /v2/v2_prep.py

norm   = the reviewed normalizer-v2 transcripts (out/voxcpm2_lora_600_dataset.normalized.jsonl).
hybrid = the same normalizer with acronyms left raw (QRIS, PDF, USB, ITS, WIB...): numbers, Rp,
         time, units and Title Case names as normalize() does. Not a product mode on purpose.
Ran against orchestrator/speech_text.py of commit ad64faf; the product normalizer is now the hybrid
one, so norm() here no longer reproduces the norm set (it is archived in HF v600/).
"""
import json, random, re, sys

sys.path.insert(0, "/orchestrator")
import speech_text as st

D = "/d/"
# Units the normalizer leaves raw. Kept out of the holdout draw (so it stays the same as the first draw)
# but trained with the actor's readings, as the user gave them: Indonesian letters for Mbps/psi, English for dB.
TRAIN_ONLY = re.compile(r"Mbps|kHz|\bpsi\b|\bdB\b|USB-C")
SPOKEN = {"yu es bi-C": "yu es bi si", "Mbps": "em be pe es", "kHz": "kilohertz", "psi": "pe es i", "dB": "di bi"}
HYBRID_UNITS = {"kHz": "kilohertz"}  # hybrid leaves letter readings raw (USB-C, Mbps, psi, dB)


def say(text, table):
    for k, v in table.items():
        text = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", v, text)
    return text

UNSEEN = [  # numbers/acronyms/units not in the training text, plus the user's rule words in new sentences
    "Total belanja Anda Rp 1.275.000, bisa dibayar pakai QRIS di kasir.",
    "Rapat koordinasi dimulai pukul 14.30 WITA di ruang dua.",
    "Suhu badan tamu tercatat 37,8°C, mohon istirahat dulu.",
    "Laptop ini memakai SSD dan CPU generasi terbaru.",
    "Silakan siapkan NIK dan surat SKCK Anda di meja pendaftaran.",
    "Pendaftaran beasiswa LPDP dibuka sampai akhir bulan.",
    "Pak PRASETYO dari bagian keuangan sudah menunggu di lobi.",
    "Kirim berkas PDF lewat email sebelum pukul 16.45 WIB.",
    "Colokkan flashdisk ke port USB di belakang TV.",
    "Server ini memakai GPU dengan memori 48 GB.",
    "Sambungkan laptop ke proyektor pakai kabel HDMI di gedung ITS.",
    "Konsumsi listrik robot ini sekitar 1,5 kWh per hari.",
    "Please plug the HDMI cable into the screen on your left.",
    "Welcome to ITS, the campus opens at 07:15 on weekdays.",
    "Please scan the QR code to open the API guide.",
]
UNSEEN_WORDS = "1.275.000 14.30 37,8 SSD CPU NIK SKCK LPDP PRASETYO 16.45 48 1,5 07:15".split()

FRONT = st.FRONT[:2] + [(re.compile(r"\b(kWh|GHz)\b"), lambda m: st.FIXED[m.group(1)])]  # units only


def keep_acronym(m):
    w = m.group(0)
    return w.capitalize() if w in st.READ_AS_WORD or len(w) > 6 else w


def hybrid_one(text):
    rules = st.EN_RULES if st.language(text) == "en" else st.RULES
    for pattern, replace in FRONT + rules[:-1] + [(rules[-1][0], keep_acronym)]:
        text = pattern.sub(replace, text)
    return text


def per_sentence(f, text):  # language is decided per sentence, as the robot's TTS gets it
    return " ".join(f(s) for s in re.split(r"(?<=[.!?])\s+", text))


def norm(t): return per_sentence(st.normalize, t)
def hybrid(t): return per_sentence(hybrid_one, t)


assert hybrid("Bayar pakai QRIS Rp 45.000 pukul 09.00 WIB di ITS, kak ANDI.") == \
    "Bayar pakai QRIS empat puluh lima ribu rupiah pukul sembilan WIB di ITS, kak Andi.", hybrid("Bayar pakai QRIS Rp 45.000 pukul 09.00 WIB di ITS, kak ANDI.")
assert hybrid("Pakai 3 kWh, kabel USB dan TV.") == "Pakai tiga kilowatt hour, kabel USB dan TV."
assert hybrid("Welcome to ITS at 07:15.") == "Welcome to ITS at seven fifteen."

rows = [json.loads(l) for l in open(D + "voxcpm2_lora_600_dataset.jsonl")]
reviewed = [json.loads(l) for l in open(D + "voxcpm2_lora_600_dataset.normalized.jsonl")]
assert len(rows) == len(reviewed) == 600 and all(a["audio"] == b["audio"] for a, b in zip(rows, reviewed))
diff = [i for i in range(600) if norm(rows[i]["text"]) != reviewed[i]["text"]]
print(f"reviewed transcripts that differ from norm() now: {diff}")

all_text = " ".join(r["text"] for r in rows)
for w in UNSEEN_WORDS:
    assert not re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", all_text), w

excluded = [i for i, r in enumerate(rows) if TRAIN_ONLY.search(r["text"])]
ok = [i for i in range(600) if i not in excluded]
lang = {i: st.language(rows[i]["text"].split(". ")[0]) for i in ok}
rng = random.Random(600)


def pick(pool, n_en, n):
    en, ind = [i for i in pool if lang[i] == "en"], [i for i in pool if lang[i] == "id"]
    return rng.sample(en, n_en) + rng.sample(ind, n - n_en)


# Rows 100-499 only for the general part: rows 0-99 were v1's training set, which would favour v1.
abbr = [i for i in ok if i >= 500]
n_en = min(2, sum(lang[i] == "en" for i in abbr))  # few English rows among 500-599
hold = sorted(pick(abbr, n_en, 10) + pick([i for i in ok if 100 <= i < 500], 5 - n_en, 20))
train = [i for i in range(600) if i not in hold]
assert hold == [101, 122, 160, 162, 165, 171, 182, 185, 275, 291, 300, 339, 361, 385, 421, 426, 432, 434, 473, 496, 513, 519, 533, 544, 551, 553, 556, 559, 581, 584]
assert len(hold) == 30 and sum(lang[i] == "en" for i in hold) >= 4


def write(name, idx, text):
    with open(D + name, "w") as f:
        for i in idx:
            f.write(json.dumps({**rows[i], "text": text(i)}, ensure_ascii=False) + "\n")


write("train_norm.jsonl", train, lambda i: say(reviewed[i]["text"], SPOKEN))
write("holdout_norm.jsonl", hold, lambda i: reviewed[i]["text"])
write("train_hybrid.jsonl", train, lambda i: say(hybrid(rows[i]["text"]), HYBRID_UNITS))
write("holdout_hybrid.jsonl", hold, lambda i: hybrid(rows[i]["text"]))
write("holdout.jsonl", hold, lambda i: rows[i]["text"])
sents = [{"sid": f"h{k}", "row": i, "audio": rows[i]["audio"], "orig": rows[i]["text"], "lang": lang[i],
          "norm": norm(rows[i]["text"]), "hybrid": hybrid(rows[i]["text"])} for k, i in enumerate(hold)]
sents += [{"sid": f"u{k}", "orig": t, "lang": st.language(t), "norm": norm(t), "hybrid": hybrid(t)} for k, t in enumerate(UNSEEN)]
json.dump({"train_only": {i: rows[i]["text"] for i in excluded}, "holdout": hold, "n_train": len(train),
           "sentences": sents}, open(D + "split_v2.json", "w"), ensure_ascii=False, indent=1)
for i in excluded:
    print(i, "norm:", say(reviewed[i]["text"], SPOKEN), "| hybrid:", say(hybrid(rows[i]["text"]), HYBRID_UNITS))
print(f"train-only {excluded}; holdout {hold} ({sum(lang[i] == 'en' for i in hold)} en); train {len(train)}")
for s in sents:
    if s["norm"] != s["hybrid"] or s["sid"][0] == "u":
        print(f"{s['sid']} [{s['lang']}] {s['orig']}\n   norm:   {s['norm']}\n   hybrid: {s['hybrid']}")
