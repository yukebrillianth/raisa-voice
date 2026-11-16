"""Render test sets through the served path (speech_text.normalize -> live tts service), as the robot hears them.

    docker compose run --rm --no-deps -T -e PYTHONPATH=/app:/v3 -v ./train/v3:/v3 -v ./out:/out orchestrator \
      python /v3/served_ab.py /out/served_<label> [sets]      # sets: length,short,seword,replies,ttfb (default all)
length   one sentence of about 3, 6, 12, 20, 30 s, Indonesian + English, 3 samples (muffle.py)
tail     the same closing clause after 0, 1, 3 or 5 clauses, 6 samples: is the same text duller late in a long sentence?
         variant "split": the same sentences as speech_text.tts_pieces sends them (one request per piece)
short    "Baik." + next sentence: two requests played back to back vs one merged request, 8 samples
seword   se-/ke- + e words written plain / "se-ekor" / "se ekor", 8 samples (gemini_judge.py --words)
replies  smooth_gen.REPLIES, one request per sentence as served, 3 samples (smooth_ana.py)
titles   academic-title sentences, 2 samples (listening only)
ttfb     the first sentence of every reply, one at a time, twice: first byte (ms) with nothing else running
Writes wavs and manifest.json in smooth_gen.py's format.
"""
import asyncio, json, os, re, sys, time, wave

import aiohttp
import numpy as np

from speech_text import normalize
from smooth_gen import REPLIES, SHORT

OUT = sys.argv[1]
SETS = (sys.argv[2] if len(sys.argv) > 2 else "length,tail,short,seword,titles,replies,ttfb").split(",")
TTS_URL = os.environ.get("TTS_URL", "http://tts:8000")
LENGTH = {
    3: ["Silakan ikuti saya ke lobi utama.", "Please follow me to the main lobby."],
    6: ["Ruang rapat ada di lantai dua, tepat di sebelah kanan lift setelah Anda keluar.",
        "The meeting room is on the second floor, right next to the lift when you step out."],
    12: ["Teaching Factory adalah tempat mahasiswa belajar membuat robot, mulai dari merancang mesin, memasang sensor, "
         "sampai menulis program yang membuat robot bisa melayani tamu seperti saya.",
         "The Teaching Factory is where students learn to build robots, from designing the machines and fitting the "
         "sensors to writing the programs that let a robot like me look after visitors."],
    20: ["Selama pameran berlangsung, Anda bisa mengunjungi lab robotika di lantai satu, melihat robot sepak bola yang "
         "sedang berlatih di lapangan kecil, mencoba berbicara dengan robot pelayan di ruang tamu, lalu mampir ke meja "
         "informasi untuk mengambil brosur dan jadwal kegiatan hari ini.",
         "While the exhibition is open, you can visit the robotics lab on the first floor, watch the football robots "
         "practising on the small pitch, try talking to the service robots in the guest room, and then stop by the "
         "information desk to pick up a brochure and today's schedule."],
    30: ["Kalau Anda datang bersama rombongan, sebaiknya daftar dulu di meja depan supaya petugas bisa menyiapkan "
         "pemandu, setelah itu rombongan akan diajak berkeliling melihat lab robotika, ruang produksi, dan area uji coba, "
         "lalu di akhir kunjungan ada sesi tanya jawab bersama mahasiswa yang membuat robot ini, jadi siapkan pertanyaan "
         "yang ingin Anda ajukan ya.",
         "If you are visiting with a group, it is best to register at the front desk first so that the staff can arrange "
         "a guide for you, after that the group will be taken around the robotics lab, the production room and the test "
         "area, and at the end of the visit there is a question and answer session with the students who built these "
         "robots, so please prepare the questions you would like to ask."],
}
SEWORDS = {"seekor": "Ada {} kucing di depan lobi.", "seenaknya": "Jangan parkir {} di depan pintu.",
           "keenam": "Ruangannya ada di lantai {}.", "keempat": "Ini kunjungan {} saya ke sini.",
           "keesokan": "Acara dilanjutkan pada {} harinya di aula."}
TAIL = {"id": ("silakan tanyakan kepada petugas di meja depan.",
              ["Selama pameran berlangsung, Anda bisa mengunjungi lab robotika di lantai satu",
               "melihat robot sepak bola yang sedang berlatih di lapangan kecil",
               "mencoba berbicara dengan robot pelayan di ruang tamu",
               "lalu mampir ke meja informasi untuk mengambil brosur dan jadwal kegiatan hari ini",
               "dan kalau masih ada pertanyaan tentang jadwal atau lokasi"]),
        "en": ("please ask the staff at the front desk.",
              ["While the exhibition is open, you can visit the robotics lab on the first floor",
               "watch the football robots practising on the small pitch",
               "try talking to the service robots in the guest room",
               "then stop by the information desk to pick up a brochure and today's schedule",
               "and if you still have questions about the schedule or the rooms"])}
TITLES = ["Pembicara hari ini adalah Prof. Dr. Ir. Budi Santoso, S.T., M.Sc. dari ITS.",
          "Silakan temui dr. Sinta, S.K.M. di ruang dua, lalu Dr. (HC) Ir. Joko, M.Eng., Ph.D. di aula.",
          "Rektor kami, Prof. Ir. Bambang, M.Sc.Eng., ASEAN Eng. membuka acara."]
FORMS = {"plain": lambda w: w, "hyphen": lambda w: w[:2] + "-" + w[2:], "space": lambda w: w[:2] + " " + w[2:]}


# Tried against the fade of long sentences and rejected: about 1 dB less fade, 0.4 s more pause per cut (smooth_report.md).
PIECE_CHARS = 150
# A piece this short comes out unstable (under about 1 s of speech), so it joins the one before it.
MIN_PIECE_CHARS = 40


def tts_pieces(spoken: str, limit: int = PIECE_CHARS) -> list[str]:
    """A long spoken sentence cut after commas into pieces of at most `limit` characters (a clause longer
    than that stays whole)."""
    if len(spoken) <= limit:
        return [spoken]
    pieces = [""]
    for clause in re.split(r"(?<=,)\s+", spoken):
        if pieces[-1] and len(pieces[-1]) + 1 + len(clause) > limit:
            pieces.append(clause)
        else:
            pieces[-1] = f"{pieces[-1]} {clause}".strip()
    if len(pieces) > 1 and len(pieces[-1]) < MIN_PIECE_CHARS:
        pieces[-2] += " " + pieces.pop()
    return pieces


async def main():
    os.makedirs(OUT, exist_ok=True)
    manifest, solo, gate = [], [], asyncio.Semaphore(4)
    async with aiohttp.ClientSession() as session:
        async def say(text, spoken=None):
            t0, first, pcm = time.monotonic(), None, bytearray()
            async with session.post(f"{TTS_URL}/v1/tts", json={"text": spoken or normalize(text)}) as r:
                r.raise_for_status()
                rate = int(r.headers["X-Sample-Rate"])
                async for chunk in r.content.iter_any():
                    first = first or (time.monotonic() - t0) * 1000
                    pcm += chunk
            return np.frombuffer(bytes(pcm), "<i2"), rate, first

        def save(name, parts, rate, **info):
            os.makedirs(os.path.dirname(f"{OUT}/{name}"), exist_ok=True)
            with wave.open(f"{OUT}/{name}", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(np.concatenate(parts).tobytes())
            manifest.append({"file": name, "bounds": np.cumsum([len(p) for p in parts]).tolist(), "rate": rate, **info})

        async def sentences(name, sents, split=False, **info):
            async with gate:  # one robot's answer: its sentences one after the other
                parts, firsts = [], []
                for s in sents:
                    for piece in tts_pieces(normalize(s)) if split else [None]:
                        x, rate, f = await say(s, piece)
                        parts.append(x); firsts.append(f)
            save(name, parts, rate, sents=sents, first_ms=firsts, encode_ms=[], **info)

        jobs = []
        if "length" in SETS:
            jobs += [sentences(f"length/l{secs:02d}_{'id' if n == 0 else 'en'}_{k}.wav", [t], set="length", target_s=secs,
                               lang="id" if n == 0 else "en", sample=k, variant="served")
                     for secs, texts in LENGTH.items() for n, t in enumerate(texts) for k in range(3)]
        if "tail" in SETS:
            for lang, (tail, clauses) in TAIL.items():
                for n in (0, 1, 3, 5):
                    text = ", ".join(clauses[:n] + [tail]) if n else tail[0].upper() + tail[1:]
                    for variant in ("served", "split") if n >= 3 else ("served",):
                        jobs += [sentences(f"tail/t{n}_{lang}_{variant}_{k}.wav", [text], split=variant == "split", set="tail",
                                           clauses=n, lang=lang, sample=k, variant=variant) for k in range(6)]
        if "short" in SETS:
            for i, (a, b) in enumerate(SHORT):
                for k in range(8):
                    jobs.append(sentences(f"short/s{i}_separate_{k}.wav", [a, b], set="short", case=i, variant="separate", sample=k))
                    jobs.append(sentences(f"short/s{i}_merged_{k}.wav", [a + " " + b], set="short", case=i, variant="merged", sample=k))
        if "seword" in SETS:
            jobs += [sentences(f"seword/{w}_{form}_{k}.wav", [t.format(spell(w))], set="seword", word=w, variant=form, sample=k)
                     for w, t in SEWORDS.items() for form, spell in FORMS.items() for k in range(8)]
        if "titles" in SETS:
            jobs += [sentences(f"titles/t{i}_{k}.wav", [t], set="titles", case=i, sample=k, variant="served")
                     for i, t in enumerate(TITLES) for k in range(2)]
        if "replies" in SETS:
            jobs += [sentences(f"replies/r{i:02d}_served_{k}.wav", s, set="replies", reply=i, variant="served", sample=k)
                     for i, s in enumerate(REPLIES) for k in range(3)]
        await asyncio.gather(*jobs)
        if "ttfb" in SETS:
            for _ in range(2):
                for s in REPLIES:
                    solo.append((await say(s[0]))[2])
    # merged short: the bound between the two sentences is unknown, smooth_ana finds it from the silence
    for m in manifest:
        if m["set"] == "short" and m["variant"] == "merged":
            m["sents"] = re.split(r"(?<=[.!?]) ", m["sents"][0], maxsplit=1)
    json.dump({"rate": manifest[0]["rate"] if manifest else 48000, "solo_ms": {"fixed": solo, "cont": [], "encode": []}, "files": manifest},
              open(OUT + "/manifest.json", "w"), ensure_ascii=False, indent=0)
    print("rendered", len(manifest), "solo first byte ms", sorted(round(x) for x in solo), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
