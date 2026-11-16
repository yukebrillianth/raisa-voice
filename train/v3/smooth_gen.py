"""Render the continuity experiments on one LoRA, served style (raisa-ref prompt + reference, t0.7).

    python smooth_gen.py <out dir> /lora/dir            # tts image, repo at /rv, ~/yuke at /yuke
replies  12+ multi-sentence answers x 3 samples, the sentences of each rendered as
         fixed    every sentence on the fixed prompt (what is served)
         whole    the whole answer in one request (cannot be streamed; the continuity to aim for)
         cont     sentence k>1 continues the generated audio of sentence k-1 (+ its text) as prompt
         conttrim same, the previous audio cut at its last voiced sample first
         refcont  prompt = reference clip followed by the previous sentence
short    "Baik." / "Oke." / "Siap!" + a next sentence: separate requests vs one merged request, 8 samples
lexicon  colok-words written plain vs with "ch", 8 samples
Writes wavs and <out>/manifest.json (per file: texts, sentence boundaries in samples, TTS first-chunk ms).
"""
import asyncio, io, json, os, sys, time, wave
import numpy as np

V = "/yuke/raisa-voice/voices/"
REPLIES = [
    ["Baik, saya antar Anda ke lobi sekarang.", "Silakan ikuti saya."],
    ["Ruang rapat ada di lantai dua.", "Silakan naik lift di sebelah kanan.", "Nanti belok kiri setelah keluar lift."],
    ["Mohon maaf, saya tidak bisa mengantar Anda ke perpustakaan.", "Lokasi itu di luar jangkauan saya.", "Apakah ada tujuan lain yang bisa saya bantu?"],
    ["Selamat datang di Teaching Factory.", "Nama saya Raisa.", "Ada yang bisa saya bantu hari ini?"],
    ["Toilet ada di ujung lorong ini.", "Jalan lurus saja, lalu belok kanan."],
    ["Tentu, saya bisa menjelaskan.", "Lab robotika dipakai untuk riset robot pelayan dan robot sepak bola.", "Mahasiswa bekerja di sana setiap hari.", "Kalau mau, saya antar Anda ke sana sekarang."],
    ["Pameran dibuka pukul sembilan pagi.", "Tiketnya bisa dibeli di meja depan.", "Harganya dua puluh lima ribu rupiah per orang."],
    ["Terima kasih sudah menunggu.", "Kita sudah sampai di ruang rapat."],
    ["Sebentar ya, saya cek dulu jadwalnya.", "Rapat berikutnya mulai pukul dua siang.", "Ruangannya masih kosong sampai saat itu."],
    ["Maaf, saya belum mengerti pertanyaan Anda.", "Bisa tolong diulangi dengan lebih pelan?"],
    ["Sure, I can take you to the lobby.", "Please follow me."],
    ["The meeting room is on the second floor.", "Take the lift on your right.", "It is the first door after the lift."],
    ["Welcome to the Teaching Factory.", "My name is Raisa.", "How can I help you today?"],
    ["I am sorry, I cannot go to the library.", "It is outside my area.", "Is there anywhere else you would like to go?"],
]
SHORT = [("Baik.", "Saya antar Anda ke lobi sekarang."), ("Oke.", "Silakan ikuti saya ke ruang rapat."),
         ("Siap!", "Kita berangkat ke lab robotika sekarang."), ("Tentu.", "Toilet ada di ujung lorong ini."),
         ("Sure.", "Please follow me to the lobby.")]
LEXICON = {"colok": "Silakan {} kabelnya di sini.", "colokan": "Ada {} listrik di bawah meja.",
           "dicolokkan": "Kabelnya sudah {} ke laptop.", "mencolok": "Warna bajunya sangat {} hari ini."}
N_REPLY, N_CASE = 3, 8


def wav_bytes(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def voiced_end(x: np.ndarray) -> int:
    level = np.abs(x)
    loud = np.flatnonzero(level > max(0.015, level.max() * 0.05))
    return int(loud[-1]) + 1 if len(loud) else len(x)


async def main():
    from nanovllm_voxcpm import VoxCPM  # here, so served_ab.py can import the texts without nano-vllm
    from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

    OUT, LORA = sys.argv[1], sys.argv[2]
    pool = VoxCPM.from_pretrained("openbmb/VoxCPM2", max_num_seqs=16, max_model_len=4096,
                                  gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", "0.4")),
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=1, max_lora_rank=32))
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    ref_wav, ref_text = open(V + "raisa-ref.wav", "rb").read(), open(V + "raisa-ref.txt", encoding="utf-8").read().strip()
    ref = await pool.encode_latents(ref_wav, "wav")
    await pool.register_lora("raisa", LORA)
    base = {"lora_name": "raisa", "ref_audio_latents": ref, "temperature": 0.7, "max_generate_length": 188}
    fixed = {"prompt_latents": ref, "prompt_text": ref_text}
    gate, manifest = asyncio.Semaphore(12), []

    async def say(text, prompt):
        async with gate:
            t0, first, chunks = time.monotonic(), None, []
            async for w in pool.generate(target_text=text, **base, **prompt):
                first = first or (time.monotonic() - t0) * 1000
                chunks.append(w)
        return np.concatenate(chunks), first

    async def encode(x):
        t0 = time.monotonic()
        latents = await pool.encode_latents(wav_bytes(x, rate), "wav")
        return latents, (time.monotonic() - t0) * 1000

    def save(name, parts, **info):
        x = np.concatenate(parts)
        os.makedirs(os.path.dirname(f"{OUT}/{name}"), exist_ok=True)
        open(f"{OUT}/{name}", "wb").write(wav_bytes(x, rate))
        manifest.append({"file": name, "bounds": np.cumsum([len(p) for p in parts]).tolist(), **info})

    async def reply(i, sents, variant, k):
        parts, first_ms, encode_ms = [], [], []
        if variant == "whole":
            x, f = await say(" ".join(sents), fixed)
            parts, first_ms = [x], [f]
        else:
            for n, s in enumerate(sents):
                prompt = fixed
                if n and variant != "fixed":
                    prev = parts[-1][: voiced_end(parts[-1])] if variant == "conttrim" else parts[-1]
                    latents, ms = await encode(prev)
                    encode_ms.append(ms)
                    prompt = ({"prompt_latents": ref + latents, "prompt_text": ref_text + " " + sents[n - 1]} if variant == "refcont"
                              else {"prompt_latents": latents, "prompt_text": sents[n - 1]})
                x, f = await say(s, prompt)
                parts.append(x); first_ms.append(f)
        save(f"replies/r{i:02d}_{variant}_{k}.wav", parts, set="replies", reply=i, variant=variant, sample=k, sents=sents,
             first_ms=first_ms, encode_ms=encode_ms)

    async def short(i, a, b, k):
        xa, _ = await say(a, fixed); xb, _ = await say(b, fixed)
        save(f"short/s{i}_separate_{k}.wav", [xa, xb], set="short", case=i, variant="separate", sample=k, sents=[a, b])
        x, _ = await say(a + " " + b, fixed)
        save(f"short/s{i}_merged_{k}.wav", [x], set="short", case=i, variant="merged", sample=k, sents=[a, b])
        x, _ = await say(a, fixed)
        save(f"short/s{i}_alone_{k}.wav", [x], set="short", case=i, variant="alone", sample=k, sents=[a])

    async def word(w, template, form, k):
        spelled = w.replace("colok", "cholok") if form == "ch" else w
        x, _ = await say(template.format(spelled), fixed)
        save(f"lexicon/{w}_{form}_{k}.wav", [x], set="lexicon", word=w, variant=form, sample=k, sents=[template.format(w)], sent=template.format(spelled))

    # Solo timing first (nothing else on the GPU): first chunk with the fixed prompt vs a continuation, and the encode.
    solo = {"fixed": [], "cont": [], "encode": []}
    for sents in REPLIES[:6]:
        x, f = await say(sents[0], fixed); solo["fixed"].append(f)
        latents, ms = await encode(x); solo["encode"].append(ms)
        _, f = await say(sents[1], {"prompt_latents": latents, "prompt_text": sents[0]}); solo["cont"].append(f)
    jobs = [reply(i, s, v, k) for i, s in enumerate(REPLIES) for v in ("fixed", "whole", "cont", "conttrim", "refcont") for k in range(N_REPLY)]
    jobs += [short(i, a, b, k) for i, (a, b) in enumerate(SHORT) for k in range(N_CASE)]
    jobs += [word(w, t, form, k) for w, t in LEXICON.items() for form in ("plain", "ch") for k in range(N_CASE)]
    await asyncio.gather(*jobs)
    json.dump({"rate": rate, "solo_ms": solo, "files": manifest}, open(OUT + "/manifest.json", "w"), ensure_ascii=False, indent=0)
    print("rendered", len(manifest), {k: round(float(np.median(v))) for k, v in solo.items()}, flush=True)
    await pool.stop()

if __name__ == "__main__":  # the pool spawns a worker that imports this file
    asyncio.run(main())
