"""Is the merged checkpoint the same voice as base + LoRA? Same text and seed, one sentence at a time, prompt + reference
raisa-ref.wav as served, at temperature 0 (nano-vllm scales the diffusion noise by the temperature: no noise at all) and at
the served 0.7 with seed 0 (the same noise in both). Cut at 250 steps (40 s): at temperature 0 most sentences never stop.

    python merge_test.py <lora dir> <merged dir> <out dir>      # tts image, GPU, stack stopped
Writes <out>/{unmerged,merged}/NN.wav and <out>/equivalence.json: per sentence the length of each output, the max and
mean absolute sample difference over the common length and the correlation; plus the whole-run RTF of each.
"""
import asyncio, json, os, sys, time, wave

import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

LORA, MERGED, OUT = sys.argv[1:4]
V = "/yuke/raisa-voice/voices/"
TEXTS = [
    "Halo, selamat datang di Teaching Factory.",
    "Ruang rapat ada di lantai dua, tepat di sebelah kanan lift.",
    "Silakan pindai kode QR di meja depan untuk mendaftar.",
    "Server ini memakai GPU dengan memori empat puluh delapan GB.",
    "Acaranya mulai pukul sembilan WIB di aula utama.",
    "Total belanja Anda seratus dua puluh lima ribu rupiah, bisa dibayar pakai QRIS.",
    "Please follow me to the main lobby, it is on the left.",
    "Welcome to ITS, the campus opens at seven fifteen on weekdays.",
    "Selama pameran berlangsung, Anda bisa mengunjungi lab robotika di lantai satu, lalu mampir ke meja informasi untuk mengambil brosur.",
    "Baik, saya antar Anda sekarang.",
]


async def render(name, model, lora, temp):
    pool = VoxCPM.from_pretrained(model, max_num_seqs=8, max_model_len=4096, gpu_memory_utilization=0.3,
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=1, max_lora_rank=32) if lora else None)
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    wav = open(V + "raisa-ref.wav", "rb").read()
    voice = {"prompt_id": await pool.add_prompt(wav, "wav", open(V + "raisa-ref.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(wav, "wav"), "temperature": temp, "seed": 0, "max_generate_length": 250}
    if lora:
        await pool.register_lora("raisa", lora)
        voice["lora_name"] = "raisa"
    for t in TEXTS[:3]:  # warm up the kernels, as the server does
        async for _ in pool.generate(target_text=t, **voice):
            pass
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    outs, secs, audio = [], 0.0, 0.0
    for i, t in enumerate(TEXTS):
        t0 = time.monotonic()
        x = np.concatenate([c async for c in pool.generate(target_text=t, **voice)])
        secs += time.monotonic() - t0
        audio += len(x) / rate
        outs.append(x)
        with wave.open(f"{OUT}/{name}/{i:02d}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    await pool.stop()
    return outs, secs / audio


async def main():
    def diff(x, y):
        n = min(len(x), len(y))
        d = np.abs(x[:n] - y[:n])
        return {"len": [len(x), len(y)], "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                "corr": float(np.corrcoef(x[:n], y[:n])[0, 1])}
    res = {"texts": TEXTS}
    for temp in (0.0, 0.7):
        a, rtf_a = await render(f"t{temp}_unmerged", "openbmb/VoxCPM2", LORA, temp)
        again, _ = await render(f"t{temp}_unmerged_again", "openbmb/VoxCPM2", LORA, temp)  # run-to-run noise
        b, rtf_b = await render(f"t{temp}_merged", MERGED, None, temp)
        res[f"t{temp}"] = {"rtf_unmerged": rtf_a, "rtf_merged": rtf_b,
                           "unmerged_vs_merged": [diff(x, y) for x, y in zip(a, b)],
                           "unmerged_vs_unmerged_again": [diff(x, y) for x, y in zip(a, again)]}
        json.dump(res, open(f"{OUT}/equivalence.json", "w"), indent=1)
        for k in ("unmerged_vs_unmerged_again", "unmerged_vs_merged"):
            print(f"temperature {temp} {k}")
            for i, r in enumerate(res[f"t{temp}"][k]):
                print(f"  {i:2d} len {r['len']} max {r['max_abs']:.4f} mean {r['mean_abs']:.5f} corr {r['corr']:.4f}")
        print(f"temperature {temp} rtf unmerged {rtf_a:.3f} merged {rtf_b:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
