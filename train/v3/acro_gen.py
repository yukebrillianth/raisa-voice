"""Render the acronym test set (acro.py): every LoRA x raw/phonetic text x temperature x SAMPLES, served style.

    python acro_gen.py <out dir> name=/lora/dir [name=/lora/dir ...]     # env TEMPS=0.7_0.5_0.3, SAMPLES=8
    SPELLINGS=1 python acro_gen.py ...        # instead: every spelling in acro.SPELLINGS (second round)
Samples are cut at 30 s (188 steps of 0.16 s): at temperature 0.3 half of them never stop.
Writes <out>/<name>/<acronym>_<lang><n>_<form>_t<temp>_<k>.wav and <out>/manifest.json for acro_judge.py.
"""
import asyncio, json, os, sys, wave
import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

from acro import cases

OUT, CKPTS = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
TEMPS = os.environ.get("TEMPS", "0.7").replace("_", " ").split()
SAMPLES = int(os.environ.get("SAMPLES", "8"))
V = "/yuke/raisa-voice/voices/"


async def main():
    pool = VoxCPM.from_pretrained("openbmb/VoxCPM2", max_num_seqs=16, max_model_len=4096,
                                  gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", "0.4")),
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=len(CKPTS), max_lora_rank=32))
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    wav = open(V + "raisa-ref.wav", "rb").read()
    voice = {"prompt_id": await pool.add_prompt(wav, "wav", open(V + "raisa-ref.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(wav, "wav")}
    for name, path in CKPTS.items():
        await pool.register_lora(name, path)
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
    gate, manifest = asyncio.Semaphore(16), []

    async def one(m):
        async with gate:
            chunks = [w async for w in pool.generate(target_text=m["text"], lora_name=m["model"], temperature=float(m["temp"]), max_generate_length=188, **voice)]
        pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(f"{OUT}/{m['file']}", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)

    for name in CKPTS:
        for acronym, n, lang, forms, left, target, right in cases(spellings=bool(os.environ.get("SPELLINGS"))):
            for f, (form, text) in enumerate(forms.items()):
                for temp in TEMPS:
                    for k in range(SAMPLES):
                        tag = form if form in ("raw", "phon") else f"s{f}"
                        manifest.append({"file": f"{name}/{acronym}_{lang}{n}_{tag}_t{temp}_{k}.wav", "model": name, "acronym": acronym,
                                         "lang": lang, "carrier": n, "form": form, "temp": temp, "text": text,
                                         "left": left, "target": target, "right": right})
    json.dump(manifest, open(OUT + "/manifest.json", "w"), ensure_ascii=False)
    await asyncio.gather(*map(one, manifest))
    print("rendered", len(manifest), flush=True)
    await pool.stop()

if __name__ == "__main__":  # the pool spawns a worker that imports this file
    asyncio.run(main())
