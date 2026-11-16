"""Render every eval sentence x 3 samples for each config (baselines and LoRA checkpoints) with nano-vllm."""
import asyncio, glob, json, os, sys, wave
import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

D = "/data/"
EXTRA = ["Halo, saya Raisa.",
         "Selamat datang di Teaching Factory, ada yang bisa saya bantu?",
         "Tiketnya seratus lima puluh ribu rupiah, buka pukul delapan sampai enam belas.",
         "Silakan ikuti saya ke ruang rapat di lantai dua.",
         "Yes, I can speak English. How can I help you today?"]
SAMPLES = 3

async def main():
    hold = [json.loads(l) for l in open(D + "holdout.jsonl")]
    sents = [(f"h{i}", r["text"]) for i, r in enumerate(hold)] + [(f"x{i}", t) for i, t in enumerate(EXTRA)]
    json.dump(sents, open("/eval/sentences.json", "w"), ensure_ascii=False)
    # step 0 is the base model (lora_B starts at zero); 999 and 1000 are the same end point.
    ckpts = [f"/ckpt/step_{s:07d}" for s in (100, 200, 300, 400, 500, 700, 1000)]
    pool = VoxCPM.from_pretrained("openbmb/VoxCPM2", max_num_seqs=16, max_model_len=4096, gpu_memory_utilization=0.6,
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=len(ckpts), max_lora_rank=32))
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    gem = open("/voices/GEMINI.wav", "rb").read()
    calm = open(D + "wavs/utt_003.wav", "rb").read()
    calm_text = json.loads(open(D + "voxcpm2_lora_100_dataset.jsonl").readlines()[2])["text"]
    v_gem = {"prompt_id": await pool.add_prompt(gem, "wav", open("/voices/GEMINI.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(gem, "wav")}
    v_calm = {"prompt_id": await pool.add_prompt(calm, "wav", calm_text), "ref_audio_latents": await pool.encode_latents(calm, "wav")}
    cfgs = {"zs_gemini": v_gem, "zs_data": v_calm}
    for c in ckpts:
        name = os.path.basename(c)
        await pool.register_lora(name, c)
        cfgs[f"lora_{name}_noref"] = {"lora_name": name}
        cfgs[f"lora_{name}_ref"] = {"lora_name": name, **v_calm}

    async def one(cfg, kw, sid, text, k):
        chunks = [w async for w in pool.generate(target_text=text, temperature=0.7, **kw)]
        pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(f"/eval/{cfg}/{sid}_{k}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)

    for cfg, kw in cfgs.items():
        os.makedirs(f"/eval/{cfg}", exist_ok=True)
        await asyncio.gather(*(one(cfg, kw, sid, t, k) for sid, t in sents for k in range(SAMPLES)))
        print("done", cfg, flush=True)
    await pool.stop()

if __name__ == "__main__":
    asyncio.run(main())
