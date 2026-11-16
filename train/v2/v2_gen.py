"""Render holdout + unseen sentences x 3 samples per LoRA, served style (raisa-ref.wav prompt+ref, t0.7).

    python v2_gen.py <out dir> <norm|hybrid> name=/ckpt/dir [name=/ckpt/dir ...]
The text sent is the sentence's `norm` or `hybrid` form from split_v2.json (as gen_eval.py, nano-vllm).
Env: SPLIT (another sentence file), CFG and TIMESTEPS (served defaults 2.0 / 10), GEN_GPU_UTIL.
"""
import asyncio, json, os, sys, wave
import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

OUT, KEY = sys.argv[1], sys.argv[2]
CKPTS = dict(a.split("=", 1) for a in sys.argv[3:])
V = "/yuke/raisa-voice/voices/"
SAMPLES = 3


async def main():
    sents = json.load(open(os.environ.get("SPLIT", "/yuke/raisa-voice-data600/split_v2.json")))["sentences"]
    pool = VoxCPM.from_pretrained("openbmb/VoxCPM2", inference_timesteps=int(os.environ.get("TIMESTEPS", "10")), max_num_seqs=16, max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", "0.6")),
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=len(CKPTS), max_lora_rank=32))
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    wav = open(V + "raisa-ref.wav", "rb").read()
    voice = {"prompt_id": await pool.add_prompt(wav, "wav", open(V + "raisa-ref.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(wav, "wav"), "temperature": 0.7,
             "cfg_value": float(os.environ.get("CFG", "2.0"))}
    for name, path in CKPTS.items():
        await pool.register_lora(name, path)

    async def one(name, s, k):
        chunks = [w async for w in pool.generate(target_text=s[KEY], lora_name=name, **voice)]
        pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(f"{OUT}/{name}/{s['sid']}_{k}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)

    for name in CKPTS:
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        json.dump(KEY, open(f"{OUT}/{name}/text_key.json", "w"))
        await asyncio.gather(*(one(name, s, k) for s in sents for k in range(SAMPLES)))
        print("done", name, flush=True)
    await pool.stop()

if __name__ == "__main__":
    asyncio.run(main())
