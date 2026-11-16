"""A/B files for listening: current zero-shot GEMINI, zero-shot calm dataset clip, chosen LoRA (2 samples each)."""
import asyncio, json, os, sys, wave
import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

LORA, LORA_REF = sys.argv[1], sys.argv[2] == "ref"   # checkpoint dir, "ref" or "noref"
D = "/data/"
SENTS = [
    "Halo, saya Raisa.",
    "Selamat datang di Teaching Factory Pusat Robotika ITS. Ada yang bisa saya bantu?",
    "Ruang rapat ada di lantai dua. Silakan ikuti saya. Kita berangkat sekarang ya.",
    "Tiketnya seratus lima puluh ribu rupiah, buka pukul delapan sampai enam belas.",
    "Mohon maaf, saya tidak bisa mengantar Anda ke perpustakaan.",
    "Sebentar ya, saya cek dulu.",
    "Yes, I can speak English. How can I help you today?",
    "Permisi, saya mau lewat. Terima kasih.",
]

async def main():
    pool = VoxCPM.from_pretrained("openbmb/VoxCPM2", max_num_seqs=16, max_model_len=4096, gpu_memory_utilization=0.5,
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=1, max_lora_rank=32))
    await pool.wait_for_ready()
    rate = (await pool.get_model_info())["output_sample_rate"]
    gem, calm = open("/voices/GEMINI.wav", "rb").read(), open(D + "wavs/utt_003.wav", "rb").read()
    calm_text = json.loads(open(D + "voxcpm2_lora_100_dataset.jsonl").readlines()[2])["text"]
    v_gem = {"prompt_id": await pool.add_prompt(gem, "wav", open("/voices/GEMINI.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(gem, "wav")}
    v_calm = {"prompt_id": await pool.add_prompt(calm, "wav", calm_text), "ref_audio_latents": await pool.encode_latents(calm, "wav")}
    await pool.register_lora("raisa", LORA)
    cfgs = {"A_sekarang_gemini": v_gem, "B_zeroshot_dataset": v_calm,
            "C_lora": {"lora_name": "raisa", **(v_calm if LORA_REF else {})}}
    os.makedirs("/ab", exist_ok=True)
    lines = []
    async def one(cfg, kw, i, text, k):
        chunks = [w async for w in pool.generate(target_text=text, temperature=0.7, **kw)]
        pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(f"/ab/{i + 1:02d}_{cfg}_{k + 1}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    for cfg, kw in cfgs.items():
        await asyncio.gather(*(one(cfg, kw, i, t, k) for i, t in enumerate(SENTS) for k in range(2)))
    with open("/ab/index.txt", "w") as f:
        f.write("NN_<config>_<sampel>.wav\n")
        f.write("A_sekarang_gemini  = yang dipakai sebelum LoRA (zero-shot, GEMINI.wav prompt+ref, t0.7)\n")
        f.write("B_zeroshot_dataset = zero-shot, klip dataset utt_003 sebagai prompt+ref\n")
        f.write(f"C_lora             = LoRA {os.path.basename(LORA)} {'+ utt_003 prompt+ref' if LORA_REF else 'tanpa referensi'}\n\n")
        for i, t in enumerate(SENTS): f.write(f"{i + 1:02d}: {t}\n")
    await pool.stop()

if __name__ == "__main__":
    asyncio.run(main())
