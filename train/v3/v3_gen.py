"""Render the v3 test sets through the served synthesis code (tts/server.py synth: Leveller + lead-silence cut).

    python v3_gen.py <out dir> hold name=/lora/dir ...    # split_v3eval.json: holdout 30 + unseen 15 + length 10, x 3
    python v3_gen.py <out dir> acro name=/lora/dir ...    # acro_tests.json: acronyms x 4, words x 8
    python v3_gen.py <out dir> ab name=/lora/dir ...      # ab_tests.json: listening pack, 2 samples
Text per model = its product text: "v2" (speech_text.normalize as served now) for names starting with "v2",
"v3" (v3_text.v3, acronyms raw, no lexicon) for the others. Prompt + reference raisa-ref.wav, temperature 0.7,
cfg 2.0, 10 steps, as served. Samples are cut at 40 s (250 steps): served has no cap, a runaway would only
cost time here and shows up as tempo. Writes <out>/<name>/<id>_<k>.wav.
"""
import asyncio, json, os, sys, wave

import numpy as np

sys.path.insert(0, "/yuke/raisa-voice/tts")
import server  # noqa: E402  served synth(), Leveller and lead cut
from nanovllm_voxcpm import VoxCPM  # noqa: E402
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig  # noqa: E402

OUT, SET = sys.argv[1], sys.argv[2]
CKPTS = dict(a.split("=", 1) for a in sys.argv[3:])
M = "/yuke/raisa-voice-train/data_v3/"
V = "/yuke/raisa-voice/voices/"


def jobs(name):
    key = "v2" if name.startswith("v2") else "v3"
    if SET == "ab":  # (id, [sentences]): one request per sentence, the robot plays them back to back
        return [(f"{k}_{n}", v[key]) for k, v in json.load(open(M + "ab_tests.json")).items() for n in range(2)]
    if SET == "spell":  # spell_tests.json (v3_acro.spell_cases): the same written forms for every model, x 8
        return [(f"{c['id']}_{k}", c[key]) for c in json.load(open(M + "spell_tests.json")) for k in range(8)]
    if SET == "hold":
        return [(f"{s['sid']}_{k}", s[key]) for s in json.load(open(M + "split_v3eval.json"))["sentences"] for k in range(3)]
    tests = json.load(open(M + "acro_tests.json"))
    return [(f"{c['id']}_{k}", c[key]) for c in tests if c.get("only", key) == key
            for k in range(8 if c["kind"] == "word" else 4)]


async def main():
    # BASE=<merged checkpoint dir> and name=none: a LoRA folded into the weights (merge_lora.py), no LoRA at run time
    loras = {n: p for n, p in CKPTS.items() if p != "none"}
    pool = VoxCPM.from_pretrained(os.environ.get("BASE", "openbmb/VoxCPM2"), max_num_seqs=16, max_model_len=4096,
                                  gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", "0.6")),
                                  lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=len(loras), max_lora_rank=32) if loras else None)
    await pool.wait_for_ready()
    server.state["pool"] = pool
    server.state["sample_rate"] = rate = (await pool.get_model_info())["output_sample_rate"]
    wav = open(V + "raisa-ref.wav", "rb").read()
    voice = {"prompt_id": await pool.add_prompt(wav, "wav", open(V + "raisa-ref.txt").read().strip()),
             "ref_audio_latents": await pool.encode_latents(wav, "wav"), "temperature": 0.7, "max_generate_length": 250}
    for name, path in loras.items():
        await pool.register_lora(name, path)
    gate = asyncio.Semaphore(16)

    async def one(name, file, text):
        async with gate:
            pcm = b"".join([c for t in ([text] if isinstance(text, str) else text) async for c in server.synth(t)])
        with wave.open(f"{OUT}/{name}/{file}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)

    for name in CKPTS:  # one model at a time: synth() reads the voice (and so the LoRA) from server.state
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        server.state["voice"] = {**voice, "lora_name": name} if name in loras else voice
        todo = jobs(name)[: int(os.environ.get("LIMIT", "0")) or None]  # LIMIT: smoke test only
        await asyncio.gather(*(one(name, f, t) for f, t in todo))
        print("done", name, len(todo), flush=True)
    await pool.stop()

if __name__ == "__main__":
    asyncio.run(main())
