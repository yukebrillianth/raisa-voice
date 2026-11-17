"""VoxCPM2 streaming TTS server: POST /v1/tts {"text"} -> raw PCM16LE mono chunks.

Runs on nano-vllm-voxcpm, which batches concurrent requests (several robots talking
at once) instead of queueing them. The plain voxcpm package keeps one static KV
cache per model, so it can only serve one sentence at a time.

The sample rate is returned in the X-Sample-Rate header (VoxCPM2 outputs 48 kHz).
When the HTTP client disconnects (barge-in), the generator is closed and nano-vllm
cancels the sequence.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
from pydantic import BaseModel

# LoRA v2: the voice actor's 600 clips with raw-acronym ("hybrid") transcripts, step 280 of 700,
# plus one calm clip of the same speaker as prompt and reference. Picked by measured speech, see
# train/README.md. Served folded into the base weights (train/v3/merge_lora.py -> /voices/raisa-v2-merged,
# VOICE_LORA=none): same measured voice, TTS first byte 104 -> 63 ms, RTF 0.136 -> 0.085 (train/v3/v3_report.md).
# The adapter itself: VOICE_BASE=openbmb/VoxCPM2 VOICE_LORA=/voices/raisa-lora-v2; v1 stays in /voices/raisa-lora.
LORA_DIR = os.environ.get("VOICE_LORA", "none")
BASE_MODEL = os.environ.get("VOICE_BASE", "/voices/raisa-v2-merged")
USE_LORA = LORA_DIR not in ("", "none")
VOICE_WAV = os.environ.get("VOICE_WAV", "/voices/raisa-ref.wav")
VOICE_TEXT_FILE = os.environ.get("VOICE_TEXT_FILE", "/voices/raisa-ref.txt")
# Share of the whole GPU this process may use (weights + KV cache); llm and asr share the card.
GPU_MEMORY_UTILIZATION = float(os.environ.get("TTS_GPU_MEMORY_UTILIZATION", "0.3"))

# Every sentence came out with about 0.5 s of digital silence before its first word (0.36-0.81 s).
# The robot plays sentences back to back, so that made the first word late and the pause between two
# sentences 0.9 s, against 0.41 s in the training recordings. Cut to LEAD_SECS, the pause is the
# sentence's own trailing silence (0.35 s). Measured in train/v3/smooth_report.md.
LEAD_SECS = 0.05
SPEECH_LEVEL = 0.01  # the silence is exact zeros; speech passes this within 5 ms of its onset

state: dict = {}


class Leveller:
    """Holds a sentence near the level it started at.

    VoxCPM2 + LoRA fades over a sentence: the last 2.5 s are 3 dB quieter than the first at 8 s and 5 dB at
    20 s, against 1 dB in the training recordings, and it is heard as a muffled voice (the spectrum does not
    change). The level of the first REF_SECS of speech is the target; a slow average of the level after it
    sets a gain that makes up what falls more than ALLOW_DB below, at most MAX_DB, ramped within each chunk.
    ALLOW_DB keeps the natural decline of about 1 dB. Measured in train/v3/smooth_report.md (level_sim.py).
    """

    REF_SECS, TAU_SECS, ALLOW_DB, MAX_DB, SPEECH_DB = 1.5, 1.5, 1.5, 6.0, -40.0

    def __init__(self, rate: int):
        self.rate, self.energy, self.samples, self.level, self.gain_db = rate, 0.0, 0, None, 0.0

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        power = float(np.mean(wav.astype(np.float64) ** 2)) if len(wav) else 0.0
        target = self.gain_db
        if 10 * np.log10(power + 1e-12) > self.SPEECH_DB:
            if self.samples < self.REF_SECS * self.rate:
                self.energy += power * len(wav)
                self.samples += len(wav)
            else:
                level = 10 * np.log10(power)
                step = min(1.0, len(wav) / self.rate / self.TAU_SECS)
                self.level = level if self.level is None else self.level + step * (level - self.level)
                ref = 10 * np.log10(self.energy / self.samples)
                target = float(np.clip(ref - self.level - self.ALLOW_DB, 0.0, self.MAX_DB))
        ramp = np.linspace(self.gain_db, target, len(wav), dtype=np.float32)
        self.gain_db = target
        return wav * 10 ** (ramp / 20)


async def synth(text: str):
    lead = int(LEAD_SECS * state["sample_rate"])
    held = np.zeros(0, dtype=np.float32)  # silence so far, None once speech has started
    level = Leveller(state["sample_rate"])
    async for wav in state["pool"].generate(target_text=text, **state["voice"]):
        wav = level(wav)
        if held is not None:
            wav = np.concatenate([held, wav])
            loud = np.flatnonzero(np.abs(wav) > SPEECH_LEVEL)
            if not len(loud):
                held = wav[-lead:]
                continue
            wav, held = wav[max(0, loud[0] - lead):], None
        yield (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = VoxCPM.from_pretrained(
        BASE_MODEL,
        max_num_seqs=8,
        max_model_len=4096,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # Must match the trained adapter (r 32 on LM and DiT attention).
        lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=1, max_lora_rank=32) if USE_LORA else None,
    )
    await pool.wait_for_ready()
    state["pool"] = pool
    # Comes back as a plain dict from the worker process.
    state["sample_rate"] = (await pool.get_model_info())["output_sample_rate"]
    wav = open(VOICE_WAV, "rb").read()
    prompt_text = open(VOICE_TEXT_FILE, encoding="utf-8").read().strip()
    if USE_LORA:
        await pool.register_lora("raisa", LORA_DIR)
    # Prompt (continue the reference recording) + reference latents at temperature 0.7: with the
    # LoRA, sentence onsets match the actor's own recordings (pitch 1.13x the rest vs 1.16x real;
    # 1.51x with the old GEMINI.wav zero-shot clone).
    state["voice"] = {
        **({"lora_name": "raisa"} if USE_LORA else {}),
        "prompt_id": await pool.add_prompt(wav, "wav", prompt_text),
        "ref_audio_latents": await pool.encode_latents(wav, "wav"),
        "temperature": 0.7,
    }
    # Warmup so the first visitor does not pay for kernel autotuning. Kernels settle per batch size:
    # without the 2..5 rounds, the first turns of five robots at once waited 250-570 ms for audio.
    async def warm(text: str):
        async for _ in synth(text):
            pass

    # Different lengths, so each round also runs the smaller batches as its sentences finish.
    texts = ["Halo, selamat datang.", "Ada yang bisa saya bantu?", "Silakan ikuti saya ke lobi.",
             "Baik, saya antar Anda sekarang.", "Ruang rapat ada di lantai dua, di sebelah kanan lift."]
    started = time.time()
    for n in (1, 2, 3, 4, 5):
        await asyncio.gather(*map(warm, texts[:n]))
    print(f"warmup {time.time() - started:.2f}s, sample_rate={state['sample_rate']}", flush=True)
    yield
    await pool.stop()


app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": "pool" in state, "sample_rate": state.get("sample_rate")}


@app.post("/v1/tts")
async def synthesize(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    return StreamingResponse(
        synth(text), media_type="application/octet-stream", headers={"X-Sample-Rate": str(state["sample_rate"])}
    )
