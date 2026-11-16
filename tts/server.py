"""VoxCPM2 streaming TTS server: POST /v1/tts {"text"} -> raw PCM16LE mono chunks.

Runs on nano-vllm-voxcpm, which batches concurrent requests (several robots talking
at once) instead of queueing them. The plain voxcpm package keeps one static KV
cache per model, so it can only serve one sentence at a time.

The sample rate is returned in the X-Sample-Rate header (VoxCPM2 outputs 48 kHz).
When the HTTP client disconnects (barge-in), the generator is closed and nano-vllm
cancels the sequence.
"""

import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
from pydantic import BaseModel

# LoRA trained on the voice actor's 100 clips (dataset-baru, step 700 of 1000), plus one calm
# clip of the same speaker as prompt and reference. Picked by measured speech, see README.
LORA_DIR = os.environ.get("VOICE_LORA", "/voices/raisa-lora")
VOICE_WAV = os.environ.get("VOICE_WAV", "/voices/raisa-ref.wav")
VOICE_TEXT_FILE = os.environ.get("VOICE_TEXT_FILE", "/voices/raisa-ref.txt")
# Share of the whole GPU this process may use (weights + KV cache); llm and asr share the card.
GPU_MEMORY_UTILIZATION = float(os.environ.get("TTS_GPU_MEMORY_UTILIZATION", "0.3"))

state: dict = {}


async def synth(text: str):
    async for wav in state["pool"].generate(target_text=text, **state["voice"]):
        yield (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        max_num_seqs=8,
        max_model_len=4096,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # Must match the trained adapter (r 32 on LM and DiT attention).
        lora_config=LoRAConfig(enable_lm=True, enable_dit=True, max_loras=1, max_lora_rank=32),
    )
    await pool.wait_for_ready()
    state["pool"] = pool
    # Comes back as a plain dict from the worker process.
    state["sample_rate"] = (await pool.get_model_info())["output_sample_rate"]
    wav = open(VOICE_WAV, "rb").read()
    prompt_text = open(VOICE_TEXT_FILE, encoding="utf-8").read().strip()
    await pool.register_lora("raisa", LORA_DIR)
    # Prompt (continue the reference recording) + reference latents at temperature 0.7: with the
    # LoRA, sentence onsets match the actor's own recordings (pitch 1.13x the rest vs 1.16x real;
    # 1.51x with the old GEMINI.wav zero-shot clone).
    state["voice"] = {
        "lora_name": "raisa",
        "prompt_id": await pool.add_prompt(wav, "wav", prompt_text),
        "ref_audio_latents": await pool.encode_latents(wav, "wav"),
        "temperature": 0.7,
    }
    # Warmup so the first visitor does not pay for kernel autotuning.
    started = time.time()
    async for _ in synth("Halo, selamat datang."):
        pass
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
