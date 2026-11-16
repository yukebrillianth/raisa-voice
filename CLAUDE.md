# raisa-voice: engineering notes

Read this before changing anything. `README.md` says how to run it; this file says what must
not break, how to prove a change works, and what is still open. Section 8 is the only part that
goes stale: update it whenever the served model, the text rules or the open work changes.

## 1. What this is

RAISA is a humanoid service robot (ITS x IOHits). It talked through Gemini Live. This repo is the
local replacement on one RTX 5090: Whisper (ASR) -> Gemma 4 12B (LLM, tools) -> VoxCPM2 + a voice LoRA (TTS), glued by a Pipecat orchestrator. Languages: Indonesian first,
English second. Target: 5 robots at once, first audio under about 0.7 s, and replies that sound
as continuous as Gemini Live.

The robot gateway is another repo (`~/IRIS/raisa-chat`, Bun). Per agent it picks
`voiceBackend: gemini | local` and falls back to Gemini when this stack is down.

## 2. Rules that are not negotiable

- **The server is shared** (`teaching-factory@10.7.101.217`). Never stop, restart or kill a
  process or container that is not `raisa-voice-*`. Check `nvidia-smi` before a GPU job.
- **Hugging Face**: the server's `hf` login belongs to someone else. Use
  `HF_TOKEN=$(cat ~/yuke/.hf_token)` per command. Never `hf auth login`. Never print the token.
  Both HF repos and the GitHub repo stay private.
- **Never commit** `.env`, `voices/`, `out/`, recordings or weights.
- **Commit and push only when the user asks.** Do not touch `~/IRIS/raisa-chat*` unless asked.
- **ASR debug stays off** (`ASR_DEBUG_DIR` empty): it stores visitors' voices.
- **GPU jobs that pause the stack**: run them on the server inside `tmux`, create
  `~/yuke/raisa-voice/TRAINING.lock`, and use a shell `trap` that always removes the lock and
  runs `docker compose up -d`. A job tied to a laptop SSH session leaves the robots without a
  voice when the laptop sleeps.
- **Never pick a model, checkpoint or spelling from one sample.** Generation is random
  (temperature 0.7). Use at least 8 samples per case and report the count that was right.
- **This work is the author's undergraduate thesis** (Teknik Komputer ITS, robotics). Every
  experiment must be repeatable and citable: keep the config, the number of samples, the raw
  numbers and the audio, and write the losing variants down too (they are the ablation tables).
  Reports live next to the scripts (`train/`, HF `v2/`); never overwrite an old report.
- **Where the voice data comes from**: `dataset-baru/` is synthetic speech generated with Gemini
  Live (`generator/generate.mjs`, prebuilt voice "Zephyr"), not a human recording. That is fine
  for research and must be stated as such in the thesis. It is NOT cleared for a product: the
  Gemini API terms forbid using the service to develop competing models. For a product, record
  a human reading the same manifests and retrain (the recipe takes about 11 minutes). INESCO is
  research-only too; `GEMINI.wav`, MMS and Piper voices must not ship either.

## 3. Map

| path | what |
| --- | --- |
| `compose.yaml` | the four services; only port 8765 is published |
| `orchestrator/bot.py` | one Pipecat pipeline per robot: VAD, turn detection, LLM, tools, fillers, context summary |
| `orchestrator/voxcpm_tts.py` | TTS service: sends screen text + word timing, then normalized text to `tts` |
| `orchestrator/speech_text.py` | written text -> spoken text (numbers, Rp, time, units, names, lexicon) |
| `orchestrator/asr_stt.py` | posts each speech segment to `asr`, with the agent's vocabulary as hotwords |
| `orchestrator/protocol.py` | WebSocket messages (mirror Gemini Live) |
| `orchestrator/tests/` | unit tests; run inside the orchestrator container |
| `asr/server.py` | faster-whisper large-v3-turbo |
| `tts/server.py` | nano-vllm-voxcpm, LoRA + reference clip, `POST /v1/tts {"text"}` -> PCM stream |
| `scripts/try_client.py` | fake robot: measures first audio (TTFAB), gaps, barge-in |
| `scripts/render_traps.py` | renders tricky sentences to `out/traps/` for listening |
| `train/`, `train/v2/` | LoRA training and evaluation scripts, `train/README.md` |
| `voices/` (not in git) | `raisa-lora/` (v1), `raisa-lora-v2/`, `raisa-ref.wav` + `.txt` |

Server paths: stack `~/yuke/raisa-voice`, training `~/yuke/raisa-voice-train` (VoxCPM clone at
commit 300e426, all checkpoints), data `~/yuke/raisa-voice-data` (100 clips) and
`~/yuke/raisa-voice-data600`.
Remote copies: GitHub `yukebrillianth/raisa-voice`; HF model `yukebrillianth/raisa-voxcpm2-lora`
(root = v1, `v2/` = v2 + every checkpoint); HF dataset `yukebrillianth/raisa-voice-dataset`
(root = 100 clips, `v600/` = 600 clips, transcripts, holdout).

## 4. Things that broke before (keep them as they are)

| keep | because |
| --- | --- |
| TTS service created with `push_start_frame=True, push_stop_frames=True` | without stop frames there are 3 s of silence after every tool call |
| LLM thinking off (`--reasoning off` and `enable_thinking=false`) | with thinking, first token takes about 6 s |
| llama.cpp image pinned by digest (CUDA 12 build) | the CUDA 13 image fails on this host's driver 590 |
| Whisper language not forced; retry as `id` only if it detects something outside id/en/ms/jw | forcing `id` made Whisper translate English speech into Indonesian |
| Whisper `vad_filter=True`, threshold 0.35 | at 0.5 it dropped "Halo." and "Ya." as silence |
| Audio streams while the sentence is still being generated | waiting for the whole sentence cost +500 ms to first audio |
| Tools registered with `cancel_on_interruption=False`; gateway always forwards `tool_response` | a visitor interrupting must not lose a tool result |
| Screen text is the original text; only the TTS request gets normalized text | "23°C" must stay "23°C" on the chat bubble |
| Scripts that start nano-vllm have `if __name__ == "__main__":` | it spawns worker processes and crashes without the guard |
| After a docker job writes files: `docker run --rm -v <dir>:/o alpine chown -R $(id -u):$(id -g) /o` | files come out root-owned and unreadable |
| `cfg_value` 2.0, `inference_timesteps` 10 | a 6-config sweep showed no gain; 16 steps cost 1.5x compute |
| Leading silence of each sentence cut to 50 ms (`tts/server.py` synth) | VoxCPM starts every sentence with 0.36-0.81 s of zeros: first word late, 0.9 s between sentences |
| `Leveller` in `tts/server.py` | without it a 20 s sentence ends 4-5 dB quieter than it starts, heard as a muffled voice |
| Title dots protected in the sentence splitter (`bot.TitleDots`) | "Prof. Dr. Ir." was sent to the TTS as a sentence of its own |
| One-word English sentence held for the next one (`bot.SentenceAggregator`) | "Sure." alone is said Indonesian "sur" 8/8 |

Other facts worth knowing:

- nano-vllm batches whole requests from different robots. It cannot interleave chunks of one request.
- `docker compose up -d orchestrator` also recreates `asr` and `tts` when their config changed.
- VoxCPM has no phoneme parser. `{HH AH0}` (CMUdict) and `{ni3}` (pinyin) work only because the
  base model saw them in training. There is no Indonesian phoneme input.
- VoxCPM's own `normalize=True` only knows Chinese and English. We never use it.
- A one-word sentence has no language context: English "Sure." alone is said Indonesian "sur" (0/8);
  Indonesian "Oke.", "Baik.", "Siap!" alone are right (8/8).

## 5. Text rules (the part most people get wrong)

**The text sent to TTS at runtime must look exactly like the transcripts the served LoRA was
trained on.** A spelling the model never saw is read by guessing: "we i be" on a model trained
with raw "WIB" comes out as English "wi ai bi".

The user's pronunciation rules (authoritative):

| written | spoken |
| --- | --- |
| QRIS, PDF, kWh | "kyuris", "pidief", "kilowatt hour" |
| USB TV VIP NFC GB IT ID QR API GPU CPU SSD AI LED PC (capitals only) | English letter names, also inside Indonesian sentences |
| other acronyms: ITS, HDMI, KTP, WIB, UMKM... | letter names of the sentence's language |
| WITA BRIN SIM RAM LAN SKEM EXPO SPARK UNAIR HIMATEKTRO, robot / person / city names in capitals | read as a word; MAGE is "ma-ge", IRIS is "ai-ris" |
| 23°C, 450 m², 4,2 GHz, 20 km/jam, Rp 45.000, 09.00 | spelled out in the sentence's language |
| Mbps, psi, dB, USB-C | "em be pe es", "pe es i", "di bi", "yu es bi si" |

What was measured about acronyms:

- Raw numbers and symbols are misread ("23°C" -> "degri", "Rp 45.000" -> "55.000.000").
  They always go through the normalizer.
- Raw acronyms are **unstable between samples**: the same text gave "GPU" right once and
  "gepiu" the next time, "HDMI" -> "hasdmi".
- Whisper cannot judge letter names: it writes "GPU" whatever was said. Use a phoneme
  recogniser or ears.
- Respelling a misread word works if tested: "colokkan" was read with a k; "cholokkan" is right
  3/3. The old spelling "tj" made other words worse.

## 6. How to prove a change works

Run from `~/yuke/raisa-voice` on the server.

```bash
docker compose exec orchestrator python -m pytest -q tests          # must stay green
docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
  python /scripts/try_client.py ws://orchestrator:8765/ws /out/q16.wav --barge-in /out/b16.wav
docker compose run --rm --no-deps -e PYTHONPATH=/app -v ./scripts:/scripts -v ./out:/out \
  orchestrator python /scripts/render_traps.py                      # then listen to out/traps/
```

Baselines to compare against (same clips, or the comparison means nothing):

| measure | value |
| --- | --- |
| first sound, 1 robot, `q16.wav` (`try_client.py` "first sound"; "TTFAB" is the first byte) | 494-608 ms (first byte 494-587) |
| barge-in: audio stops after the visitor starts | about 330 ms, 0 stale frames, 0 late words |
| TTS first byte, alone | about 112 ms (66-189), and it is speech; RTF 0.123 |
| 5 robots at once, `scripts/bench_robots.sh` (`q16.wav`) | first sound 1226 ms median (1097-2618); per stage asr ~370, LLM first token ~400, TTS ~216 ms |
| join between sentences of a reply | 0.41 s silence, as in the training audio |
| VRAM | about 23 of 32 GB (llm 11, tts 8.4, asr 2.8) |

Voice quality is scored with `train/v2/v2_gen.py` + `v2_ana.py` on a fixed holdout of 30 clips
plus 15 unseen sentences, 3 samples each: Whisper CER/WER against the **original** text, tempo
against the real recording, sentence-onset pitch ratio, f0. The real recordings score onset 1.22
with 30/90 clips above 1.3x, so that is the target, not zero. Differences below 0.01 CER are noise.

## 7. Training

- Official VoxCPM LoRA script and config: LR 1e-4, rank 32 / alpha 32 on LM and DiT, effective
  batch 16. About 11 min for 700 steps on the 5090. Needs the stack stopped (VRAM).
- Dataset format: jsonl `{audio, text, ref_audio?}`, 30-50% of rows with `ref_audio`, clips 3-30 s,
  transcript word-for-word equal to the audio.
- Pick the checkpoint by measured speech, never by loss (validation loss is lowest at step 100).
- 100 clips -> 600 clips did not move the Whisper metrics. More of the same data does not help;
  new data must add what is missing (see section 8).
- Always keep the same holdout so runs stay comparable (`split_v2.json` on HF `v600/`).

## 8. State and open work (update this section)

As of 2026-11-17 (measurements: `train/v3/smooth_report.md`, `train/v3/acro_report.md`):

- Served: LoRA v2-hybrid step 280 (`voices/raisa-lora-v2`), acronyms sent raw, overrides QRIS ->
  "kyuris", WIB -> "we i be", colokkan/dicolokkan/colokan -> "ch", seenaknya -> "se-enaknya";
  academic titles spelled out; lead silence cut, `Leveller`, batch warmup in `tts/server.py`;
  LLM settable by env. v1 stays in `voices/raisa-lora` (`VOICE_LORA=/voices/raisa-lora`, compose passes it).
- **Known wrong** (acronym baseline in `acro_report.md`, two judges): raw GPU "gepiu", GB "gebe",
  API "api", PDF "pedeef", HDMI "hasdemi" on hybrid (66% of Indonesian acronym readings right);
  "we i be" is read "wi ai bi" on every model (a lone "i" is English "ai"). The training audio
  itself reads several of these wrong (`out/acronym_audit_600.json`), so the fix is the verified
  v3 acronym set (`dataset-baru/voxcpm2_lora_v3_acronym.jsonl`), not a lexicon. Also "seekor" ->
  "se-kor" 5/8 and "keenam" -> "ke-nam" 7/8, whatever the spelling: needs data.
- Measured and rejected (do not retry without new evidence): cross-sentence continuation (pitch drifts
  down, text repeated/skipped, joins no better than the trim), splitting long sentences at commas
  (1 dB less fade, +0.4 s pause per cut), temperature 0.5/0.3 (worse, samples that never stop),
  5 Whisper workers (no faster: GPU-bound), merging Indonesian one-word sentences.

Open work, in order:

1. Train v3 with the verified acronym set (being prepared separately); score it with
   `train/v3/acro_gen.py` + `acro_judge.py` (phoneme judge, validated 12/13 against the user's ears)
   and `gemini_judge.py` (gemini-4-hydrogen, 3 asks) on the same test set, against `acro_report.md`.
   Then set the acronym text rules to match the new transcripts.
2. Split turns under load: with 5 robots, a visitor's pause mid-request ends the turn when its
   transcript is late, the rest interrupts and restarts the LLM (+1.3 s, 4-12 of 20 turns). Smart
   Turn / user-turn-stop settings, not the voice. Whisper with 5 robots ~380 ms (GPU-bound).
3. Pitch spread between the sentences of one reply is higher than in the training audio (1.57 vs
   1.15 semitones, English 2.43): every sentence restarts from the prompt clip.
4. Dataset v3 general part: the 280-line script (`voxcpm2_lora_v3_280_dataset.jsonl`, short answers, long
   paragraphs, conversational, titles) plus `dataset-baru/voxcpm2_lora_v3_tambahan.jsonl`
   (English). Add sentences longer than 10.5 s (none in the 600 clips) and "seekor"/"keenam".
5. Never tested: a real robot microphone, echo cancellation, the kiosk through the gateway, the
   Leveller on the robot's speaker, an external LLM through the env variables.
6. Port to NVIDIA GH200 (ARM64 CPU, 96 GB GPU) at Indosat's AI-RAN lab, a likely second host.
   llama.cpp has an official arm64 `server-cuda` image. CTranslate2's arm64 wheel on PyPI is
   CPU-only, so `asr` needs CTranslate2 built from source with CUDA (or another Whisper runtime).
   `tts` needs flash-attn compiled for arm64. Nothing here has been run on ARM64 yet. Measure
   robot-to-datacentre network latency first: it adds directly to first audio.
7. With the acronym judges in place, the cheap weight-level options can be scored: average
   neighbouring LoRA checkpoints, scale the LoRA below 1.0, train rank 64, and self-distillation
   (generate acronym sentences, keep only samples the judge accepts, add them to the training set
   at no more than about 10% of the data).
8. Thesis evidence still missing: the same scenarios measured on Gemini Live (latency per
   stage, first audio), a listening test with people (MOS, and A/B against Gemini Live; the pack
   `out/smooth_ab/` is a start), ASR error rate on real robot-microphone recordings, and
   tool-call success rate.

Decisions already made (do not reopen without new evidence):

| decision | reason |
| --- | --- |
| llama.cpp, not vLLM | vLLM audio for Gemma 4 was nightly-only; llama.cpp gives 160 tok/s |
| separate Whisper, not audio straight into Gemma | direct audio failed inside tool conversations |
| WebSocket, not WebRTC | fine on the campus LAN; revisit only if jitter is measured |
| no Indonesian phoneme dataset | the model has no such input, 65 min cannot teach one |
| no full fine-tune | needs about 40 GB VRAM and hundreds of hours; LoRA reaches about 98% of it |
