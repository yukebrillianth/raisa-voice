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
| `orchestrator/everyday.py` | URLs, e-mail, Jl./No./Kec., 2.5 GHz, 3x, S1/D3, 5G, $20, 1/2, Roman numerals, ± & /, e.g./vs.; dots that never end a sentence |
| `orchestrator/text_tagger.py`, `langid.py` | optional sentence model (int8 ONNX on CPU, `models/text_tagger`): language of each word, reading of each capital; function words otherwise |
| `orchestrator/acronym_clf.py`, `acronym_data.csv` | model for capitals nobody listed (word / Indonesian letters / English letters); trained in the Dockerfile |
| `orchestrator/titles.py`, `title_lexicon.json` | titles around names (Prof., Dr. (H.C.), S.T., Sp.PD-KGEH, Ph.D.), lexicon + grammar; built by `train/titles/` |
| `orchestrator/asr_stt.py` | posts each speech segment to `asr`, with the agent's vocabulary as hotwords |
| `orchestrator/protocol.py` | WebSocket messages (mirror Gemini Live) |
| `orchestrator/tests/` | unit tests; run inside the orchestrator container |
| `asr/server.py` | faster-whisper large-v3-turbo |
| `tts/server.py` | nano-vllm-voxcpm, merged voice weights (or base + LoRA) + reference clip, `POST /v1/tts {"text"}` -> PCM stream |
| `scripts/try_client.py` | fake robot: measures first audio (TTFAB), gaps, barge-in |
| `scripts/render_traps.py` | renders tricky sentences to `out/traps/` for listening |
| `train/tagger/` | sentence generator with blind judges, XLM-R fine-tune + ONNX export, report against the rules (`README.md`) |
| `train/acronyms/` | acronym data generator with blind judges, model numbers (`README.md`) |
| `train/titles/` | title lexicon: seed, LLM generator with blind judges, build + synthetic evaluation (`README.md`) |
| `train/`, `train/v2/`, `train/v3/` | LoRA training, evaluation and merge scripts, `train/README.md`, `train/v3/v3_report.md` |
| `voices/` (not in git) | `raisa-v2-merged/` (served: v2 folded into the base, 4.6 GB), `raisa-lora/` (v1), `raisa-lora-v2/`, `raisa-ref.wav` + `.txt` |

Server paths: stack `~/yuke/raisa-voice`, training `~/yuke/raisa-voice-train` (VoxCPM clone at
commit 300e426, all checkpoints), data `~/yuke/raisa-voice-data` (100 clips) and
`~/yuke/raisa-voice-data600`, `~/yuke/raisa-voice-data-v3` (v3 clips); v3 manifests and test sets in
`~/yuke/raisa-voice-train/data_v3`, checkpoints `ckpt_v3`, renders `eval_v3`. The disk is about 99% full: check `df -h ~`.
Remote copies: GitHub `yukebrillianth/raisa-voice`; HF model `yukebrillianth/raisa-voxcpm2-lora`
(root = v1, `v2/` = v2 + every checkpoint, `v3/` = v3 + checkpoints + eval, `v3/merged/` = the served merged v2); HF dataset `yukebrillianth/raisa-voice-dataset`
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
| Served weights = `voices/raisa-v2-merged` (`VOICE_BASE`, `VOICE_LORA=none`); a LoRA change means re-running `train/v3/merge_lora.py` | the adapter path stays as rollback (`VOICE_BASE=openbmb/VoxCPM2 VOICE_LORA=/voices/raisa-lora-v2`); serving a fresh adapter on the merged base would apply it twice |
| `Leveller` in `tts/server.py` | without it a 20 s sentence ends 4-5 dB quieter than it starts, heard as a muffled voice |
| Title dots protected in the sentence splitter (`bot.TitleDots`) | "Prof. Dr. Ir." was sent to the TTS as a sentence of its own |
| A line break ends a sentence (`bot.TitleDots._line_end`); a line with no end mark gets a "." before the TTS | "**1. Universitas Airlangga**" and the line under it went to the voice as one sentence |
| HDMI spelled "ha-dé-ém-i", not with English letters | measured 8/8 (raw 1/8); a local rewrite moved it to English letters by mistake |
| One-word English sentence held for the next one (`bot.SentenceAggregator`) | "Sure." alone is said Indonesian "sur" 8/8 |

Other facts worth knowing:

- nano-vllm batches whole requests from different robots. It cannot interleave chunks of one request.
- `docker compose up -d orchestrator` also recreates `asr` and `tts` when their config changed.
- VoxCPM has no phoneme parser. `{HH AH0}` (CMUdict) and `{ni3}` (pinyin) work only because the
  base model saw them in training. There is no Indonesian phoneme input.
- VoxCPM's own `normalize=True` only knows Chinese and English. We never use it.
- nano-vllm at temperature 0 outputs silence (the diffusion noise is scaled by the temperature), and two runs with the
  same seed at 0.7 differ: compare models by measures over many samples, never sample by sample.
- A one-word sentence has no language context: English "Sure." alone is said Indonesian "sur" (0/8);
  Indonesian "Oke.", "Baik.", "Siap!" alone are right (8/8).

## 5. Text rules (the part most people get wrong)

**The text sent to TTS at runtime must look like the transcripts the served LoRA was trained on,
unless a respelling is measured better.** A spelling the model never saw is read by guessing: "we i be"
on a model trained with raw "WIB" comes out as English "wi ai bi". The one measured respelling rule:
an acronym in an Indonesian sentence (not English-letter, not read as a word) is sent as Indonesian letter
names with é and hyphens, "WIB" -> "wé-i-bé", "HDMI" -> "ha-dé-ém-i" (`speech_text.say_capitals`; WIB, HDMI,
NIK 8/8 vs raw 3/8, 1/8, 5/8 on the served voice).

The user's pronunciation rules (authoritative):

| written | spoken |
| --- | --- |
| QRIS, PDF, kWh | "kyuris", "pidief", "kilowatt hour" |
| USB TV VIP NFC GB IT ID QR API GPU CPU SSD AI LED PC (capitals only) | English letter names, also inside Indonesian sentences |
| other acronyms: ITS, HDMI, KTP, WIB, UMKM... | letter names of the sentence's language |
| WITA BRIN SIM RAM LAN SKEM EXPO SPARK UNAIR HIMATEKTRO, robot / person / city names in capitals | read as a word; MAGE is "ma-ge", IRIS is "ai-ris" |
| 23°C, 450 m², 4,2 GHz, 20 km/jam, Rp 45.000, 09.00 | spelled out in the sentence's language |
| 2019-2024, 08.00-10.00, 5-10 | "... sampai ..." ("to" in English, years as "twenty nineteen") |
| RTX 5090, ISO 9001, ESP32 | "RTX lima puluh sembilan puluh", digits one by one, "ESP tiga puluh dua" |
| any acronym in an English sentence ("send a PDF file") | raw, never Indonesian letters |
| its.ac.id, humas@its.ac.id, Jl. Raya No. 5, S1, 5G, 4K, 1990-an, Bab IV, Perang Dunia II, Elizabeth II, 1/2, 24/7 | "i-té-és dot a-cé dot i-dé", "humas et ...", "Jalan Raya nomor lima", "és satu", "five G", "four K", "...sembilan puluhan", "Bab empat", "Perang Dunia kedua", "Elizabeth kedua", "setengah", "dua puluh empat jam setiap hari" |
| menang 3-1, skor 2-2 | "tiga lawan satu" ("three to one"); without a score word it is a range |
| 2.5 GHz in an Indonesian sentence, 3x, $20, ± & ~ / | "dua koma lima", "tiga kali", "dua puluh dolar", "kurang lebih", "dan", "sekitar", "atau" |
| Mbps, psi, dB, USB-C | "em be pe es", "pe es i", "di bi", "yu es bi si" |

What was measured about acronyms:

- Raw numbers and symbols are misread ("23°C" -> "degri", "Rp 45.000" -> "55.000.000").
  They always go through the normalizer.
- Raw acronyms are **unstable between samples**: the same text gave "GPU" right once and
  "gepiu" the next time, "HDMI" -> "hasdmi".
- Whisper cannot judge letter names: it writes "GPU" whatever was said. Use a phoneme
  recogniser or ears.
- Judging cost (2026-11-17): about 3,700 single-clip gemini-4-hydrogen calls cost the user about
  Rp 450k, mostly thinking tokens (about 635 per call). Use the local phoneme judge first. For a
  Gemini second opinion use `gemini-4-carbon`, thinking "low", 10 clips per request
  (`train/v3/judge_agree.py`): 86% agreement with the old hydrogen reference at about 1/30 of
  the output tokens. Every Gemini judge disagrees with itself on 15-20% of borderline clips, so
  claims in the thesis need a human-labelled subset. gemini-3.8-flash is biased towards "right".
- Training on the readings does not fix them by itself: LoRA v3 (232 verified acronym clips, 3-9 per acronym, raw
  text) left GPU "gepiu", GB "gebe", API, CPU, HD, PDF, WIT at 0-2/8 and moved the total within noise
  (`train/v3/v3_report.md`). v3 is not served. The WIB/WIT override is replaced by the é-hyphen rule; QRIS "kyuris"
  and the colok/seenaknya lexicon stay.
- The phoneme judge moves about 5 points between two renders of the same voice (432 readings): smaller gaps are noise.
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
| TTS first byte, alone (`train/v2/v2_ttfab.py`) | about 63 ms (merged v2; 104 ms with the adapter), RTF 0.085 (0.136) |
| 5 robots at once, `scripts/bench_robots.sh` (`q16.wav`) | first sound about 1170 ms median (864-2387); per stage asr ~370, LLM first token ~300-450, TTS ~180 ms |
| join between sentences of a reply | 0.41 s silence, as in the training audio |
| VRAM | about 23 of 32 GB (llm 11, tts 8.4, asr 2.8) |

Voice quality is scored with `train/v3/v3_gen.py` (the served synth) + `v2_ana.py` / `v3_ana.py` (see `run_v3.sh`), or
the older `train/v2/v2_gen.py` + `v2_ana.py`, on a fixed holdout of 30 clips
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

As of 2026-11-17 (measurements: `train/v3/v3_report.md`, `train/v3/smooth_report.md`, `train/v3/acro_report.md`):

- Served: LoRA v2-hybrid step 280 **folded into the base weights** (`voices/raisa-v2-merged`, `VOICE_LORA=none`):
  same voice within noise, TTS first byte 104 -> 63 ms, RTF 0.136 -> 0.085, five-robot first sound unchanged.
  Acronyms: English-letter ones (USB GPU PDF...) raw; other acronyms in Indonesian sentences as "wé-i-bé"
  letter names (generic rule); QRIS -> "kyuris", colokkan/dicolokkan/colokan -> "ch", seenaknya -> "se-enaknya";
  academic titles spelled out by `titles.py` (2026-11-19: composed degrees, consultant parts, dotless/spaced
  writing, English names in English sentences; 100% on 20k synthetic cases, not yet listened to on the voice); lead silence cut, `Leveller`, batch warmup in `tts/server.py`; LLM settable by env. Adapters kept: `voices/raisa-lora-v2` (rollback), `voices/raisa-lora` (v1).
- LoRA v3 (1089 clips: old 600 cleaned, 303 new general incl. 12-28 s paragraphs, 232 verified acronym clips; HF `v3/`)
  was trained and **not served**: holdout equal to v2, acronyms +2..3 points (noise), English and judge-holdout acronyms
  worse. Two of its acronym clips have the generator's pronunciation note read aloud (`acr_009`, `acr_052`).
- **Known wrong** (`train/v3/v3_report.md`, phoneme judge + gemini-4-carbon): GPU "gepiu", GB "gebe", API "apii",
  CPU, PDF "pedeef" (English-letter list, sent raw) 0-2/8 on every model; SKCK/KTP as é-hyphen 1/8 and 6/8 by the
  phoneme judge (Carbon 8/8): check by ear. Training on 3-9 verified clips each (v3) did not move them. "seekor" 1-3/8
  and "keenam" 0-3/8 under every spelling ("se-ékor", "ke-enam", "ke-ènam" tried): needs data.
- Measured and rejected (do not retry without new evidence): cross-sentence continuation (pitch drifts
  down, text repeated/skipped, joins no better than the trim), splitting long sentences at commas
  (1 dB less fade, +0.4 s pause per cut), temperature 0.5/0.3 (worse, samples that never stop),
  5 Whisper workers (no faster: GPU-bound), merging Indonesian one-word sentences.

- Text (2026-11-19, branch `normalizer-fixes`, from running real LLM replies through `bot.SentenceAggregator`):
  ranges, dates, model numbers, English dates/years and Rp in English sentences are spoken; unlisted capitals go
  to `acronym_clf.py` (80% on unseen tokens vs 43% for the old length rule); a sentence with no function words takes
  the language of the previous sentence of the reply (`voxcpm_tts`); tenant words via `SPEECH_LEXICON_PATH`.
  None of it listened to on the voice yet. `EN_ACRONYM_STYLE=spell` ("ji-pi-yu") exists but is off: unmeasured.
  The text tagger (`train/tagger/`) is wired in but has no trained model yet: without `models/text_tagger`
  nothing changes. When installed it decides the sentence language only when function words do not (none,
  or both languages), and capital readings only after the user's lists and the labelled table.
  Line breaks now end sentences (lists and headings were one TTS request), list numbers read "satu, ...";
  `everyday.py` covers URLs, e-mail, street abbreviations, decimals, Roman numerals, currency, fractions.

Open work, in order:

1. Acronyms the model's prior wins (GPU, GB, API, CPU, HD, PDF, WIT): v3b = v3 data without `acr_009`/`acr_052`,
   the acronym set repeated 3-4x (or self-distilled judged samples, item 7), same `run_v3.sh` test, phoneme judge
   first. Second judge: gemini-4-carbon, thinking low, 10 clips per request (`train/v3/carbon_judge.py`, 11/11 on the
   user-labelled files, about 150 tokens per clip); gemini-4-hydrogen costs about 1,270 tokens per ask and with
   thinking low matched the user's ears only 9/11. Gemini credit is limited: use it sparingly. If a LoRA wins, merge it (`merge_lora.py`) before serving.
2. Split turns under load: with 5 robots, a visitor's pause mid-request ends the turn when its
   transcript is late, the rest interrupts and restarts the LLM (+1.3 s, 4-12 of 20 turns). Smart
   Turn / user-turn-stop settings, not the voice. Whisper with 5 robots ~380 ms (GPU-bound).
3. Pitch spread between the sentences of one reply is higher than in the training audio (1.57 vs
   1.15 semitones, English 2.43): every sentence restarts from the prompt clip.
4. "seekor" (1-2/8) and "keenam" (0/8) are in no training data yet; record or generate sentences with them.
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
8. A/B the ASR and LLM the user proposed (2026-11-17), same scenarios, keep only what measures better:
   Qwen3-ASR-1.7B (supports id/ms/en, vLLM backend, streaming) vs Whisper turbo on real robot-mic
   recordings (WER, latency with 5 robots, hotword handling); Gemma 4 26B-A4B QAT Q4 (MoE, 4B active)
   vs Gemma 4 12B in llama.cpp (TTFT, tokens/s with 5 slots, tool-call success, VRAM: about
   15-16 GB + KV against 11 GB now, so watch the 32 GB card). vLLM only if a non-GGUF 4-bit build
   exists and it wins at 5 robots; llama.cpp stays otherwise.
9. Server hygiene: on 2026-11-17 the four raisa-voice containers disappeared (not stopped: removed)
   while other sessions were logged into the shared account; they were brought back with
   `docker compose up -d`. If it happens again, check `docker events` and ask the user before
   assuming. Disk was cleaned the same day (old Qwen3-TTS experiments and local LoRA checkpoints,
   all checkpoints are on HF), 121 GB free.
10. Thesis evidence still missing: the same scenarios measured on Gemini Live (latency per
   stage, first audio), a listening test with people (MOS, and A/B against Gemini Live; the pack
   `out/smooth_ab/` is a start), ASR error rate on real robot-microphone recordings, and
   tool-call success rate.
11. Titles: render `train/titles/gen_title_cases.py --tts-out` sentences and judge the long English
   expansions ("Master of Science in Engineering") inside Indonesian sentences, 8 samples each;
   `TITLE_POST_MODE=short` is unmeasured. Grow the lexicon with `generate_title_dataset.mjs`, grounded on
   Permendikbudristek 6/2022 (59/2018 is revoked) and rector decrees.
12. Text rules from 2026-11-19 (ranges, model numbers, English dates, acronym model guesses): render
   `scripts/render_traps.py`-style sentences, 8 samples each. Then A/B `EN_ACRONYM_STYLE=spell` against raw for GPU,
   GB, API, CPU, PDF with the phoneme judge; switch it on only if it wins.
13. Text tagger: run `train/tagger/generate_sentences.mjs` (about 20k sentences, one strong judge + Gemma),
   train XLM-R base and large, keep both `report.md` files (unseen-capital accuracy against the rules, ms per
   sentence), deploy the better one only if it beats the rules on unseen capitals without costing first sound.

Decisions already made (do not reopen without new evidence):

| decision | reason |
| --- | --- |
| llama.cpp, not vLLM | vLLM audio for Gemma 4 was nightly-only; llama.cpp gives 160 tok/s |
| separate Whisper, not audio straight into Gemma | direct audio failed inside tool conversations |
| WebSocket, not WebRTC | fine on the campus LAN; revisit only if jitter is measured |
| no Indonesian phoneme dataset | the model has no such input, 65 min cannot teach one |
| no full fine-tune | needs about 40 GB VRAM and hundreds of hours; LoRA reaches about 98% of it |
