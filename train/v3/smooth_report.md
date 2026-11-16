# Making replies sound continuous: measurements (2026-11-16/17)

Every change below was measured before it was kept. Served stack at the start: commit `d112184` (LoRA v2-hybrid step 280,
`raisa-ref.wav` prompt + reference, temperature 0.7, cfg 2.0, 10 steps). Scripts: `scripts/bench_robots.sh`,
`scripts/stage_times.py`, `train/v3/*.py`. Audio for every table is in `out/` on the server (not in git); the listening
pack is `out/smooth_ab/` (index.txt). Acronym baseline: `acro_report.md`.

## Summary

| change | kept | why (numbers below) |
| --- | --- | --- |
| Cut each sentence's leading silence to 50 ms (`tts/server.py`) | yes | every sentence began with 0.36-0.81 s of digital silence; first sound 945-1491 -> 494-608 ms solo, pause between sentences 0.90 -> 0.41 s (training audio 0.41) |
| Hold a long sentence's level (`Leveller`, `tts/server.py`) | yes | the "muffled" voice is a loudness fade: end of a 20-30 s sentence 4.4-5.0 dB under its start (training 1 dB); after 0.8-1.9 dB, spectrum unchanged |
| Split long sentences at commas for the TTS | no | fade only 1 dB less, +0.4 s pause per cut |
| TTS warmup with batches of 1-5 (`tts/server.py`) | yes | first 5-robot round after a restart: TTS first byte 571-588 ms on v2 before, 238-286 ms with it (and that byte is now speech) |
| Academic titles spelled out, no sentence split on their dots | yes | aggregator split "Prof. Dr. Ir." into its own sentence; tests |
| LLM model, key, URL, extra body from env | yes | defaults unchanged; README "Another LLM" |
| One-word English sentence sent with the next one | yes | "Sure." alone 0/8 right ("Sur"), with the next sentence 7/8; Indonesian one-word sentences 8/8 alone, not held |
| Cross-sentence continuation (previous sentence as prompt) | no | joins no better than the trim alone, voice drifts lower (178 Hz vs 188), text repeated or skipped |
| Lexicon: dicolokkan, colokan -> "ch"; seenaknya -> "se-enaknya" | yes | 0/8 -> 8/8, 5/8 -> 7/8, 5/8 -> 7/8 |
| se-/ke- + e rule ("se-ekor") | no | seekor 3/8 in all three spellings, keenam 1-3/8; needs data |
| 5 Whisper workers instead of 2 | no | Whisper per request unchanged under 5 robots (~400 ms): the GPU is the limit |

## 1. Five robots at once: v1 vs v2

Same clip (`q16.wav`, two phrases with a pause), 5 robots started together, 3 measured rounds x 2 repeats per LoRA after
a warmup round. Stages from the orchestrator log (`stage_times.py`): asr = Whisper request, llm = request to first token,
sent = first token to first complete sentence, tts = TTS first chunk, lead = silence at the start of that chunk.

| LoRA | turns | first byte median (min-max) ms | asr | llm | sent | tts | lead | split turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (`raisa-lora`) | 30 | 1176 (818-2004) | 381 | 354 | 123 | 100 | 657 | 8/30 |
| v2 (`raisa-lora-v2`) | 30 | 1139 (870-2060) | 379 | 432 | 90 | 78 | 670 | 6/30 |
| v2, one robot | 4 | 475 (464-487) | 122 | 138 | 59 | 49 | 639 | 0/4 |
| v2 after this work, 5 robots | 20 | first sound 1226 (1097-2618) | 370 | 404 | 104 | 216 | 49 | 4/20 |

- v2 is not slower than v1: same first byte, same stages. The older "328-671 ms" figure was another clip and setup.
- Where the time goes with 5 robots: Whisper ~120 -> ~380 ms (2 workers, GPU-bound: 5 workers did not lower it),
  first LLM token ~130 -> ~400 ms (5 prompts prefilled together), and **split turns**: the clip's first phrase ends the
  turn when its transcript arrives late, the second phrase then interrupts and restarts the LLM (+1.3 s). 4-12 of 20
  turns per run, timing-dependent. That is Smart Turn / turn-stop behaviour, not the voice.
- Until now the first byte was silence: the robot heard its first word ~0.66 s after "first byte" (lead column). First
  sound for 5 robots before = first byte + lead = about 1.8 s median; after the trim 1.23 s.
- TTS first chunk is higher now (78-100 -> ~216 ms with 5 robots) because the chunk is speech, not silence.

## 2. Academic titles

`speech_text.py`: TITLES_BEFORE / TITLES_AFTER, longest match first, "Dr." Doktor vs "dr." Dokter, a degree that also
ends the sentence keeps one dot. The sentence splitter (Pipecat SimpleTextAggregator + sentencex) split
"Prof. Dr. Ir. Budi, S.T., M.Sc." into "Prof. Dr. Ir." | "Budi Santoso, S.T., M.Sc. akan hadir besok." and "S.Tr." | "T., M.Kom.";
`bot.TitleDots` protects title dots (and a dot glued to the next letter while it streams in). Screen text unchanged; word
timing weighs "Prof." as "Profesor" with no stop pause. Tests: 8 normalizer cases, 3 aggregator cases, 1 timing case.

## 3. LLM by env

`LLM_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_EXTRA_BODY` (JSON; default turns thinking off). Defaults = the old hardcoded
values; compose passes them from `.env`. Not tested against an external API.

## 4. Lead silence and joins (continuity)

14 replies (10 Indonesian, 4 English; 2-4 sentences) x 3 samples. fixed / whole / cont / conttrim / refcont were rendered
in one nano-vllm pool before the trim (`smooth_gen.py`): fixed = as served then; whole = the whole reply in one request
(cannot stream, reference only); cont = sentence k>1 prompted with the previous sentence's audio + text (reference latents
kept); conttrim = same with the previous audio cut at its last voiced sample; refcont = prompt = reference clip + previous
sentence. served+trim = through the live server after the lead-silence trim (`served_ab.py`). Training audio = the
voice's own 2-4-sentence training recordings (48). CER = Whisper turbo vs the text.

| variant | joins | gap s | pitch step | size | level step | pitch sd | level sd | tempo sd | drift st | f0 Hz | onset / >1.3x | lead s | CER | first chunk ms: k=1 / k>1 | encode ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training audio (2-4 sentences) | 48 | 0.41 (0.34-0.48) | +2.0 | 5.1 | 2.3 | 1.15 | 0.4 | nan | -2.3 (4.8) | 197 | 1.09 / 26 of 95 | 0.02 | | | |
| fixed | 72 | 0.90 (0.72-1.22) | +3.4 | 3.5 | 3.8 | 1.31 | 0.4 | 0.40 | +0.7 (5.7) | 188 | 1.06 / 22 of 112 | 0.49 | 0.035 | 81 / 76 | nan |
| whole | 72 | 0.62 (0.44-0.87) | +2.0 | 3.8 | 3.0 | 1.44 | 0.6 | nan | -0.8 (5.5) | 181 | 1.05 / 18 of 112 | 0.56 | 0.055 | 81 / nan | nan |
| cont | 72 | 0.70 (0.51-0.98) | +3.0 | 4.1 | 2.4 | 0.84 | 0.8 | 0.51 | -0.7 (4.1) | 178 | 1.03 / 18 of 113 | 0.57 | 0.054 | 79 / 77 | 37 |
| conttrim | 72 | 0.37 (0.00-1.15) | +1.8 | 3.0 | 2.0 | 0.77 | 0.7 | 0.48 | -0.8 (3.5) | 172 | 0.97 / 16 of 106 | 0.48 | 0.042 | 77 / 73 | 37 |
| refcont | 72 | 0.68 (0.52-0.93) | +3.4 | 3.9 | 4.1 | 1.07 | 0.4 | 0.41 | -0.2 (4.2) | 185 | 1.05 / 21 of 113 | 0.52 | 0.040 | 73 / 77 | 37 |
| served+trim | 72 | 0.41 (0.35-0.49) | +4.0 | 4.5 | 4.7 | 1.57 | 0.4 | 0.37 | +0.2 (6.4) | 186 | 1.08 / 29 of 114 | 0.04 | 0.030 | 128 / 134 | nan |

First chunk with nothing else on the GPU (6 pairs): fixed prompt 142 ms, continuation 43 ms + encoding the previous sentence 8 ms (nano-vllm in-process, not through HTTP).
Served TTS first byte after the trim (HTTP, one at a time, 28 sentences): median 107 ms (86-135); that byte is now speech, before it was about 0.5 s of silence.

| variant | lang | gap s | pitch step size | pitch sd | drift st | CER |
|---|---|---|---|---|---|---|
| fixed | id | 0.87 | 3.3 | 1.06 | +0.1 | 0.050 |
| fixed | en | 0.97 | 5.6 | 1.89 | +1.8 | 0.000 |
| whole | id | 0.64 | 4.1 | 1.44 | -0.8 | 0.077 |
| whole | en | 0.56 | 3.3 | 1.44 | -1.5 | 0.000 |
| cont | id | 0.70 | 3.9 | 0.76 | -0.6 | 0.074 |
| cont | en | 0.73 | 4.4 | 1.50 | -1.4 | 0.003 |
| conttrim | id | 0.28 | 2.7 | 0.67 | -0.6 | 0.059 |
| conttrim | en | 0.88 | 4.2 | 1.28 | -1.8 | 0.000 |
| refcont | id | 0.68 | 4.0 | 0.96 | -0.5 | 0.053 |
| refcont | en | 0.71 | 3.6 | 1.16 | +0.2 | 0.007 |
| served+trim | id | 0.40 | 4.3 | 1.31 | +0.2 | 0.042 |
| served+trim | en | 0.47 | 4.9 | 2.43 | +0.5 | 0.000 |

- The pause between sentences was 0.90 s because every sentence starts with ~0.5 s of silence (0.36-0.81 s). Cut to
  50 ms, the pause is the sentence's own trailing silence: 0.41 s (0.35-0.49), the same as the training audio.
- Continuation does not beat that. cont lowers the pitch (f0 178 Hz vs 188 served, 197 training) and failed in 2 of 42
  replies (sentence repeated, "Silakan naik lift ..." twice; a sentence skipped). conttrim runs sentences together
  (gap 0.00 s at the 10th percentile) and drops f0 to 172 Hz. refcont keeps f0 but its joins are no better
  (pitch step 3.9 vs 3.5, level step 4.1 vs 3.8). Added latency would have been small (encode 37 ms under load, 8 ms alone).
  Decision: off. Not tested in the live pipeline (5 robots, barge-in) because it was rejected first.
- Pitch spread inside an answer is higher served than in the training audio (1.57 vs 1.15 semitones), mostly English
  (2.43): every sentence restarts from the prompt. Left as is; continuation was the candidate fix and failed.

## 5. Very short first sentences

5 cases x 8 samples. separate = two requests played back to back, merged = one request. Rows without "+trim" are from
before the trim (nano-vllm pool), "+trim" through the live server after it.

separate = two requests as served, merged = one request. Heard = Whisper has the short word; total = seconds of audio;
gap = silence between the short word and the next sentence; pitch/level step across it.

| case | variant | heard | total s | lead s | gap s | pitch step | level step |
|---|---|---|---|---|---|---|---|
| Baik. | separate | 8/8 | 4.00 | 0.45 | 0.84 | +0.8 | 0.5 |
| Baik. | merged | 8/8 | 3.84 | 0.53 | 0.73 | +2.7 | 1.0 |
| Baik. | alone | 7/8 | 1.28 | 0.51 | | | |
| Baik. | separate+trim | 8/8 | 3.26 | 0.04 | 0.54 | +1.2 | 0.4 |
| Baik. | merged+trim | 8/8 | 3.48 | 0.04 | 0.76 | +0.7 | 1.1 |
| Oke. | separate | 8/8 | 4.40 | 0.64 | 0.81 | -6.8 | 1.7 |
| Oke. | merged | 8/8 | 4.00 | 0.59 | 0.54 | -2.4 | 1.9 |
| Oke. | alone | 2/8 | 1.44 | 0.56 | | | |
| Oke. | separate+trim | 8/8 | 3.39 | 0.04 | 0.47 | +3.2 | 3.8 |
| Oke. | merged+trim | 8/8 | 3.35 | 0.04 | 0.63 | +0.2 | 1.5 |
| Siap! | separate | 8/8 | 4.24 | 0.48 | 0.92 | -9.5 | 0.6 |
| Siap! | merged | 8/8 | 4.24 | 0.52 | 0.91 | -7.9 | 1.5 |
| Siap! | alone | 6/8 | 1.04 | 0.41 | | | |
| Siap! | separate+trim | 8/8 | 3.31 | 0.11 | 0.44 | -7.5 | 0.5 |
| Siap! | merged+trim | 8/8 | 3.64 | 0.10 | 0.73 | -7.5 | 1.0 |
| Tentu. | separate | 8/8 | 4.56 | 0.49 | 1.15 | -0.4 | 1.5 |
| Tentu. | merged | 8/8 | 4.00 | 0.53 | 0.58 | +0.7 | 2.3 |
| Tentu. | alone | 8/8 | 1.28 | 0.51 | | | |
| Tentu. | separate+trim | 8/8 | 3.25 | 0.04 | 0.40 | +0.1 | 2.9 |
| Tentu. | merged+trim | 8/8 | 3.62 | 0.04 | 0.79 | -3.4 | 1.7 |
| Sure. | separate | 4/8 | 4.00 | 0.54 | 1.09 | -4.2 | 1.5 |
| Sure. | merged | 8/8 | 3.44 | 0.45 | 0.56 | -3.2 | 0.9 |
| Sure. | alone | 4/8 | 1.36 | 0.56 | | | |
| Sure. | separate+trim | 0/8 | 2.78 | 0.06 | 0.44 | -5.7 | 1.2 |
| Sure. | merged+trim | 7/8 | 3.04 | 0.07 | 0.63 | -1.4 | 1.1 |

- Indonesian one-word sentences are right alone (8/8) and, after the trim, separate gives the shorter, natural gap
  (0.40-0.54 s vs 0.63-0.79 merged). Not merged.
- "Sure." alone is said "Sur"/"Sore" (0/8 separate after the trim, 4/8 before); merged 7-8/8. A single English word has
  no language context. `bot.SentenceAggregator` holds a one-word English sentence (`language()` = en; "sure okay ok sorry
  certainly alright" added to EN_WORDS) and sends it with the next sentence; at the end of a reply it goes alone, so
  nothing is delayed when nothing follows. Cost when something follows: the LLM time of the next sentence (first
  sentence 59 ms median alone, 75-180 ms with 5 robots, `sent` column). Not measured end to end (the LLM would have to
  answer in English starting with one word).

## 6. "Mendem": loudness over a long sentence

`muffle.py`. One sentence of ~3-30 s, Indonesian + English, 3 samples, through the live server; then the same closing
clause after 0, 1, 3 or 5 clauses (6 samples) so the words measured are identical; high band = energy above 4 kHz as a
share of all (dB), voiced frames separately; fade = level of the last 2.5 s minus the first 2.5 s.

Fade of whole sentences (last 2.5 s minus first 2.5 s, median):

| sentence | training recordings | served before (trim only) | served now (Leveller) |
| --- | --- | --- | --- |
| 4-8 s | -1.1 dB (577 clips) | -1.7 dB | -1.2 dB |
| 8-11 s | -0.6 dB (22 clips) | -1.7 dB | -2.0 dB (worst -4.6) |
| ~16 s | none longer than 10.5 s | -4.4 dB (worst -5.8) | -1.9 dB (worst -3.1) |
| ~21 s | none | -5.0 dB (worst -6.4) | -0.8 dB (worst -2.6) |

Same closing clause, early vs late (served = one request, pieces = split at commas, levelled = Leveller applied offline
to the same one-request clips with the server's chunk size, `level_sim.py`):

| lang | clauses before | sent as | sentence s | high band dB | voiced high band dB | centroid Hz | level dB | fade dB | change vs alone: high band / voiced / centroid / level | pause at cut s |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 0 | one request | 2.3 | -20.1 | -24.4 | 763 | 35.5 | +0.0 | +0.0 / +0.0 / +0% / +0.0 |  |
| en | 0 | one request, levelled | 2.4 | -20.1 | -24.4 | 763 | 35.6 | +0.0 | +0.0 / +0.0 / +0% / +0.0 |  |
| en | 1 | one request | 7.3 | -20.8 | -24.6 | 824 | 34.2 | -2.9 | -0.7 / -0.2 / +8% / -1.3 |  |
| en | 1 | one request, levelled | 7.3 | -20.5 | -24.7 | 813 | 36.3 | -1.3 | -0.4 / -0.3 / +7% / +0.7 |  |
| en | 3 | one request | 13.3 | -21.5 | -24.6 | 788 | 33.7 | -3.6 | -1.4 / -0.2 / +3% / -1.8 |  |
| en | 3 | one request, levelled | 13.3 | -21.2 | -24.5 | 782 | 37.1 | -0.9 | -1.1 / -0.1 / +3% / +1.6 |  |
| en | 3 | pieces | 13.8 | -19.8 | -24.0 | 803 | 34.2 | -2.7 | +0.3 / +0.4 / +5% / -1.3 | 0.45 (0.34-0.80) |
| en | 5 | one request | 20.7 | -20.2 | -23.5 | 932 | 33.0 | -4.0 | -0.2 / +0.9 / +22% / -2.5 |  |
| en | 5 | one request, levelled | 20.7 | -20.0 | -23.6 | 923 | 37.1 | -0.6 | +0.1 / +0.8 / +21% / +1.5 |  |
| en | 5 | pieces | 23.4 | -20.0 | -24.3 | 766 | 34.2 | -3.1 | +0.1 / +0.1 / +0% / -1.2 | 0.38 (0.29-0.46) |
| id | 0 | one request | 2.8 | -25.0 | -26.5 | 527 | 35.6 | -0.9 | +0.0 / +0.0 / +0% / +0.0 |  |
| id | 0 | one request, levelled | 2.8 | -24.8 | -26.4 | 523 | 35.8 | -0.8 | +0.0 / +0.0 / +0% / +0.0 |  |
| id | 1 | one request | 7.8 | -24.9 | -26.1 | 598 | 34.9 | -2.6 | +0.1 / +0.4 / +13% / -0.7 |  |
| id | 1 | one request, levelled | 7.8 | -24.8 | -26.1 | 590 | 36.9 | -0.8 | +0.0 / +0.3 / +13% / +1.1 |  |
| id | 3 | one request | 15.3 | -25.4 | -26.8 | 608 | 34.3 | -3.6 | -0.4 / -0.3 / +15% / -1.3 |  |
| id | 3 | one request, levelled | 15.3 | -25.5 | -26.7 | 604 | 37.8 | -0.3 | -0.6 / -0.3 / +15% / +2.0 |  |
| id | 3 | pieces | 15.0 | -23.9 | -26.0 | 593 | 34.3 | -3.3 | +1.1 / +0.4 / +12% / -1.4 | 0.43 (0.36-0.48) |
| id | 5 | one request | 22.2 | -24.7 | -26.4 | 740 | 33.0 | -4.6 | +0.3 / +0.0 / +40% / -2.6 |  |
| id | 5 | one request, levelled | 22.2 | -24.5 | -26.4 | 739 | 37.4 | +0.0 | +0.4 / -0.0 / +41% / +1.6 |  |
| id | 5 | pieces | 25.3 | -24.6 | -25.7 | 568 | 33.7 | -3.4 | +0.4 / +0.7 / +8% / -1.9 | 0.41 (0.31-0.46) |

- The spectrum does not change late in a sentence: voiced high band within ±0.7 dB, centroid even rises. What changes is
  loudness: a 20 s sentence ends 4-5 dB quieter (training recordings: 1 dB). Heard as a duller, further-away voice.
  The training clips are all 10.5 s or shorter, so long sentences are outside what the LoRA saw.
- Splitting at commas (150 characters) gains about 1 dB and adds a 0.4 s pause at each cut (the sentence gets ~3 s longer).
  Rejected.
- `Leveller`: target = level of the first 1.5 s of speech; a 1.5 s average of the level sets a gain for whatever falls
  more than 1.5 dB under it (max +6 dB), ramped within each 0.16 s chunk, only on speech chunks. Fade after: -0.3 to -1.3 dB
  offline, -0.8 to -2.0 dB live (other samples). Short sentences unchanged.

## 7. Lexicon

colok-words (phoneme judge: "col" vs "kol"/"tol", 8 samples; Whisper spelling for reference):

c = the phoneme recogniser hears "col" (tʃ), not "kol"/"tol"; Whisper = its spelling of the word.

| word | spelling | c heard | heard as | Whisper wrote |
|---|---|---|---|---|
| colok | plain | 8/8 | colo x8 | colok x8 |
| colok | ch | 7/8 | colo x7, kolo x1 | colok x7, kolok x1 |
| colokan | plain | 5/8 | colokan x5, kolokan x3 | colokan x5, kolokan x3 |
| colokan | ch | 7/8 | colokan x6, coloka x1, kolokan x1 | colokan x7, kolokan x1 |
| dicolokkan | plain | 0/8 | dikolokan x6, ditolokan x2 | ? x6, ditolokkan x2 |
| dicolokkan | ch | 8/8 | dicolokan x7, dicoloqan x1 | dicolokkan x8 |
| mencolok | plain | 8/8 | mencolok x7, mencolokh x1 | mencolok x8 |
| mencolok | ch | 8/8 | mencolok x4, mancolok x4 | mencolok x8 |

se-/ke- + e words (Gemini judge, gemini-4-hydrogen, 3 asks, majority; right = both e's heard; 8 samples; 273 calls):

| word | plain | "se-ekor" | "se ekor" | heard when wrong |
| --- | --- | --- | --- | --- |
| seekor | 3/8 | 3/8 | 3/8 | se-kor |
| seenaknya | 5/8 | 7/8 | 7/8 | se-nak-nya |
| keenam | 1/8 | 2/8 | 3/8 | ke-nam |
| keempat | 8/8 | 8/8 | 8/8 | |
| keesokan | 8/8 | 8/8 | 8/8 | |

Kept: dicolokkan, colokan -> "ch" (plain wrong, "ch" 7-8/8); seenaknya -> "se-enaknya". No generic rule: the spelling does
not move seekor or keenam. Needs data: seekor, keenam.

## 8. Latency and barge-in, before vs after

| measure | before (d112184) | after |
| --- | --- | --- |
| one robot, `q16.wav`: first byte | 464-510 ms | 494-587 ms |
| one robot, `q16.wav`: first sound | 945-1491 ms | 494-608 ms |
| one robot, `gemini16.wav`: first byte / first sound | 131-155 / 395-451 ms | 212-235 / 235-239 ms |
| TTS first byte, alone (28 sentences) | 45-65 ms (silence) | 112 ms median, 66-189 (speech) |
| barge-in: audio stops after the visitor starts | 327 ms, 0 stale frames, 0 late words | 327-330 ms, 0, 0 |
| first audio of the reply after a barge-in | 377 ms | 418-462 ms |
| 5 robots, first sound | ~1.8 s median (first byte 1139 + lead 670) | 1226 ms median (1097-2618) |
| tests | 72 | 83 |

## Not tested

- Continuation and the one-word merge end to end with 5 robots and barge-in (continuation was rejected offline; the merge
  is covered by unit tests only).
- The Leveller with real robot speakers; levels were measured on files, not on the robot.
- An external LLM through the new env variables.
- A listening test with people: the pack in `out/smooth_ab/` is for the user's ear.
