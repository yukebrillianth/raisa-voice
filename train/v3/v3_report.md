# LoRA v3 and the LoRA merge (2026-11-17)

Result: **v3 was not deployed** (acronym gain within run-to-run noise, the worst acronyms unchanged). **LoRA v2 is now
served folded into the base weights** (`voices/raisa-v2-merged`): the same voice within noise, TTS first byte
104 -> 63 ms, RTF 0.136 -> 0.085. **Indonesian-rule acronyms are now sent as Indonesian letter names with é and
hyphens** ("wé-i-bé", "ha-dé-ém-i"; one generic rule in `speech_text.py`, section 6b). Raw tables: `prep_v3_report.md`, `hold_v3_report.md`, `acro_v3_report.md`,
`merge_hold_report.md`, `merge_acro_report.md`; raw numbers and audio: server `~/yuke/raisa-voice-train/eval_v3/`,
HF `v3/eval/`, laptop `out/eval_v3/`. All data is synthetic Gemini Live speech (research only).

## 1. Data (`v3_prep.py`, rules fixed before looking at the results)

| part | clips in | dropped | why | trained | minutes |
|---|---|---|---|---|---|
| old 600 | 600 | 18 (2 of them in the v2 holdout) + 30 holdout | audit majority contradicts the rules: QRIS letter by letter 3, GPU "ge pe u"/"gi pi u" 3, WIB "wip"/"wib"/"wi i be" 3, MAGE "meij" 2, PDF "pe de ef", NFC "en ef ci", WIT "wip", SKEM "es ka em", GB "gi ga bait", IRIS "i ris", QA "kyu ei" in an Indonesian sentence | 554 (v2 hybrid transcripts) | 60.5 |
| new general 280 + 30 English | 310 | 7 | Whisper: "Siap!" heard "See ya!" (CER 1.0), v3_074 cut after its first sentence (generator log: socket closed). Gemini: IRIS "i ris" 2, QRIS "ki ar ai es", QR "ki ar", FTEIC "...i ka" | 303 | 53.3 |
| acronym set | 258 | 7 not verified + 19 judge holdout | `.verify.json` ok only; holdout = 1 Indonesian clip per acronym with >= 6 verified clips | 232 | 17.2 |
| **total** | | | | **1089** | **131.0** (2.2 h; 40% with ref_audio) |

Whisper check: large-v3-turbo, beam 5, transcribed as id and as en, the one closer to the script kept (1-s clips are
detected as English), Whisper's digits/titles put through `v3_text.v3()`; median CER 0.000, 90th percentile 0.020.
Transcripts: `v3_text.v3()` = the product normalizer without the overrides (acronyms raw, no lexicon, READ_AS_WORD in
Title Case: "MAGE" -> "Mage").

**Found after training**: 2 trained acronym clips contain the generator's pronunciation note read aloud after the
sentence (`acr_009` TV, `acr_052` ID: "... Catatan pelafalan, jangan dibacakan, ID diucapkan ID"); the judge-holdout
clip `acr_037` too (only its text is used). Gemini verification passed them because it only checks the acronym. Whisper
on all 251 verified clips found no others. Drop them before any retrain.

## 2. Training

Official config (`lora_v3.yaml`: LR 1e-4, r 32 / alpha 32 on LM + DiT, batch 2 x 8, warmup 100), 1021 steps = 15 epochs
of 68 steps, checkpoint + validation every epoch. **1018 s (17 min)**, 24.6 GB VRAM, 0 clips filtered by
max_batch_tokens (longest 28.6 s).

| step | 0 | 68 | 136 | 204 | 272 | 340 | 408 | 476 | 544 | 612 | 680 | 748 | 816 | 884 | 952 | 1020 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val loss total | 1.071 | 0.920 | **0.797** | 0.822 | 0.813 | 0.807 | 0.827 | 0.826 | 0.841 | 0.835 | 0.842 | 0.848 | 0.850 | 0.856 | 0.857 | 0.850 |
| val stop loss | 0.290 | 0.141 | 0.043 | 0.042 | 0.044 | 0.051 | 0.071 | 0.071 | 0.086 | 0.083 | 0.090 | 0.095 | 0.097 | 0.097 | 0.098 | 0.098 |

## 3. Holdout (v2 holdout 30 + 15 unseen + 10 long sentences, x 3, served synth with Leveller, product text per model)

| model | CER | WER | tempo | onset | >1.3x | f0 | fade dB | unseen CER | unseen WER | long fade dB | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real recordings | | | 1.00 | 1.22 | 10/30 | 188 | -1.2 | | | | |
| v2 hybrid s280 (served text) | 0.043 | 0.071 | 0.91 | 1.13 | 21/90 | 182 | -1.0 | 0.026 | 0.055 | -1.8 | 0.232 |
| v3 s0408 (cand.) | 0.047 | 0.077 | 0.92 | 1.10 | 16/90 | 176 | -0.9 | 0.033 | 0.076 | -1.0 | **0.209** |
| v3 s0612 (cand.) | 0.047 | 0.077 | 0.91 | 1.15 | 19/90 | 181 | -0.8 | 0.030 | 0.069 | -0.7 | 0.228 |
| v3 s0884 (cand.) | 0.042 | 0.071 | 0.91 | 1.16 | 19/90 | 181 | -0.8 | 0.029 | 0.067 | -0.8 | 0.224 |
| v3, all 15 checkpoints | 0.041-0.047 | 0.068-0.080 | 0.89-0.97 | 1.10-1.21 | 16-36/90 | 172-187 | -0.7..-1.2 | 0.029-0.035 | 0.067-0.076 | -0.2..-1.3 | 0.209-0.313 |

No v3 checkpoint is better or worse than v2 beyond noise on the holdout (CER differences < 0.01). Unseen CER is 0.003-0.009
higher on v3. Long sentences fade less on v3 (-0.7 vs -1.8 dB). Candidates = 3 best scores (s0612 is also the best of the
second half).

## 4. Acronyms (2 new carriers x 4 samples per acronym, 54 acronyms; product text per model)

Primary judge: the phoneme judge P (validated 12/13 against the user's ears). Gemini G (3 asks) covered all 572 v2 clips
and 128 of v3 s0408 before the budget stop (judges agree on 86% of 700 clips, kappa 0.66).

| set | v2 (served) | v3 s0408 | v3 s0612 | v3 s0884 | v2 merged (same weights, second run) |
|---|---|---|---|---|---|
| Indonesian, 432 readings | P 71%, 23/54 under 7/8; G 74% | P 73%, 22/54 | P 74%, 20/54 | P 74%, 20/54 | P 76%, 16/54 |
| the 22 acronyms of acro_report.md | P 60%; G 69% | P 60% | P 64% | P 62% | P 63% |
| English carriers (HDMI ITS QR API) | P 100%; G 100% | P 94% | P 88% | P 81% | P 88% |
| judge holdout (19 acronym clips never trained) | P 47%; G 70% | P 43% | P 43% | P 42% | P 49% |
| same 128 clips judged by G | G 87/128, P 80/128 | G 89/128, P 75/128 | | | |
| Carbon second opinion, 432 Indonesian readings | C 79%, 16/54 under 7/8 (agrees with P 89%) | | C 74%, 19/54 (agrees 88%) | | |

**Noise**: the merged v2 renders (same voice, see section 6) score 5 points above the unmerged v2 renders on the same
432 readings. So v3's +2..3 points are within run-to-run noise, and v3 is worse on the English carriers and the judge
holdout. Decision rule "clearly higher": not met.

Per acronym, unchanged by v3 (P, all four models, 8 samples): GPU 0/8 "gepiu", GB 0/8 "gebe", API 0/8 "apii",
CPU 0-1/8, HD 0/8 "hesydi", FTEIC 0/8 "...isi", PDF 0-2/8 "pedeef", WIT 0-1/8. The verified training clips themselves read
GPU/GB/HD/FTEIC right (P on the training audio: GB 3/3, HD 3/3, FTEIC 3/3, GPU 2/6 "jipiu"), so 3-9 clips per acronym in
1089 do not move the model's prior. Improved on v3: WIB raw 3-6/8 vs 0/8 for v2's "we i be"; ITB, IPK, SPARK, UI small.

## 5. Special words (8 samples; P / G where judged)

| word | v2 served text | v2 | v3 s0408 | v3 s0612 | v3 s0884 |
|---|---|---|---|---|---|
| seekor | seekor | P 2/8, G 2/8 "se-kor" | P 1/8 | P 2/8 | P 1/8 |
| keenam | keenam | P 0/8, G 1/8 "ke-nam" | P 0/8 | P 0/8 | P 0/8 |
| seenaknya raw | se-enaknya | P 6/8, G 7/8 | P 6/8 | P 4/8 | P 2/8 |
| seenaknya as "se-enaknya" | | | P 7/8 | P 8/8 | P 8/8 |
| colokkan raw | cholokkan | P 8/8, G 8/8 | P 8/8 | P 8/8 | P 8/8 |
| MAGE | Mage | P 8/8, G 8/8 | P 8/8 | P 8/8 | P 8/8 |
| IRIS | Iris | P 7/8, G 3/8 "i ris" | P 6/8 | P 6/8 | P 7/8 |
| SKEM | Skem | P 7/8, G 8/8 | P 8/8 | P 8/8 | P 8/8 |
| WIB | we i be | P 0/8, G 0/8 "wi ai bi" | P 6/8 | P 3/8 | P 4/8 |
| QRIS | kyuris | P 5/8, G 8/8 | P 6/8, G 8/8 | P 6/8 | P 4/8 |

No v3 data contains seekor/keenam; they stay wrong. Text rules unchanged (v3 not served).

## 6. LoRA merge (`merge_lora.py`, `merge_test.py`, `run_merge.sh`, `run_mergeq.sh`), on v2

W' = W + (alpha/r) B A for all 192 adapted q/k/v/o layers (LM + DiT), added to the unfused base keys; nano-vllm packs
q/k/v into qkv_proj at load as for the base model. The delta is 0.8% of the weight norm (median); stored in bf16 the
rounding loses 19% of the delta's norm (median, max 64%) as noise. Loaded with `VOICE_BASE=/voices/raisa-v2-merged
VOICE_LORA=none` (no lora_config).

Sample-level equivalence cannot be shown: at temperature 0 nano-vllm generates silence (max |x| < 0.0001; 5/10 run to the
40-s cap, the rest 23-24 s for 3-6 s sentences), and at 0.7 with seed 0 two runs of the same unmerged weights already differ (10 sentences):

| pair, temperature 0.7, seed 0 | mean abs diff | median correlation | same length |
|---|---|---|---|
| unmerged vs unmerged again | 0.036 | 0.77 | 7/10 |
| unmerged vs merged | 0.045 | 0.69 | 6/10 |

So equivalence was measured on quality, same sets and measures as section 3/4 (v2 adapter vs v2 merged):

| measure | v2 adapter | v2 merged |
|---|---|---|
| holdout CER / WER | 0.043 / 0.071 | 0.045 / 0.074 |
| tempo / onset / >1.3x / f0 | 0.91 / 1.13 / 21/90 / 182 | 0.90 / 1.09 / 15/90 / 181 |
| unseen CER / WER | 0.026 / 0.055 | 0.025 / 0.053 |
| fade / long fade | -1.0 / -1.8 dB | -1.1 / -1.7 dB |
| acronyms Indonesian P (432) | 71% | 76% |

Speed, served `tts` (two A/B rounds, `ttfab.txt`, `bench.txt`):

| measure | adapter round 1 / 2 | merged round 1 / 2 |
|---|---|---|
| TTS first byte alone, median (max) | 106 (139) / 102 (183) ms | 61 (85) / 65 (99) ms |
| RTF alone | 0.136 / 0.136 | 0.084 / 0.086 |
| 5 robots, first sound median (n=30) | 1166 ms (864-2387) | 1168 ms (931-2067) |
| 5 robots, TTS stage median | 202 ms | 179 ms |

Decision: same voice within noise and the TTS is measurably faster (-40% first byte, -37% RTF); five-robot first
sound is unchanged (ASR + LLM dominate). **Merged v2 served.** Rollback: `VOICE_BASE=openbmb/VoxCPM2
VOICE_LORA=/voices/raisa-lora-v2 docker compose up -d`.

After deploy (`tests`, `try_client.py` x3 with barge-in, `bench_robots.sh`, `render_traps.py`):

| check | result | baseline |
|---|---|---|
| pytest | 83 passed | green |
| first sound, 1 robot | 552 / 540 / 528 ms | 494-608 |
| barge-in | 328 ms, 0 stale frames, 0 late words (3/3) | ~330 ms, 0, 0 |
| 5 robots first sound | run medians 1223 / 1066 / 2055 ms (run 3: LLM first sentence 1.4 s) | 1226 median |
| render_traps | all rendered to `out/traps/` (server) | listen |

## 6b. Spelling acronyms as "wé-i-bé" (user's finding by ear, `run_spell.sh`, `spell_report.md`)

Indonesian letter names with the taling e marked and joined by "-" (a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és
té u vé wé éks yé zét), one carrier x 8 samples, P = phoneme judge, C = Carbon:

| acronym | raw, merged v2 (served) | é-hyphen, merged v2 | raw, v3 s0612 | é-hyphen, v3 s0612 |
|---|---|---|---|---|
| WIB | P 3/8, C 6/8 ("we i be": P 0/8, C 0/8) | **P 8/8, C 8/8** | P 4/8, C 5/8 | P 8/8, C 8/8 |
| HDMI | P 1/8, C 1/8 | **P 8/8, C 8/8** | P 1/8, C 1/8 | P 8/8, C 8/8 |
| NIK (unseen) | P 5/8, C 5/8 | **P 8/8, C 8/8** | P 8/8, C 6/8 | P 8/8, C 8/8 |
| ITS / PKM | P 8/8, C 8/8 | P 8/8, C 8/8 | P 8/8, C 8/8 | P 8/8, C 8/8 |
| UMKM | P 7/8, C 8/8 | P 8/8, C 8/8 | P 8/8, C 8/8 | P 8/8, C 8/8 |
| KTP | P 8/8, C 8/8 | P 6/8, C 8/8 | P 8/8, C 8/8 | P 7/8, C 8/8 |
| BPJS | P 8/8, C 8/8 | P 7/8, C 8/8 | P 8/8, C 8/8 | P 8/8, C 8/8 |
| SKCK (unseen) | P 7/8, C 8/8 | P 1/8 "eskatseka", C 8/8 | P 7/8, C 8/8 | P 3/8, C 8/8 |
| **total** | P 55/72, C 60/72 | **P 62/72, C 72/72** | P 60/72, C 61/72 | P 66/72, C 72/72 |

Words (P only: Carbon answered 8/8 for every spelling, it writes the text): seekor 3/8 raw, 1/8 "se-ékor" (v3 2/8,
5/8); keenam 0/8 raw, 3/8 "ke-enam", 2/8 "ke-ènam" (v3 1/8, 0/8, 2/8). No spelling fixes them.

Decision: on the served model the é-hyphen form is right more often overall and fixes WIB, HDMI and NIK; v3 does not
read them raw >= 7/8 either. Implemented as one rule in `speech_text.say_capitals`: every acronym of 2-6 capitals in an
Indonesian sentence is spelled this way, except `EN_ACRONYMS` (USB TV VIP NFC GB IT ID QR API GPU CPU SSD AI LED PC PDF,
English letters by the user's rule, sent raw) and `READ_AS_WORD` (now with OK). The WIB/WIT override is gone; QRIS
"kyuris" and the colok/seenaknya lexicon stay. English sentences unchanged. Screen text unchanged. SKCK by P and KTP 6/8
by P are the open doubts (Carbon 8/8 on both). After deploy: 84 tests pass, first sound 553 / 501 / 509 ms, barge-in
328 ms with 0 stale frames, 5 robots median 1209 ms (run 1 slower right after the restart), traps rendered.

## 7. Gemini use and the budget stop

Calls in this task: 72 (acronyms in the new general clips) + 1,481 recorded in `acro/gemini.json` (700 clips) + up to
about 450 unrecorded (answers lost on a 429 crash and on the stop) + 2 pings + 8 for the low-thinking check, about
1,560-2,010. Tokens per call (gemini-4-hydrogen, one audio clip): default thinking prompt 649 (458 text + 191 audio),
thinking 595, output 25 = 1,269; `thinkingLevel: low` (MINIMAL is rejected) prompt 404-649, thinking 167-354, output
23-64 = 706-907. Low thinking on the user-labelled files: **9/11**; it calls `02_wib_final` and `06_hdmi_its_final`
right where the user heard them wrong (it writes the reading it expects). P agrees 12/13.

After the budget stop, by the coordinator's measured setup: gemini-4-carbon, thinking low, 10 clips per request. On the
user-labelled files it agreed **11/11** (2 calls), so it was used as the second opinion: 87 calls for v2 + v3 s0612
(864 clips) and 39 for the spelling test (384 clips). Per clip: prompt 139-145, thinking 0-1, output 10-11 tokens.
Calls since the stop: 8 hydrogen + 2 + 87 + 39 = **136** (cap 150). Carbon is not usable for se-/ke- words.

## 8. Still wrong / next

- GPU, GB, API, CPU, HD, FTEIC, PDF, WIT read wrong 0-2/8 by every model, v3 included. 3-9 clean clips each are not
  enough against the base model's prior. Next experiment (cheap, 17 min): drop the 2 contaminated clips and repeat the
  acronym set 3-4x in the training manifest (or self-distil judged samples), then the same P-judged test.
- seekor 1-3/8, keenam 0-3/8 under every spelling and model: no training data has them.
- é-hyphen: SKCK (P 1/8 vs raw 7/8) and KTP (P 6/8) need an ear check; untested acronyms follow the rule unmeasured.
- The phoneme judge itself moves 5 points between two renders of the same voice: compare models only with gaps above that.
