# Voice LoRA (VoxCPM2)

How `voices/raisa-lora/` was made, on the RTX 5090 with the stack stopped.

Data: `dataset-baru/` (100 clips of the voice actor, 10.4 min, 24 kHz mono, 13 English),
copied to `~/yuke/raisa-voice-data/` and split into `train.jsonl` (90) and `holdout.jsonl`
(10, 2 English; indices in `holdout_idx.json`).

```bash
cd ~/yuke/raisa-voice-train            # VoxCPM repo clone (commit 300e426) + the files here
docker build -t raisa-voice-train .     # the tts image + VoxCPM's training script
docker run -d --name raisa-lora-train --gpus all --ipc=host \
  -v ~/.cache/huggingface:/hf:ro -v ~/yuke/raisa-voice-data:/data:ro \
  -v $PWD/ckpt:/ckpt -v $PWD/lora.yaml:/lora.yaml:ro \
  -e HF_HOME=/hf -e HF_DATASETS_CACHE=/tmp/ds -e HF_HUB_OFFLINE=1 \
  raisa-voice-train python scripts/train_voxcpm_finetune.py --config_path /lora.yaml
```

`lora.yaml` is the official `conf/voxcpm_v2/voxcpm_finetune_lora.yaml` (LR 1e-4, r 32 on LM
and DiT, effective batch 16, 1000 steps, about 178 passes over 90 clips), saving every 100
steps. It took 15.5 min. Validation loss is lowest at step 100 and then creeps up (mostly the
stop loss), so the checkpoint is picked by measured speech, not loss:

1. `dataset_stats.py`: the actor's own f0 (185 Hz) and sentence-onset pitch ratio (1.16).
2. `gen_eval.py`: 15 sentences x 3 samples per checkpoint (with and without reference) and
   two zero-shot baselines, rendered by nano-vllm-voxcpm in the tts image.
3. `ana_eval.py`: Whisper large-v3 CER/WER, tempo vs the real holdout recordings, onset pitch
   ratio (first 0.4 s vs the rest), f0.
4. `ab_render.py`: listening files (`out/lora_ab/`).

Step 700 with `raisa-ref.wav` (dataset clip `utt_003`) as prompt and reference was chosen.

## LoRA v2 (served)

600 clips (65.8 min, `dataset-baru/voxcpm2_lora_600_dataset.jsonl`), same 30-clip holdout for
every run. Scripts in `v2/`, run on the server from `~/yuke/raisa-voice-train/v2/`:

1. `v2_prep.py`: holdout, and two transcript sets: `norm` (everything spelled out) and `hybrid`
   (numbers, Rp, times, units, Title Case names spelled out; acronyms raw).
2. `run_v2.sh`: trains both (701 steps, checkpoint every 70), renders holdout 30 + 15 unseen
   sentences x 3 with `v2_gen.py`, scores them with `v2_ana.py` (Whisper CER/WER, tempo, onset).
   Hybrid step 280 won: holdout CER 0.045, onsets > 1.3x 16/90, unseen WER 0.069
   (v1: 0.044, 22/90, 0.096); by ear hybrid read acronyms best.
3. `run_final.sh cwords`: "colokkan" is read with a k; spelled "cholokkan" (or `{CH OW1}lokkan`)
   it is right 3/3. Other c-words (cocok, cepat, pencet, cicil, cuci, cabai, kaca) were already right.
4. `run_final.sh sweep`: cfg 2.0/2.5/3.0 x timesteps 10/16 on the product text, plus TTFAB/RTF
   through `tts/server.py` (`v2_ttfab.py`). No config was clearly better than the defaults within
   sample noise, and 16 steps cost +11..14 ms TTFAB and 1.5x RTF, so cfg 2.0 / 10 steps stay.

The served text is `orchestrator/speech_text.py`: the hybrid transcripts exactly, plus overrides
picked by ear (QRIS "kyuris", WIB/WIT spelled "we i be"/"we i te", "cholokkan"). Weights,
checkpoints and reports: HF `v2/` (model) and `v600/` (dataset).

## LoRA v3 (trained, not served) and the merged v2 (served)

Scripts in `v3/`, run on the server from `~/yuke/raisa-voice-train/v3/`; full tables in `v3/v3_report.md`.

1. `v3_prep.py asr|gemini|build`: Whisper check of the new clips (id and en, CER > 0.15 or 2+ words added/missing),
   Gemini acronym check of the new clips with an acronym, drop of the old clips whose audited acronym reading breaks the
   user's rules (`acronym_audit_600.json`), verified acronym clips only, holdouts (v2's 30 + 19 acronym clips), manifests
   in `data_v3/` (1089 clips, 2.2 h), the eval split and the acronym test set (`v3_acro.py`: 2 new carriers per acronym).
   Transcripts: `v3_text.v3()` (the product normalizer, acronyms raw, no lexicon).
2. `run_v3.sh`: smoke test, training (`lora_v3.yaml`, 15 epochs x 68 steps, 17 min), every checkpoint through the served
   synth (`v3_gen.py` imports `tts/server.py`) on holdout + unseen + long sentences (`v2_ana.py measure|asr`,
   `v3_ana.py fade|hold`), the 3 best into the acronym test (x 4 samples, words x 8) and the phoneme judge
   (`v3_ana.py phones|acro`). `v3_ana.py gemini` is the optional second judge (laptop).
   v3 did not beat v2 beyond noise; not served.
3. `merge_lora.py <base snapshot> <lora> <out>`: W + (alpha/r) B A into a copy of the base weights (CPU, 7 s).
   `run_merge.sh` (merge_test.py at temperature 0 and 0.7 seed 0, then served A/B/A/B: `v2_ttfab.py` +
   `bench_robots.sh`) and `run_mergeq.sh` (the merged weights through the same holdout and acronym sets).
   Served: `VOICE_BASE=/voices/raisa-v2-merged VOICE_LORA=none` (compose default).
4. `hf_upload_v3.py`: HF `v3/` (adapter of step 612, every checkpoint, scripts, data reports, eval numbers) and
   `v3/merged/` (the served merged v2).
