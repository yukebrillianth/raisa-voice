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
