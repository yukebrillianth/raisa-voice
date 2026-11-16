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
