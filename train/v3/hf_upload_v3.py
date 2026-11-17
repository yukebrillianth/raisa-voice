"""Archive LoRA v3 to the private HF model repo under v3/ (never touches the root = v1 or v2/).

    HF_TOKEN=$(cat ~/yuke/.hf_token) python3 hf_upload_v3.py <served adapter dir> [<merged dir>]    # server, host python
Uploads: v3/ adapter (lora_weights + lora_config of the chosen checkpoint), v3/checkpoints/ (every checkpoint, logs,
train.log), v3/train/ (scripts, config, data reports, test sets), v3/reports, and v3/merged/ only when given.
"""
import sys

from huggingface_hub import HfApi

REPO = "yukebrillianth/raisa-voxcpm2-lora"
T = "/home/teaching-factory/yuke/raisa-voice-train/"
api = HfApi()
assert api.model_info(REPO).private, "the repo must stay private"
up = lambda folder, path, **kw: api.upload_folder(repo_id=REPO, folder_path=folder, path_in_repo=path,
                                                  commit_message=f"v3: {path}", **kw)
up(sys.argv[1], "v3", allow_patterns=["lora_weights.safetensors", "lora_config.json"])
up(T + "ckpt_v3", "v3/checkpoints")
up(T + "v3", "v3/train", ignore_patterns=["__pycache__/*", "*.pyc"])
up(T + "data_v3", "v3/train/data", allow_patterns=["*.json", "*.jsonl", "*.md"])
up(T + "eval_v3", "v3/eval", allow_patterns=["*.json", "*.txt", "*.md"])
if len(sys.argv) > 2:
    up(sys.argv[2], "v3/merged", allow_patterns=["model.safetensors", "merge_report.json", "config.json"])
files = api.list_repo_files(REPO)
print(len([f for f in files if f.startswith("v3/")]), "files under v3/; private:", api.model_info(REPO).private)
