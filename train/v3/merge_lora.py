"""Fold a VoxCPM2 LoRA into a copy of the base weights, so nano-vllm can serve it without lora_config.

    python merge_lora.py <base snapshot dir> <lora dir> <out dir>      # tts image, CPU is enough

W' = W + (alpha / r) * B @ A for every adapted linear, as nano-vllm's lora_loader applies it at run time
(scaling = alpha / rank; y = x W^T + scaling * x A^T B^T). The adapter's keys name the unfused layers
(`...self_attn.q_proj.lora_A`); the base checkpoint keeps q/k/v and gate/up unfused too (`...q_proj.weight`),
and nano-vllm packs them into qkv_proj / gate_up_proj at load (packed_modules_mapping), so each delta is added
to the weight of the same name and the packing stays nano-vllm's. The sum is done in float32 and stored in the
base dtype (bf16), like the rest of the checkpoint. Every other file of the snapshot is copied unchanged.
"""
import json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIXES = ("base_model.model.", "model.", "module.")  # lora_loader._strip_known_prefixes


def strip(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if name.startswith(p):
                name, changed = name[len(p):], True
    return name


def deltas(lora_dir: str) -> dict[str, torch.Tensor]:
    cfg = json.load(open(os.path.join(lora_dir, "lora_config.json")))["lora_config"]
    pairs = {}
    with safe_open(os.path.join(lora_dir, "lora_weights.safetensors"), "pt") as f:
        for k in f.keys():
            base, kind = k.rsplit(".", 1)
            assert kind in ("lora_A", "lora_B"), k
            pairs.setdefault(strip(base) + ".weight", {})[kind] = f.get_tensor(k).float()
    rank = {p["lora_A"].shape[0] for p in pairs.values()}
    assert len(rank) == 1 and rank.pop() == cfg["r"], "rank mismatch"
    scale = float(cfg["alpha"]) / cfg["r"]
    return {k: scale * (p["lora_B"] @ p["lora_A"]) for k, p in pairs.items()}


def main(base_dir: str, lora_dir: str, out_dir: str):
    d = deltas(lora_dir)
    os.makedirs(out_dir, exist_ok=True)
    merged, stats = {}, []
    with safe_open(os.path.join(base_dir, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        missing = set(d) - set(f.keys())
        assert not missing, f"adapter layers not in the base checkpoint: {sorted(missing)[:5]}"
        for k in f.keys():
            w = f.get_tensor(k)
            if k in d:
                assert d[k].shape == w.shape, (k, d[k].shape, w.shape)
                new = (w.float() + d[k]).to(w.dtype)
                # how much of the delta survives the bf16 rounding (the merge's only approximation)
                kept = ((new.float() - w.float()) - d[k]).norm() / d[k].norm()
                stats.append((k, float(d[k].norm() / w.float().norm()), float(kept)))
                w = new
            merged[k] = w.contiguous()
    save_file(merged, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    for name in os.listdir(base_dir):
        if name != "model.safetensors":
            shutil.copy(os.path.realpath(os.path.join(base_dir, name)), os.path.join(out_dir, name))
    rel = sorted(s[1] for s in stats)
    err = sorted(s[2] for s in stats)
    report = {"lora": lora_dir, "layers_merged": len(stats), "delta_norm_over_weight_norm_median": rel[len(rel) // 2],
              "bf16_rounding_error_of_delta_median": err[len(err) // 2], "bf16_rounding_error_of_delta_max": err[-1]}
    json.dump(report, open(os.path.join(out_dir, "merge_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    a = torch.zeros(4, 3); b = torch.zeros(5, 4)  # self-check of the delta orientation: [out, r] @ [r, in] = [out, in]
    assert (b @ a).shape == (5, 3)
    main(*sys.argv[1:4])
