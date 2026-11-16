"""Phoneme recogniser used to judge how an acronym was read (Whisper writes "GPU" for any reading).

wav2vec2 fine-tuned on espeak phonemes, greedy CTC decoding from the vocabulary file, so neither
the phonemizer package nor espeak is needed.
"""
import json

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


class Phones:
    def __init__(self, model: str = MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to(self.device).eval()
        vocab = json.load(open(hf_hub_download(model, "vocab.json"), encoding="utf-8"))
        self.tokens = {i: t for t, i in vocab.items()}
        self.blank = vocab["<pad>"]

    @torch.no_grad()
    def __call__(self, path: str) -> list[str]:
        y, _ = librosa.load(path, sr=16000)
        y = (y - y.mean()) / (y.std() + 1e-7)  # the model's feature extractor normalizes the same way
        ids = self.model(torch.from_numpy(y)[None].to(self.device)).logits[0].argmax(-1).cpu().numpy()
        ids = ids[np.insert(ids[1:] != ids[:-1], 0, True)]
        return [self.tokens[i] for i in ids if i != self.blank and not self.tokens[i].startswith("<")]


if __name__ == "__main__":
    import sys

    phones = Phones()
    for f in sys.argv[1:]:
        print(f.split("/")[-1], "|", " ".join(phones(f)), flush=True)
