Real holdout recordings: f0 median 188 Hz, onset median 1.22, >1.3x 10/30, fade median -1.2 dB

Holdout 30 x 3 samples (CER/WER vs the original text, Whisper large-v3-turbo); unseen = 15 new sentences x 3; long = the 12/20/30 s sentences of served_ab.LENGTH, id + en, x 3. fade = level of the last 2.5 s of speech minus the first 2.5 s (dB, median), with the served Leveller. score = CER + unseen CER + 0.5*|tempo-1| + 0.5*share(onset>1.3).

| model | CER | WER | tempo | onset | onset >1.3x | f0 | fade | unseen CER | unseen WER | long CER | long fade | long s | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2h | 0.043 | 0.071 | 0.91±0.09 | 1.13 | 21/90 | 182 | -1.0 | 0.026 | 0.055 | 0.004 | -1.8 | 16.3 | 0.232 |
| v2m | 0.045 | 0.074 | 0.90±0.09 | 1.09 | 15/90 | 181 | -1.1 | 0.025 | 0.053 | 0.004 | -1.7 | 16.5 | 0.202 |

Candidates for the acronym test: 
