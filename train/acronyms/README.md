# Acronyms: data and model

How a capitalised token nobody listed is read: as a word (`Unesa`), spelled in Indonesian
(`u-gé-ém`) or spelled in English (`GPU`). The model is `orchestrator/acronym_clf.py`; its data is
`orchestrator/acronym_data.csv`; the Dockerfile trains it at build time.

Order inside `speech_text.say_capitals` (Indonesian sentences):

1. `READ_AS_WORD`, `EN_ACRONYMS`, `ID_ACRONYMS` and the tenant lexicon (`SPEECH_LEXICON_PATH`).
2. Any token in `acronym_data.csv` (or a verified generated file): its label, looked up.
3. The model's guess, if its confidence is at least `ACRONYM_CLF_CONF` (0.5).
4. Indonesian letter names, the measured rule.

English sentences skip all of this: acronyms stay raw ("You need to send a PDF file").

## Numbers (2026-11-19, 626 curated tokens)

Five-fold cross-validation grouped by the first three letters, so `KEMENKES` and `KEMENDAG` are
never on both sides:

| | accuracy on tokens the model never saw |
| --- | --- |
| model (argmax) | 79.9% (F1: WORD 0.92, SPELL_ID 0.75, SPELL_EN 0.72) |
| old rule: longer than 6 letters is a word, else Indonesian letters | 43.3% |
| always Indonesian letters | 34.7% |

Thresholds above 0.5 only made the total worse: sending an unsure token to Indonesian letters
is wrong as often as the model is. SPELL_ID against SPELL_EN is the hard part: "PNJ" and "GPU"
look alike; which alphabet people use depends on the field, not the letters. More data from the
right fields is what moves this, not a bigger model.

Not measured: how the voice says the result. Raw `GPU`, `API`, `PDF` are known wrong in Indonesian
sentences (0-2/8, CLAUDE.md section 8). `EN_ACRONYM_STYLE=spell` writes them as "ji-pi-yu"; render
and judge it (8 samples each, `train/v3/acro_judge.py`) before turning it on.

## Curated data

`acronym_data.csv` came from the hand lists in the old `train_acronym_classifier.py`, with:

- tokens that had two labels left out (KB, TB, CV, SPI, TDP) and tokens with two common readings
  or that are Indonesian words (IPA, ISI, POS, SPA, CAN, IPS, DP, PB, LT, KM, ETA, CS, PM, PR, IT,
  CI, CD, AD, AL, AU, MA);
- ATM, COD, PPNS, BPPT, UPNVJ, UPNVJT to SPELL_ID; BIN, UNESA, UINSA, IKIP, PENS to WORD;
- HDMI back to SPELL_ID ("ha-dé-ém-i" measured 8/8);
- the user's rules from CLAUDE.md section 5 on top of everything (source `user-rule`).

## Growing it

```bash
cd train/acronyms
cp llm.example.json llm.json         # generator + judges from different model families
node generate_acronym_dataset.mjs generate --config llm.json --target 3000
node generate_acronym_dataset.mjs verify --config llm.json
node generate_acronym_dataset.mjs export          # out/accepted.csv, out/review.csv
cd ../../orchestrator
python acronym_clf.py train --extra ../train/acronyms/out/accepted.csv --report ../train/acronyms/out
```

Curated rows always win; a generated row only adds a token nobody labelled, and disagreements are
printed. To keep generated tokens for good, append the reviewed rows to `acronym_data.csv`.
`ACRONYM_LOG_PATH` logs every token the model was asked about in production: the best list to
label next.
