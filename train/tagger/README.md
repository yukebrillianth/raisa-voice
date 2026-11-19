# Text tagger: language of every word, reading of every capital

`orchestrator/text_tagger.py` runs a fine-tuned multilingual encoder over each sentence before it
goes to the voice and tags every word:

| tag | meaning |
| --- | --- |
| ID, EN, JV | Indonesian, English, Javanese word |
| NAME | person, place, brand: no language of its own |
| WORD, SPELL_ID, SPELL_EN | a capitalised token said as a word, in Indonesian letters, in English letters |
| X | digits and punctuation |

It replaces two weaker guesses that look at less than the sentence:

- the sentence language came from a short list of function words, so "Great choice!" or
  "Specs: 2 GPU, 64 GB RAM." had no language at all;
- an unlisted capital was read from its letters alone (`acronym_clf.py`), so "PNJ" (a polytechnic,
  Indonesian letters) and "GPU" (English letters) look the same to it.

## How speech_text.py uses it (and what it never overrides)

Sentence language: if the sentence has function words of only one language ("the", "is", or
"yang", "ya"), they decide. The tagger decides the rest: no function words, or both kinds.
A weak tagger therefore cannot break the easy cases.

Capitals in Indonesian sentences, first match wins:

1. the user's lists (`READ_AS_WORD`, `EN_ACRONYMS`, `ID_ACRONYMS`, tenant lexicon);
2. the labelled table (`acronym_data.csv` + verified generated rows);
3. the tagger, if its confidence is at least `TEXT_TAGGER_CONF` (0.6);
4. `acronym_clf.py`'s guess from the letters;
5. Indonesian letter names.

English sentences keep acronyms raw, as before.

Without `models/text_tagger` on the host everything behaves exactly as without this change.

## Making the data

```bash
cd train/tagger
cp llm.example.json llm.json            # generator + 2-3 judges from different model families
node generate_sentences.mjs generate --config llm.json --target 20000
node generate_sentences.mjs label --config llm.json
node generate_sentences.mjs export      # out/train.jsonl, out/review.jsonl
```

The generator writes sentences the way RAISA talks: Indonesian (38%), code-switched (27%),
English (20%), Javanese (8%), English with Indonesian words (7%), over ten domains, with capitals
in about 70% of them. Every judge tags every word blind at temperature 0. A word's tag is the
majority; a capital needs every judge to agree; the user's rules (`source=user-rule` rows of
`acronym_data.csv`) override the judges. Sentences that fail go to `review.jsonl`.

Cost: labelling sends about 15 sentences per request. With gemini-4-carbon (thinking low) as a
judge, 20,000 sentences is about 1,350 requests per judge. Gemini credit is limited
(CLAUDE.md section 5); a local Gemma 4 judge costs nothing but is weaker, so pair it with one
strong judge. More sentences help most where the review file is long: that is where the judges,
and so the model, are least sure.

## Training (GPU host)

```bash
pip install torch transformers tokenizers onnx onnxruntime scikit-learn
python train_tagger.py --data out/train.jsonl --out out/xlmr-base
python train_tagger.py --data out/train.jsonl --base FacebookAI/xlm-roberta-large --out out/xlmr-large --lr 1.5e-5
```

XLM-R knows Indonesian, Javanese and English (CC-100). Train base and large, then pick by
`report.md`: accuracy on unseen capitals against the latency line. Every sentence waits for the
tagger before it goes to the TTS, so a few milliseconds matter (first-sound target about 0.7 s).

`report.md` and `report.json` hold, for the same sentences:

- **unseen**: sentences with capitals held out of training entirely (15% of all capital tokens).
  This is the honest number: the robot meets acronyms nobody labelled.
- **random**: new sentences with tokens the model saw.
- three columns: the rules served today, the tagger alone, and the served order above;
- the int8 ONNX size, milliseconds per sentence on one CPU thread, and how many tags the int8
  model keeps equal to the trained one.

Keep every report (`out/<run>/report.md`) with its seed and data size: they are the ablation table.

## Training on a Mac (Apple Silicon, 16 GB)

The script picks the Apple GPU (MPS) by itself; it trains in fp32 there. The ONNX file it exports
runs the same on the x86 server.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers tokenizers onnx onnxruntime scikit-learn
python train_tagger.py --data out/train.jsonl --out out/xlmr-base --batch 16 --accum 2
```

- XLM-R base (278M parameters) fits: weights plus AdamW state are about 4.5 GB. Close other big
  apps; unified memory is shared with macOS.
- XLM-R large (560M) is tight on 16 GB: `--batch 4 --accum 8`, expect hours, and a swap-heavy
  machine. Train large on the server or the GH200 instead.
- Not measured on a Mac yet: write the minutes from `report.md` down with the run.

## Deploying

```bash
# on the server, from ~/yuke/raisa-voice
mkdir -p models && cp -r <run dir> models/text_tagger      # model.onnx, tokenizer.json, labels.json
docker compose build orchestrator && docker compose up -d orchestrator
docker compose exec orchestrator python -m pytest -q tests  # the rule tests run without the tagger
```

Upload the chosen model to the HF model repo (`HF_TOKEN=$(cat ~/yuke/.hf_token)`, never commit it).

## Checking the pipeline without an LLM or a GPU

```bash
python smoke.py --out /tmp/tagger-smoke
python train_tagger.py --data /tmp/tagger-smoke/data.jsonl --base /tmp/tagger-smoke/tiny-base \
    --out /tmp/tagger-smoke/model --epochs 8 --batch 32 --lr 2e-3
TEXT_TAGGER_DIR=/tmp/tagger-smoke/model python -c "import sys; sys.path.insert(0, '../../orchestrator'); \
    import text_tagger; print(text_tagger.analyse('Please bring your KTP to floor 3.'))"
```

Template sentences and a 2-layer model: the numbers mean nothing, it only proves data -> training
-> int8 ONNX -> runtime works. On 2026-11-19 it did: int8 tags 99.9% equal to the trained model,
0.4 ms per sentence for the tiny model.

## Not done

- No real data or real model yet: both need the generator run with real LLMs and a GPU.
- Not listened to: a better tag only helps if the voice says the result right (CLAUDE.md section 6).
- The tag set has no "number language" yet: digits inside an English phrase of an Indonesian
  sentence are still read in Indonesian.
