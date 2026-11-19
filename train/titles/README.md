# Titles: lexicon, generator, build

How the robot reads names with titles, e.g.

```
Prof. Dr. (H.C.) Ir. Bambang Pramujati, S.T., M.Sc.Eng., Ph.D., IPU, ASEAN Eng.
-> Profesor Doktor Honoris Causa Insinyur Bambang Pramujati, Sarjana Teknik, Master of Science in
   Engineering, Doctor of Philosophy, Insinyur Profesional Utama, Asean Engineer.
```

The reader is `orchestrator/titles.py`. It has no trained weights on purpose: a title has one
right reading, and a neural normalizer can invent a wrong one that sounds confident. The
reading comes from `orchestrator/title_lexicon.json` and a small grammar. What gets "trained" is
the lexicon, and the build scores the reader before the lexicon is written.

## Files

| file | what |
| --- | --- |
| `title_seed.json` | hand-checked entries; wins every conflict |
| `generate_title_dataset.mjs` | LLM generator plus blind multi-judge check -> `out/accepted.jsonl`, `out/review.jsonl` |
| `build_title_lexicon.py` | merges seed + accepted, validates, writes the lexicon, scores the reader |
| `gen_title_cases.py` | synthetic test sentences; `--tts-out` also gives spoken sentences for voice data |
| `../../orchestrator/tests/test_titles.py` | fixed cases, run with the other unit tests |

## Grammar in short

- Indonesian degrees are composed: level (`S.`, `S.Tr.`, `M.`, `M.Tr.`, `A.Md.`, `A.Ma.`, `A.P.`)
  times field code (`Kom`, `T`, `KM`, `I.Kom` ...). 46 fields give about 320 degrees from one table,
  so `A.Md.Kep.` or `S.Tr.Kes.` work without anyone listing them. `Sp.` times specialist codes,
  plus `-KGEH` style consultant parts and `(K)`.
- Explicit entries (foreign degrees, professions, pre titles) override composed ones.
  `M.S.` is Magister Sains / Master of Science, not the composed "Magister Sastra".
- Writing variants are one title: `S.T.`, `S. T.`, `ST`, `Ph.D`, `PhD`.
- After a name, the first title needs a Title Case name before the comma, so `ITS, UGM` and
  `seri ST, MT` stay acronyms. `CA`, `MD`, `M.A.` without dots only count after another title
  (`San Jose, CA`).
- `H.` is Haji at a sentence start, after a lowercase word or after `Bapak`/`Ibu`; after a
  capitalised word or an abbreviation it is an initial (`Bambang H. Santoso`, `S. H. Wibowo`).
- Unknown dotted degrees inside a chain are spelled out with letter names and written to
  `TITLE_LOG_PATH`. That log is the best input for the next generator round.

Language follows `speech_text.language()`. Code-switched sentences count as Indonesian: Indonesian
degrees get Indonesian names and English degrees stay English, as people say them. English
sentences use English names (`S.T.` -> Bachelor of Engineering, `Prof.` -> Professor).

`TITLE_POST_MODE` sets the titles after the name: `expand` (default, as served before),
`short` (`és-té`, `pi-eich-di`; not measured on the voice yet) or `drop`.

## Growing the lexicon

```bash
cd train/titles
cp llm.example.json llm.json        # generator + 2 judges from different model families
node generate_title_dataset.mjs generate --config llm.json --per-category 200 \
     --grounding lampiran_permendikbudristek_6_2022.txt
node generate_title_dataset.mjs verify --config llm.json
node generate_title_dataset.mjs export
less out/review.jsonl                # fix by hand what is worth keeping, add it to title_seed.json
python build_title_lexicon.py --accepted out/accepted.jsonl --report out/build
cd ../../orchestrator && python -m pytest -q tests
```

Notes:

- Permenristekdikti No. 59/2018 is revoked; the current rule is Permendikbudristek No. 6/2022.
  Rector decrees (e.g. IPB's "Penetapan Gelar Akademik, Gelar Profesi, dan Gelar Vokasi") list
  degrees per study programme and are the best grounding text.
- A judge sees only the abbreviation, never the generator's answer. A row is accepted only when
  every judge says it is used and all expansions agree (word-set overlap >= 0.8).
- Gemini credit is limited (see CLAUDE.md section 5): a batch of 20 rows per judge request keeps
  verification cheap.

## What the build score means

`build_title_lexicon.py` reports exact-match accuracy on synthetic sentences per kind (dotless,
spaced, sentence end, English, code-switch, modes, consultant, unknown, look-alikes). It checks
that titles are found and read as the lexicon says. It cannot check that the lexicon itself is
right: that is what the judges and `review.jsonl` are for. It also says nothing about how the
voice pronounces the expansion; for that, render `--tts-out` sentences and judge them like the
acronym tests (8 samples per case).

Measured on 2026-11-19 (seed lexicon, 20,264 cases each, seeds 1-3): 100% on every kind,
0.06 ms per sentence on one CPU core.
