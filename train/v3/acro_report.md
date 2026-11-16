# Acronym readings: baseline before the v3 training set (2026-11-16/17)

What the served voice does with acronyms on the three existing LoRAs, measured before the verified acronym set
(`dataset-baru/voxcpm2_lora_v3_acronym.jsonl`) is trained. No acronym model or text form was changed from these
numbers (the user's call: the training data itself reads GPU, GB, QRIS, PDF differently from its transcripts, see
`out/acronym_audit_600.json`, so v3 data is the fix). Raw data: `out/acro_matrix/`, `out/acro_spell/` (wavs at 16 kHz,
`heard.json` = recogniser tokens, `gemini.json` = Gemini answers), `out/acro_matrix_t03_aborted/` (temperature 0.3).

## Setup

- Models: v1 (`voices/raisa-lora`, 100 clips), v2-norm step 280, v2-hybrid step 280 (served), all with `raisa-ref.wav`
  as prompt + reference, cfg 2.0, 10 steps, `acro_gen.py`.
- Test set (`acro.py`): 21 acronyms x 2 Indonesian carrier sentences + 4 acronyms x 2 English ones, 8 samples each;
  text "raw" (acronym as written) vs "phon" (the user's reading written out). Temperature 0.7 and 0.5 (0.3 aborted:
  half the samples never stopped, up to 320 s, so it was cut). Second round: other spellings of WIB, HDMI, API.
- Judge P (`phones.py`, `acro.py`, `acro_judge.py`): wav2vec2-xlsr-53-espeak-cv-ft phoneme recogniser, greedy CTC; the
  acronym's phones are cut out by aligning the carrier sentence; right = the rule reading or one vowel-colour slip.
  Validated on the files the user labelled by ear: agrees on 12/13 readings (the one disagreement, WIB in
  `u7_pdf_wib_norm`, the user heard right, P heard "wiibi"); it also called GPU/GB in `u9_gpu_gb_hybrid` wrong
  ("gepiu", "gebi"), which the user had labelled right and Gemini heard wrong 3/3, so that label is being rechecked.
- Judge G (`gemini_judge.py`): gemini-4-hydrogen, temperature 0, 3 asks (third only when the first two differ),
  majority of the heard sound in Indonesian spelling, graded with the same rule; no majority counts as wrong.
  Run on the spelling round only (912 clips, 1,967 calls); the matrix (2,400 clips at 0.7) has P only, stopped by
  the scope change.

## Spelling round (P and G)

Judges agree on 751/912 clips (82%, Cohen's kappa 0.64); P right 479, G right 552, G without a majority 15.

## API (eipiai), spellings, temperature 0.7

| written | hybrid | norm | v1 |
|---|---|---|---|
| A P I | P 0/16 `apei`x11<br>G 0/16 `a pe i`x13 | P 0/16 `apei`x11<br>G 0/16 `a pe i`x16 | P 0/16 `apei`x15<br>G 0/16 `a pe i`x13 |
| API | P 0/16 `apii`x10<br>G 0/16 `a pe i`x7 | P 0/16 `apii`x15<br>G 0/16 `a pi i`x8 | P 0/16 `apii`x9<br>G 1/16 `a pe i`x9 |
| e pi ai | P 15/16 `epie`x1<br>G 13/16 `e pi ai i`x3 | P 12/16 `epia`x2<br>G 15/16 `e pi ei`x1 | P 9/16 `epiei`x6<br>G 10/16 `e pi e i`x4 |
| ei pi ai | P 13/16 `eiai`x1<br>G 16/16 | P 15/16 `epia`x1<br>G 15/16 `ei pi i`x1 | P 16/16<br>G 15/16 `ei pi`x1 |

## WIB (weibe), spellings, temperature 0.7

| written | hybrid | norm | v1 |
|---|---|---|---|
| W I B | P 2/16 `weaibe`x7<br>G 5/16 `we ai bi`x4 | P 1/16 `weaibe`x7<br>G 4/16 `we ai bi`x6 | P 2/16 `weaibe`x11<br>G 7/16 `we ai be`x4 |
| WIB | P 9/16 `aibi`x1<br>G 12/16 `a i bi`x1 | P 8/16 `wip`x2<br>G 6/16 `wib`x3 | P 12/16 `weaibe`x2<br>G 13/16 `wi ai bi`x1 |
| we i be | P 0/16 `wiaibi`x6<br>G 1/16 `we ai bi`x8 | P 3/16 `weaibi`x4 runaway 1<br>G 3/16 `we ai bi`x6 | P 1/16 `weaibi`x5<br>G 3/16 `we ai bi`x8 |
| we ih be | P 6/16 `iibe`x2<br>G 3/16 `wi ih be`x5 | P 8/16 `wiikhbi`x2<br>G 2/16 `wi ih be`x5 | P 7/16 `ueikhbe`x2<br>G 4/16 `we ih be`x7 |
| we-i-be | P 13/16 `iibe`x2<br>G 10/16 `wib`x6 | P 16/16<br>G 12/16 `wib`x4 | P 15/16 `iibe`x1<br>G 13/16 `wib`x2 |
| wei be | P 12/16 `ebe`x2<br>G 16/16 | P 12/16 `be`x2<br>G 15/16 `tidak pasti`x1 | P 12/16 `ebe`x3<br>G 16/16 |
| weibe | P 9/16 `waibø`x2<br>G 16/16 | P 11/16 `iwaibe`x3<br>G 16/16 | P 12/16 `waibø`x2<br>G 16/16 |
| wé i bé | P 15/16 `weipe`x1<br>G 14/16 `wib`x2 | P 13/16 `eibe`x3<br>G 15/16 `wib`x1 | P 15/16 `eibe`x1<br>G 16/16 |

## HDMI (hadeemi), spellings, temperature 0.7

| written | hybrid | norm | v1 |
|---|---|---|---|
| H D M I | P 0/16 `hadeemai`x6<br>G 7/16 `hasy de em i`x2 | P 0/16 `hadeemai`x4<br>G 6/16 `ha de em ai`x3 | P 1/16 `hadeemai`x7<br>G 8/16 `ha de em ai`x2 |
| HDMI | P 0/16 `hasydeemi`x3<br>G 3/16 `hasy de em i`x2 | P 3/16 `hacdeemi`x4<br>G 4/16 `hach de em i`x3 | P 13/16 `hajeemi`x2<br>G 14/16 `ha je em i`x2 |
| ha de em i | P 8/16 `hadeemai`x8<br>G 7/16 `ha de em ai`x9 | P 12/16 `hadeemai`x2<br>G 14/16 `ha de em ai`x2 | P 6/16 `hadeemai`x7<br>G 7/16 `ha de em ai`x9 |
| ha de em ih | P 13/16 `hadeemih`x2<br>G 13/16 `ha de em ih`x2 | P 15/16 `hadeemikh`x1<br>G 15/16 `ha de em ih`x1 | P 10/16 `hadeemikh`x6<br>G 13/16 `ha de em ih`x2 |
| ha de emi | P 16/16<br>G 16/16 | P 16/16<br>G 16/16 | P 16/16<br>G 16/16 |
| ha-de-em-i | P 15/16 `hadeem`x1<br>G 16/16 | P 15/16 `thadeemi`x1<br>G 16/16 | P 15/16 `hdeemi`x1<br>G 16/16 |
| hadeemi | P 2/16 `hadimi`x12<br>G 6/16 `ha di mi`x8 | P 4/16 `hadimi`x12<br>G 5/16 `ha di mi`x8 | P 5/16 `hadimi`x11<br>G 11/16 `ha di mi`x4 |

## Main matrix (P only)

Reading the table: "best form per acronym" is what one lexicon could reach on that model, choosing raw or written-out
per acronym by these numbers. Temperature 0.5 is worse than 0.7 on norm and hybrid (Indonesian 85% -> 69%, 84% -> 63%,
with 69-87 runaway samples), no better on v1 (85% both), so lower temperature was not pursued.

## Indonesian sentences, temperature 0.7

| acronym (rule reading) | hybrid phon | hybrid raw | norm phon | norm raw | v1 phon | v1 raw |
|---|---|---|---|---|---|---|
| USB (yuesbi) | P 16/16 | P 15/16 `tyuesbi`x1 | P 15/16 `uesbi`x1 | P 16/16 | P 16/16 | P 16/16 |
| TV (tifi) | P 16/16 | P 15/16 `if`x1 | P 15/16 `tif`x1 | P 14/16 `if`x1 | P 16/16 | P 16/16 |
| GPU (jipiyu) | P 15/16 `jipyu`x1 | P 0/16 `gepiu`x10 | P 16/16 | P 0/16 `gepiu`x8 | P 16/16 | P 0/16 `gepeu`x8 |
| GB (jibi) | P 15/16 `jbi`x1 | P 0/16 `gebe`x10 | P 16/16 | P 0/16 `gebe`x8 | P 16/16 | P 0/16 `gebe`x12 |
| API (eipiai) | P 13/16 `epia`x3 | P 0/16 `apii`x13 | P 15/16 `epia`x1 | P 0/16 `apii`x13 | P 14/16 `epiei`x1 | P 0/16 `apii`x8 |
| QR (kyuar) | P 14/16 `kyar`x1 | P 16/16 | P 8/16 `kuar`x5 | P 14/16 `kuer`x1 | P 14/16 `kiar`x2 | P 14/16 `kiar`x1 |
| NFC (enefsi) | P 12/16 `endefsi`x2 | P 10/16 `nefsi`x5 | P 14/16 `nefsi`x1 | P 15/16 `enfsi`x1 | P 16/16 | P 14/16 `nefsi`x2 |
| VIP (fiaipi) | P 15/16 `fiipi`x1 | P 12/16 `fiepi`x3 | P 14/16 `fiapi`x1 | P 12/16 `fiepi`x1 | P 16/16 | P 14/16 `fiai`x1 |
| ID (aidi) | P 16/16 | P 15/16 `ai`x1 | P 12/16 `aid`x2 | P 16/16 | P 15/16 `adi`x1 | P 15/16 `aibi`x1 |
| IT (aiti) | P 13/16 `ati`x2 | P 16/16 | P 15/16 `aidi`x1 | P 14/16 `iti`x2 | P 14/16 `ait`x2 | P 15/16 `iti`x1 |
| PDF (pidief) | P 16/16 | P 4/16 `pedeef`x12 | P 16/16 | P 2/16 `pedeef`x14 | P 15/16 `pibief`x1 | P 0/16 `pedeef`x16 |
| QRIS (kyuris) | P 15/16 `kuris`x1 | P 9/16 `kuris`x7 | P 15/16 `kuris`x1 | P 11/16 `kuris`x5 | P 13/16 `kuris`x3 | P 12/16 `kuris`x2 |
| WIB (weibe) | P 1/16 `wiaibi`x4 | P 6/16 `uibe`x1 | P 3/16 `weaibi`x3 | P 5/16 `wip`x2 | P 0/16 `wiaibi`x3 | P 8/16 `weaibe`x2 |
| ITS (itees) | P 16/16 | P 16/16 | P 16/16 | P 16/16 | P 16/16 | P 16/16 |
| HDMI (hadeemi) | P 7/16 `hadeemai`x9 | P 0/16 `hacdeemi`x2 | P 13/16 `hadeemai`x3 | P 1/16 `hasydeemi`x3 | P 8/16 `hadeemai`x7 | P 14/16 `hdeemi`x1 |
| KTP (katepe) | P 16/16 | P 15/16 `katee`x1 | P 14/16 `katpe`x1 | P 14/16 `katee`x1 | P 14/16 `katep`x1 | P 16/16 |
| UMKM (uemkaem) | P 16/16 | P 15/16 `emkaem`x1 | P 16/16 | P 15/16 `umkaem`x1 | P 16/16 | P 16/16 |
| BPJS (bepejees) | P 16/16 | P 14/16 `bepjees`x2 | P 16/16 | P 15/16 `bepjees`x1 | P 16/16 | P 13/16 `bejees`x1 |
| WITA (wita) | P 14/16 `ita`x2 | P 15/16 `ita`x1 | P 12/16 `ita`x4 | P 14/16 `ita`x1 | P 14/16 `fita`x1 | P 15/16 `ita`x1 |
| UNAIR (unair) | P 6/16 `uner`x4 | P 14/16 `unayer`x1 | P 8/16 `unaiyer`x5 | P 15/16 `unayer`x1 | P 7/16 `unaiyer`x3 | P 16/16 |
| RAISA (raisa) | P 13/16 `risa`x1 | P 14/16 `rasa`x1 | P 16/16 | P 14/16 `rsa`x1 | P 15/16 `rsa`x1 | P 16/16 |
| **all**: right, acronyms under 7/8 | P 84%, 7 under 7/8<br>runaway 0 | P 66%, 9 under 7/8<br>runaway 0 | P 85%, 6 under 7/8<br>runaway 0 | P 66%, 8 under 7/8<br>runaway 0 | P 85%, 4 under 7/8<br>runaway 0 | P 73%, 7 under 7/8<br>runaway 0 |
| best form per acronym (P) | **hybrid**: 90%; raw for QR IT WIB WITA UNAIR RAISA; under 7/8: API NFC WIB HDMI <br> **norm**: 92%; raw for USB QR NFC ID WIB WITA UNAIR; under 7/8: WIB HDMI <br> **v1**: 94%; raw for IT WIB HDMI KTP WITA UNAIR RAISA; under 7/8: QRIS WIB | | | | | |

## Indonesian sentences, temperature 0.5

| acronym (rule reading) | hybrid phon | hybrid raw | norm phon | norm raw | v1 phon | v1 raw |
|---|---|---|---|---|---|---|
| USB (yuesbi) | P 15/16 ``x1 runaway 1 | P 16/16 | P 15/16 `yu`x1 runaway 1 | P 16/16 | P 16/16 | P 16/16 |
| TV (tifi) | P 10/16 ``x4 runaway 4 | P 10/16 ``x5 runaway 5 | P 9/16 ``x6 runaway 6 | P 11/16 ``x3 runaway 3 | P 16/16 | P 16/16 |
| GPU (jipiyu) | P 14/16 `jipyu`x1 | P 0/16 `gepiu`x10 | P 13/16 `jipyu`x1 | P 0/16 `gepiu`x7 | P 15/16 `jipyu`x1 | P 0/16 `gepeu`x8 |
| GB (jibi) | P 13/16 ``x3 runaway 3 | P 0/16 `gebe`x8 runaway 3 | P 13/16 ``x2 runaway 2 | P 0/16 `gebi`x6 runaway 2 | P 16/16 | P 0/16 `gebe`x11 |
| API (eipiai) | P 8/16 ``x5 runaway 5 | P 0/16 `apii`x12 runaway 3 | P 14/16 `epia`x1 | P 0/16 `apii`x14 runaway 1 | P 15/16 `epiei`x1 | P 0/16 `apei`x12 |
| QR (kyuar) | P 11/16 `kiar`x2 runaway 1 | P 13/16 ``x3 runaway 3 | P 13/16 ``x2 runaway 2 | P 14/16 `indaikodekyuar`x1 runaway 1 | P 13/16 `kyar`x1 runaway 1 | P 16/16 |
| NFC (enefsi) | P 8/16 ``x7 runaway 7 | P 10/16 ``x4 runaway 3 | P 9/16 `nefsi`x3 runaway 3 | P 13/16 ``x2 runaway 2 | P 15/16 `nefsi`x1 | P 13/16 `nefsi`x2 runaway 1 |
| VIP (fiaipi) | P 11/16 ``x5 runaway 5 | P 11/16 ``x3 runaway 3 | P 6/16 ``x7 runaway 7 | P 9/16 ``x4 runaway 4 | P 16/16 | P 13/16 `fiepi`x3 |
| ID (aidi) | P 6/16 ``x10 runaway 10 | P 9/16 ``x7 runaway 7 | P 11/16 ``x5 runaway 5 | P 14/16 ``x2 runaway 2 | P 13/16 ``x2 runaway 2 | P 10/16 ``x6 runaway 6 |
| IT (aiti) | P 9/16 ``x6 runaway 6 | P 10/16 ``x5 runaway 5 | P 11/16 ``x5 runaway 5 | P 13/16 ``x2 runaway 2 | P 13/16 ``x2 runaway 2 | P 13/16 ``x3 runaway 3 |
| PDF (pidief) | P 13/16 ``x3 runaway 3 | P 2/16 `pedeef`x11 runaway 1 | P 15/16 ``x1 runaway 1 | P 1/16 `pedeef`x12 runaway 2 | P 15/16 `kpidief`x1 | P 0/16 `pedeef`x16 |
| QRIS (kyuris) | P 8/16 ``x8 runaway 8 | P 7/16 `kuris`x5 runaway 4 | P 13/16 ``x3 runaway 3 | P 8/16 `kuris`x4 runaway 4 | P 13/16 `kuris`x3 | P 14/16 ``x1 runaway 1 |
| WIB (weibe) | P 0/16 ``x6 runaway 6 | P 2/16 ``x8 runaway 9 | P 1/16 ``x7 runaway 7 | P 6/16 ``x7 runaway 7 | P 0/16 `wiaibi`x5 runaway 2 | P 12/16 ``x2 runaway 2 |
| ITS (itees) | P 14/16 ``x2 runaway 2 | P 16/16 | P 15/16 `aites`x1 | P 16/16 | P 16/16 | P 16/16 |
| HDMI (hadeemi) | P 7/16 `hadeemai`x4 runaway 5 | P 0/16 ``x4 runaway 4 | P 5/16 ``x6 runaway 8 | P 3/16 `hacdeemi`x2 runaway 1 | P 8/16 `hadeemai`x7 | P 14/16 `hatemi`x1 |
| KTP (katepe) | P 12/16 ``x4 runaway 4 | P 12/16 ``x3 runaway 3 | P 12/16 ``x3 runaway 3 | P 14/16 ``x2 runaway 2 | P 13/16 `kapepe`x2 | P 16/16 |
| UMKM (uemkaem) | P 7/16 ``x8 runaway 8 | P 7/16 ``x9 runaway 9 | P 9/16 ``x7 runaway 7 | P 10/16 ``x6 runaway 6 | P 13/16 ``x3 runaway 3 | P 13/16 ``x2 runaway 2 |
| BPJS (bepejees) | P 15/16 `b`x1 runaway 1 | P 16/16 | P 16/16 | P 14/16 `bepjees`x2 | P 15/16 `bepjees`x1 | P 15/16 `bpejees`x1 |
| WITA (wita) | P 12/16 `ita`x3 runaway 1 | P 12/16 ``x4 runaway 4 | P 13/16 ``x2 runaway 3 | P 10/16 `ita`x4 runaway 2 | P 16/16 | P 16/16 |
| UNAIR (unair) | P 7/16 ``x5 runaway 5 | P 10/16 ``x5 runaway 5 | P 8/16 ``x3 runaway 3 | P 11/16 ``x5 runaway 5 | P 13/16 `unaiyer`x2 runaway 1 | P 15/16 ``x1 runaway 1 |
| RAISA (raisa) | P 12/16 ``x2 runaway 2 | P 13/16 ``x2 runaway 2 | P 12/16 ``x3 runaway 3 | P 14/16 `rsa`x1 runaway 1 | P 15/16 `risa`x1 | P 14/16 ``x1 runaway 1 |
| **all**: right, acronyms under 7/8 | P 63%, 17 under 7/8<br>runaway 87 | P 52%, 18 under 7/8<br>runaway 73 | P 69%, 16 under 7/8<br>runaway 69 | P 59%, 14 under 7/8<br>runaway 47 | P 85%, 9 under 7/8<br>runaway 11 | P 72%, 10 under 7/8<br>runaway 17 |
| best form per acronym (P) | **hybrid**: 68%; raw for USB QR NFC ID IT WIB ITS BPJS UNAIR RAISA; under 7/8: TV GB API QR NFC VIP ID IT PDF QRIS WIB HDMI KTP UMKM WITA UNAIR RAISA <br> **norm**: 78%; raw for USB TV QR NFC VIP ID IT WIB ITS KTP UMKM UNAIR RAISA; under 7/8: TV GPU GB NFC VIP IT QRIS WIB HDMI UMKM WITA UNAIR <br> **v1**: 93%; raw for QR QRIS WIB HDMI KTP UNAIR; under 7/8: ID IT WIB UMKM | | | | | |

## English sentences, temperature 0.7

| acronym (rule reading) | hybrid phon | hybrid raw | norm phon | norm raw | v1 phon | v1 raw |
|---|---|---|---|---|---|---|
| HDMI (eicdiemai) | P 14/16 `ejdeemai`x1 | P 12/16 `heictiemai`x1 | P 16/16 | P 12/16 `hejdiemai`x2 | P 15/16 `hejdeemai`x1 | P 9/16 `heijdiemai`x2 |
| ITS (aities) | P 12/16 `aidies`x2 | P 15/16 `itees`x1 | P 15/16 `aitiaz`x1 | P 14/16 `ities`x1 | P 14/16 `aitiest`x1 | P 15/16 `aitdies`x1 |
| QR (kyuar) | P 12/16 `kyuai`x1 | P 14/16 `kyuaaur`x1 | P 10/16 `kyuai`x3 | P 15/16 `kyuai`x1 | P 14/16 `kyu`x1 | P 15/16 `kyiai`x1 |
| API (eipiai) | P 13/16 `eipi`x1 | P 14/16 `eipi`x2 | P 14/16 `piai`x1 | P 16/16 | P 7/16 `eipieiai`x4 | P 16/16 |
| **all**: right, acronyms under 7/8 | P 80%, 3 under 7/8<br>runaway 0 | P 86%, 1 under 7/8<br>runaway 0 | P 86%, 1 under 7/8<br>runaway 0 | P 89%, 1 under 7/8<br>runaway 0 | P 78%, 1 under 7/8<br>runaway 0 | P 86%, 1 under 7/8<br>runaway 0 |
| best form per acronym (P) | **hybrid**: 89%; raw for ITS QR API; under 7/8: - <br> **norm**: 97%; raw for QR API; under 7/8: - <br> **v1**: 95%; raw for ITS QR API; under 7/8: - | | | | | |

## English sentences, temperature 0.5

| acronym (rule reading) | hybrid phon | hybrid raw | norm phon | norm raw | v1 phon | v1 raw |
|---|---|---|---|---|---|---|
| HDMI (eicdiemai) | P 4/16 ``x12 runaway 12 | P 10/16 ``x4 runaway 4 | P 7/16 ``x7 runaway 7 | P 6/16 ``x6 runaway 6 | P 10/16 `hecdeemai`x2 | P 10/16 `hejdiemai`x3 runaway 1 |
| ITS (aities) | P 10/16 ``x5 runaway 5 | P 8/16 ``x8 runaway 8 | P 12/16 ``x3 runaway 3 | P 6/16 ``x6 runaway 7 | P 13/16 ``x2 runaway 2 | P 14/16 ``x2 runaway 2 |
| QR (kyuar) | P 10/16 ``x6 runaway 6 | P 8/16 ``x6 runaway 6 | P 11/16 ``x4 runaway 4 | P 11/16 ``x4 runaway 4 | P 10/16 `kua`x2 runaway 2 | P 15/16 ``x1 runaway 1 |
| API (eipiai) | P 2/16 ``x8 runaway 8 | P 7/16 ``x9 runaway 9 | P 7/16 ``x7 runaway 7 | P 8/16 ``x7 runaway 8 | P 8/16 `eipiaiai`x3 | P 14/16 `eipai`x1 runaway 1 |
| **all**: right, acronyms under 7/8 | P 41%, 4 under 7/8<br>runaway 31 | P 52%, 4 under 7/8<br>runaway 27 | P 58%, 4 under 7/8<br>runaway 21 | P 48%, 4 under 7/8<br>runaway 25 | P 64%, 4 under 7/8<br>runaway 4 | P 83%, 1 under 7/8<br>runaway 5 |
| best form per acronym (P) | **hybrid**: 58%; raw for HDMI API; under 7/8: HDMI ITS QR API <br> **norm**: 59%; raw for API; under 7/8: HDMI ITS QR API <br> **v1**: 83%; raw for ITS QR API; under 7/8: HDMI | | | | | |

## What the numbers say

| finding | numbers (P, temperature 0.7, Indonesian) |
| --- | --- |
| Raw acronyms on the served hybrid: right 66%, 9 acronyms under 7/8 | GPU 0/16 "gepiu", GB 0/16 "gebe", API 0/16 "api", PDF 4/16 "pedeef", HDMI 0/16 "hasdeemi", QRIS 9/16 "kuris", WIB 6/16 |
| The reading written out fixes the consonant errors on every model | GPU 15-16/16, GB 15-16/16, PDF 15-16/16 |
| "we i be" (served for WIB) is wrong on every model: the lone "i" is read English "ai" | 0-3/16 (P), 1-3/16 (G); "weibe" / "wei be" 12-16/16 (G) |
| "ha de em i" -> "ha de em ai" for the same reason; "ha de emi" is 16/16 on all three models with both judges | |
| Best lexicon per model, Indonesian: v1 94%, norm 92%, hybrid 90%; none reaches 95% with no acronym under 7/8 | v1 misses QRIS, WIB; norm WIB, HDMI; hybrid API, NFC, WIB, HDMI |
| English sentences: raw is right 86-89%; per acronym (HDMI written out, the rest raw) norm 97%, v1 95%, hybrid 89%, none under 7/8 | QR written "kyu ar" loses its r |
| Judges: agree on 82% of spelling clips, Cohen's kappa 0.64; G is more lenient (552 vs 479 right of 912) | |

Needs data (no spelling fixes them reliably on any model): QRIS (kyuris/kuris), WIB, HDMI in Indonesian if kept raw,
API, NFC on hybrid; plus the non-acronym words "seekor" and "keenam" (see smooth_report.md).
