# F5-TTS Voice-Cloning Evaluation Report

*Generated 2026-09-28 10:37 - checkpoint `best_model.pt` (epoch 91, 0.63 GB), Whisper `large-v3`, CFG {2.0}, NFE {32}, sway {-1.0}, speed 1.0, cross-fade 0.15 s, seed 0, chunking on.*

## 1. Setup

* Test split: **21** held-out utterances (8.1 min of real speech), none of which share a recording with the training data (duplicates were removed by fingerprint + waveform cross-correlation before splitting).
* Conditions: `alpha_a` = weight interpolation (1-a)·pretrained + a·fine-tuned over the EMA weights (0.0, 1.0); `baseline` is alpha 0 (zero-shot `F5TTS_Base`), `finetuned` alpha 1 (`best_model.pt`); `copysynth` = the real recordings passed through mel -> vocos (the vocoder ceiling); `gt` = the real recordings.
* Prompting: `cross` = <= 12 s exact-text prompt cut from a *different* test clip (forced alignment keeps the text exact). `fixed` = one enrolment prompt for all utterances. The total duration is fixed to prompt + ground-truth length (on).
* `fixed` prompt: `data/splits/valid/38.wav` cut to 9.8 s, DNSMOS OVRL 3.14 (best of 21 valid-split candidates), used for every utterance.
* Generated audio: `output/evaluation/<condition>/<prompt>/`, copy-synthesis: `output/evaluation/copysynth/`, prompts: `output/evaluation/prompts/`, `prompts_fixed/` (kept out of `output/generated_audio/`).

## 2. Metric sanity: SIM-o separates speakers, resemblyzer does not

WavLM-large + ECAPA-TDNN cosine (the Seed-TTS / F5-TTS "SIM-o" model):

| pairs | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| same speaker (real test clip vs real test clip) | 210 | 0.603 | 0.610 | 0.327 | 0.930 |
| different speaker (real test clip vs F5-TTS example speakers) | 105 | 0.029 | 0.031 | -0.202 | 0.314 |
| same speaker, resemblyzer (for comparison) | 210 | 0.879 | 0.881 | 0.712 | 0.985 |
| different speaker, resemblyzer | 105 | 0.573 | 0.548 | 0.442 | 0.781 |

The same-speaker SIM-o spread across this speaker's own recordings is the reference for what a "perfect" clone can score against a different real recording: the target is the same-speaker distribution, not 1.0.

## 3. Ceiling: real audio vs the vocoder path vs the models

| Metric | real audio | copy-synthesis (mel->vocos) | baseline / cross | finetuned / cross | baseline / fixed | finetuned / fixed |
|---|---:|---:|---:|---:|---:|---:|
| NMOS (higher better) | 1.305 | 1.296 | 1.734 | 1.425 | 1.932 | 1.568 |
| MOS (higher better) | 2.673 | 2.692 | 3.017 | 2.863 | 3.093 | 3.044 |
| ASR-WER (lower better) | 0.120 | 0.138 | 0.207 | 0.263 | 0.106 | 0.106 |
| ASR-CER (lower better) | 0.056 | 0.069 | 0.139 | 0.174 | 0.059 | 0.054 |
| SIM-o (higher better) | - | 0.919 | 0.447 | 0.579 | 0.542 | 0.619 |
| MCD (lower better) | - | 3.350 | 13.259 | 13.159 | 12.003 | 11.719 |
| VDE (lower better) | - | 0.061 | 0.313 | 0.312 | 0.322 | 0.304 |

If copy-synthesis sits at the real-audio level, the recordings themselves bound the naturalness scores and no training can lift a clone above the pretrained baseline on NMOS/DNSMOS; if copy-synthesis is well above real audio, the models are losing something the mel/vocoder path preserves.

## 4. Results per prompting mode

### `cross` prompt

| condition | alpha | CFG | NFE | sway | retained % [95 % CI] | WER | SIM-o | CER | SIM-o(prompt) | SEC | MCD | VDE | FAD | NMOS | SMOS | CMOS | MOS | normalised % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 2.0 | 32 | -1.0 | 18.4 [2.2, 36.5] | **0.207** | **0.447** | 0.139 | 0.594 | 0.852 | 13.26 | 0.313 | 5.49 | 1.734 | 4.13 | - | 3.017 | 67.3 |
| finetuned | 1.00 | 2.0 | 32 | -1.0 | 20.4 [4.3, 39.0] | **0.263** | **0.579** | 0.174 | 0.637 | 0.859 | 13.16 | 0.312 | 3.77 | 1.425 | 4.19 | -0.310 | 2.863 | 63.3 |

Paired deltas vs `baseline` over the same 21 utterances, mean [95 % bootstrap CI, 10 000 resamples]; `*` = CI excludes 0. Retention deltas are in percentage points over the shared markers:

| condition | Δretained (pp) | ΔWER | ΔSIM-o | ΔCER | ΔSIM-o(prompt) | ΔSEC | ΔMCD | ΔVDE | ΔNMOS | ΔMOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| finetuned | +2.0 [-10.7, +15.2] | +0.056 [+0.002, +0.114] * | +0.035 [-0.031, +0.098] | +0.132 [+0.087, +0.179] * | +0.042 [+0.002, +0.088] * | +0.007 [-0.005, +0.019] | -0.10 [-0.66, +0.47] | -0.001 [-0.025, +0.026] | -0.310 [-0.443, -0.184] * | -0.154 [-0.258, -0.051] * |

* Conditions that **dominate the fine-tuned checkpoint at default decoding** on the objective axes (>= retention, <= WER, >= SIM-o, strictly better on at least one): none.
* Conditions that dominate the zero-shot baseline on the same axes: none.

### `fixed` prompt

| condition | alpha | CFG | NFE | sway | retained % [95 % CI] | WER | SIM-o | CER | SIM-o(prompt) | SEC | MCD | VDE | FAD | NMOS | SMOS | CMOS | MOS | normalised % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 2.0 | 32 | -1.0 | 22.4 [4.9, 41.2] | **0.106** | **0.542** | 0.059 | 0.672 | 0.867 | 12.00 | 0.322 | 7.97 | 1.932 | 4.26 | - | 3.093 | 69.4 |
| finetuned | 1.00 | 2.0 | 32 | -1.0 | 26.5 [7.5, 47.4] | **0.106** | **0.619** | 0.054 | 0.696 | 0.876 | 11.72 | 0.304 | 5.64 | 1.568 | 4.35 | -0.363 | 3.044 | 61.2 |

Paired deltas vs `baseline` over the same 21 utterances, mean [95 % bootstrap CI, 10 000 resamples]; `*` = CI excludes 0. Retention deltas are in percentage points over the shared markers:

| condition | Δretained (pp) | ΔWER | ΔSIM-o | ΔCER | ΔSIM-o(prompt) | ΔSEC | ΔMCD | ΔVDE | ΔNMOS | ΔMOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| finetuned | +4.1 [-9.1, +18.4] | +0.000 [-0.054, +0.042] | -0.005 [-0.062, +0.031] | +0.077 [+0.050, +0.102] * | +0.024 [+0.006, +0.041] * | +0.010 [+0.000, +0.019] * | -0.28 [-0.74, +0.15] | -0.017 [-0.042, +0.011] | -0.363 [-0.459, -0.274] * | -0.048 [-0.158, +0.054] |

* Conditions that **dominate the fine-tuned checkpoint at default decoding** on the objective axes (>= retention, <= WER, >= SIM-o, strictly better on at least one): none.
* Conditions that dominate the zero-shot baseline on the same axes: none.

Chunking (f5_tts splits long text at `max_chars` derived from the prompt; 1.1.22 allocates the fixed duration across chunks by text length):

* `baseline` / `cross`: 14 of 21 utterances synthesised in >1 chunk (19: 8, 81: 5, 117: 5, 55: 4, 160: 4, 97: 3, 239: 3, 11: 2, 30: 2, 52: 2, 70: 2, 180: 2, 216: 2, 291: 2)
* `baseline` / `fixed`: 16 of 21 utterances synthesised in >1 chunk (19: 5, 81: 5, 97: 5, 55: 4, 70: 3, 117: 3, 160: 3, 239: 3, 11: 2, 30: 2, 51: 2, 103: 2, 180: 2, 216: 2, 242: 2, 291: 2)
* `finetuned` / `cross`: 14 of 21 utterances synthesised in >1 chunk (19: 8, 81: 5, 117: 5, 55: 4, 160: 4, 97: 3, 239: 3, 11: 2, 30: 2, 52: 2, 70: 2, 180: 2, 216: 2, 291: 2)
* `finetuned` / `fixed`: 16 of 21 utterances synthesised in >1 chunk (19: 5, 81: 5, 97: 5, 55: 4, 70: 3, 117: 3, 160: 3, 239: 3, 11: 2, 30: 2, 51: 2, 103: 2, 180: 2, 216: 2, 242: 2, 291: 2)

Metric notes:

* **ASR-WER / CER** - Whisper `large-v3` transcripts vs the written transcript after lower-casing and punctuation removal. Because the transcript deliberately spells non-standard pronunciations, the *real* recordings do not score 0 either: the ground-truth WER above is the ceiling for this speaker.
* **SIM-o** - WavLM-large + ECAPA-TDNN speaker-verification cosine between the generated utterance and the real recording of the same text; **SIM-o (prompt)** is the same model against the prompt audio (the paper definition). **SEC** is resemblyzer (GE2E), kept for continuity; it saturates on same-speaker pairs (section 2).
* **MCD** - `pymcd` mel-cepstral distortion (dB) with DTW alignment. **VDE** - fraction of DTW-aligned frames whose pyin voicing decision differs.
* **FAD** - Frechet distance between wav2vec2-base layer-6 embeddings (1 s windows) of the generated set and the real test set; set-level, indicative only at n = 21.
* **NMOS** - UTMOS22-strong; **MOS** - DNSMOS P.835 OVRL. The real clips are RoFormer-isolated vocals from music-backed recordings, so their scores are a reference point, not an upper bound.
* **SMOS** - SEC mapped from [0.50, 0.95] to [1, 5]. **CMOS** - NMOS(condition) - NMOS(baseline) per utterance (> 0 = clone preferred).

## 5. Training convergence and stability

* Checkpoint on disk: **epoch 91** = trainer's best epoch **91**, validation loss **0.8913**.
* Run: 100 epochs, 4636 optimizer updates (planned 60, `--grad_accum` 1, `--lr` 1e-05, `--val_repeats` 4).
* First epoch: train 0.9897 / val 0.9810 -> final epoch: train 0.7785 / val 0.8916 (val change -9.1 % to the best epoch).
* Last 10 epochs: val-loss std 0.0001, train-loss std 0.0585; the CFM loss is an MSE against a random (timestep, mask, cond-drop) draw, so its level and flatness say little about clone quality - see the metric tables instead.
* Learning rate: 9.99e-06 peak -> 1.00e-06 final; peak VRAM 9.4 GiB; mean epoch time 0.7 min.
* Curves: `logs/training_convergence.png`; metrics table: `logs/training_metrics.csv` / `.xlsx`.

## 6a. Phone-level pronunciation retention (primary measurement)

Markers are words where the **speaker's** realised phone sequence differs from the **zero-shot pretrained model's** realisation of the same word in the same sentence by at least `0.7` PER (phone edit distance, forced-aligned word spans, CTC phone recogniser). Homophones, inflection and tokenisation splits are excluded *by construction* - they produce near-identical phone strings and so never clear the threshold.

`SPD` = mean phone distance from the condition to the **speaker** (lower = closer to how he actually says the word). It never references the prior, so unlike `retained %` it cannot be inflated by a condition merely drifting away from the pretrained model.

### `cross` prompt

* Markers: **218** word types of 778 aligned words (one marker per word type per utterance; repeated tokens averaged).
* Synthesis seeds pooled per condition: **1**  _(single seed - see the caveat below)_
* Anchors: zero-shot `baseline` SPD 0.872 (floor), copy-synthesis SPD 0.258 (ceiling).

| condition | SPD | 95 % CI | % of the way to the speaker | retained % | 95 % CI | n |
|---|---|---|---|---|---|---|
| `copysynth` | 0.258 | [0.218, 0.315] | 100.0 % | 89.7 % | [81.7, 95.4] | 214 |
| `finetuned` | 0.721 | [0.645, 0.794] |  24.5 % | 28.3 % | [16.2, 41.4] | 191 |
| `baseline` | 0.872 | [0.862, 0.882] |   0.0 % | 0.0 % | [0.0, 0.0] | 218 |

Paired deltas vs `baseline` (CI excludes zero = the difference is real):

| condition | dSPD | 95 % CI | separates? |
|---|---|---|---|
| `copysynth` | -0.612 | [-0.648, -0.559] | yes |
| `finetuned` | -0.139 | [-0.214, -0.069] | yes |

### `fixed` prompt

* Markers: **209** word types of 778 aligned words (one marker per word type per utterance; repeated tokens averaged).
* Synthesis seeds pooled per condition: **1**  _(single seed - see the caveat below)_
* Anchors: zero-shot `baseline` SPD 0.862 (floor), copy-synthesis SPD 0.253 (ceiling).

| condition | SPD | 95 % CI | % of the way to the speaker | retained % | 95 % CI | n |
|---|---|---|---|---|---|---|
| `copysynth` | 0.253 | [0.210, 0.316] | 100.0 % | 90.3 % | [83.3, 95.5] | 206 |
| `finetuned` | 0.712 | [0.651, 0.772] |  24.7 % | 20.6 % | [12.0, 29.9] | 204 |
| `baseline` | 0.862 | [0.848, 0.875] |   0.0 % | 0.0 % | [0.0, 0.0] | 209 |

Paired deltas vs `baseline` (CI excludes zero = the difference is real):

| condition | dSPD | 95 % CI | separates? |
|---|---|---|---|
| `copysynth` | -0.607 | [-0.649, -0.545] | yes |
| `finetuned` | -0.149 | [-0.206, -0.093] | yes |

**Reliability caveat.** `retained %` is a thresholded statistic and is noisy at a single seed: two independent syntheses of *identical weights* have differed by 10 pp on the `fixed` prompt, with a paired CI excluding zero. `SPD` passed that same identity test. Pooled over >= 3 seeds both statistics agree between independent training runs to within ~2-5 pp. Read single-seed `retained %` as indicative only; use `SPD` with >= 3 pooled seeds for any comparison that decides something.

## 6b. Idiosyncratic pronunciation assessment - legacy ASR proxy (`finetuned`, `cross` prompt, Whisper `large-v3`)

The speaker's transcript spells words the way he says them. Whisper therefore *mis-hears* those words on the **real** audio, which marks every non-standard pronunciation in the test set; for each marker we check what Whisper hears on the clone:

* Markers found in the real test audio: **49** words across 21 utterances.
* **Retained** (clone mis-heard the same way -> pronunciation reproduced): **10 (20.4 %)**
* **Normalised** (Whisper hears the dictionary spelling -> clone "corrected" the accent): 31 (63.3 %)
* Other: 8. The zero-shot baseline retains 18.4 %.

| utterance | written | heard on real audio | heard on clone | outcome |
|---|---|---|---|---|
| 7 | shh | sweet |  | other |
| 11 | suleiman | sullivan | suleiman | normalised |
| 11 | suleiman | sullivan | suleiman | normalised |
| 11 | suleiman | sullivan | suleiman | normalised |
| 11 | suleiman | sullivan | suleiman | normalised |
| 13 | a | others | a | normalised |
| 19 | post | postgraduate | postgraduate | retained |
| 19 | ranking | ranked | ranking | normalised |
| 19 | university | universities | universities | retained |
| 30 | youre | work | are | other |
| 30 | working | in | working | normalised |
| 52 | piece | peace | peace | retained |

Retention for every condition is in the tables of section 4 (`retained %`, with bootstrap CIs over utterances).

Caveat on the marker definition: a marker is "Whisper hears a different word on the real audio", which also fires on Whisper's own lexical priors (rare proper nouns such as `suleiman` -> `sullivan`), on morphology (`ranking` -> `ranked`, `university` -> `universities`) and on tokenisation (`post` -> `postgraduate`). Those are counted alongside genuine segmental idiosyncrasies, and a clone that pronounces a name *correctly per the transcript* is scored "normalised". Treat the aggregate as an ASR-proxy, not a phonetic measurement - section 6a measures the phones directly and supersedes it. This section is retained only for continuity with earlier reports.

## 7. Per-utterance results (`finetuned`, `cross` prompt)

|   stem |   duration |   n_chunks |   wer |   cer |   sim_o |   sec |    mcd |   vde |   nmos |   cmos |   mos |
|-------:|-----------:|-----------:|------:|------:|--------:|------:|-------:|------:|-------:|-------:|------:|
|      7 |     13.700 |      1.000 | 0.167 | 0.115 |   0.633 | 0.771 | 15.770 | 0.426 |  1.724 |  0.004 | 3.084 |
|     11 |     20.262 |      2.000 | 0.136 | 0.104 |   0.666 | 0.894 | 12.228 | 0.183 |  1.413 | -0.721 | 2.999 |
|     13 |     11.600 |      1.000 | 0.522 | 0.484 |   0.346 | 0.822 | 17.129 | 0.600 |  1.290 | -0.229 | 2.803 |
|     19 |     44.677 |      8.000 | 0.362 | 0.242 |   0.589 | 0.886 | 15.623 | 0.295 |  1.285 | -0.898 | 2.687 |
|     30 |     17.011 |      2.000 | 0.273 | 0.166 |   0.623 | 0.800 | 14.498 | 0.309 |  1.544 | -0.137 | 3.007 |
|     51 |     17.630 |      1.000 | 0.286 | 0.150 |   0.680 | 0.839 | 13.313 | 0.250 |  1.363 | -0.091 | 2.677 |
|     52 |     14.190 |      2.000 | 0.143 | 0.039 |   0.393 | 0.801 | 16.888 | 0.311 |  1.341 | -0.516 | 2.708 |
|     55 |     39.312 |      4.000 | 0.275 | 0.111 |   0.629 | 0.898 | 15.634 | 0.251 |  1.412 | -0.988 | 2.587 |
|     70 |     33.565 |      2.000 | 0.136 | 0.092 |   0.710 | 0.867 | 10.656 | 0.320 |  1.323 | -0.082 | 2.883 |
|     81 |     46.564 |      5.000 | 0.078 | 0.021 |   0.815 | 0.940 | 10.970 | 0.225 |  1.397 | -0.068 | 3.082 |
|     97 |     37.820 |      3.000 | 0.037 | 0.006 |   0.734 | 0.945 | 10.379 | 0.223 |  1.392 | -0.110 | 3.098 |
|    103 |     16.179 |      1.000 | 0.545 | 0.061 |   0.544 | 0.899 | 11.610 | 0.307 |  1.753 | -0.321 | 3.119 |
|    107 |      8.221 |      1.000 | 1.100 | 0.976 |   0.344 | 0.806 | 15.217 | 0.326 |  1.304 | -0.433 | 2.300 |
|    117 |     27.093 |      5.000 | 0.765 | 0.589 |   0.502 | 0.855 | 17.167 | 0.344 |  1.242 | -0.176 | 2.446 |
|    160 |     36.943 |      4.000 | 0.150 | 0.134 |   0.718 | 0.927 | 10.719 | 0.296 |  1.350 | -0.069 | 2.967 |
|    180 |     12.616 |      2.000 | 0.067 | 0.019 |   0.475 | 0.800 | 10.238 | 0.269 |  1.383 | -0.078 | 3.210 |
|    216 |     14.735 |      2.000 | 0.061 | 0.026 |   0.643 | 0.872 | 11.131 | 0.245 |  1.456 | -0.691 | 3.127 |
|    239 |     28.766 |      3.000 | 0.062 | 0.018 |   0.567 | 0.861 | 13.864 | 0.326 |  1.306 |  0.000 | 2.761 |
|    242 |     21.000 |      1.000 | 0.172 | 0.205 |   0.574 | 0.890 |  8.990 | 0.451 |  1.452 | -0.316 | 2.763 |
|    291 |     14.941 |      2.000 | 0.000 | 0.000 |   0.633 | 0.902 | 12.462 | 0.268 |  1.753 | -0.633 | 2.905 |
|    313 |      9.743 |      1.000 | 0.182 | 0.103 |   0.341 | 0.756 | 11.847 | 0.330 |  1.431 |  0.050 | 2.917 |

## 8. Files

* `checkpoints/evaluation_frontier.png` - SIM-o / retention against NMOS / DNSMOS / WER across alpha (both prompt modes).
* `checkpoints/evaluation_metrics.png` - box/strip plots of every metric (fine-tuned vs baseline vs copy-synthesis vs ground truth) and the FAD bars.
* `checkpoints/evaluation_results.csv` / `.json` - long-form per-utterance numbers (one row per condition x prompt x utterance), transcripts and pronunciation markers, plus the condition summary with CIs.
* `logs/training_convergence.png`, `logs/training_metrics.csv` / `.xlsx`, `logs/finetune_tts.log` - training curves, metrics and log.
