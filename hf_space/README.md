---
title: Aktar_Khan_TTS
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.28.0"
app_file: app.py
pinned: false
---

# Fine-tuned F5-TTS voice clone

A single-speaker fine-tune of [F5-TTS](https://github.com/SWivid/F5-TTS) (`F5TTS_Base`,
337M-parameter DiT + conditional flow matching), trained on ~67 minutes of one speaker's
recordings to reproduce their accent and idiosyncratic pronunciation rather than the
base model's standard-pronunciation prior.

## ⚠️ Before you deploy this publicly

This clones **one specific, identifiable real person's voice**. As shipped, this Space lets
anyone type arbitrary text and generate audio in that voice. Before making it public:

- Confirm you have the speaker's explicit consent for **open-ended, public** use of their
  cloned voice (consent to the original recordings/training data is not the same thing).
- Consider gating the Space (private, or behind acknowledgement of terms), limiting it to a
  fixed set of demo sentences, or watermarking output, rather than unrestricted free-text input.
- Add a visible disclosure that audio is AI-generated wherever it's shared downstream.

## Model variants

Two checkpoints are bundled, both EMA weights extracted as
`theta(alpha) = (1-alpha) * theta_pretrained + alpha * theta_finetuned`:

| variant | when to use it | speaker-phone-distance (SPD)\* | WER | SIM-o |
|---|---|---|---|---|
| **alpha = 1.0** (default) | WER matters | 0.745 | 0.120 (matches real speaker's own WER of 0.120) | 0.615 |
| alpha = 1.2 | pronunciation-retention matters more | 0.688 (\*\*significantly lower\*\*, CI [-0.092,-0.019]) | 0.196 (significantly higher) | 0.637 |

\*SPD = mean phone edit distance between the clone's realisation of idiosyncratic-pronunciation
marker words and the real speaker's realisation of the same words (lower = closer to the real
speaker; see Methodology). This is not a free lunch: alpha=1.2 buys retention at a real,
statistically confirmed WER cost. Neither variant dominates the other — pick based on your priority.

## Usage

Type text in the box and press Generate. The bundled reference voice (`reference/`) is used by
default; you can substitute your own clean 5-12s reference clip + exact transcript in the
"advanced" section if you want to clone a different voice with the same fine-tuned model weights
(F5-TTS is a *zero-shot* architecture - the fine-tune biases it toward this speaker's phonology,
but it still accepts any reference at inference time).

Recommended decoding (already the defaults): NFE=32, CFG=2.0, sway=-1.0. A systematic 60-point
grid search over CFG x NFE x sway found no configuration that beats these defaults by a
statistically significant margin - the model/alpha choice matters far more than decoding
parameters.

## Methodology summary

- **Retention metric**: word-level markers are the words where the real speaker's phone
  realisation (forced-aligned + CTC phone-recognised) diverges from the pretrained model's
  zero-shot realisation of the same word by a calibrated threshold (excludes homophones,
  inflection, and tokenisation artefacts by construction, since those produce near-identical
  phone strings). Scored by phone edit distance to the real speaker vs. to the pretrained prior.
- **Validated for reproducibility**: two independent trainings of this model (different
  epochs/checkpoints, statistically identical validation loss) agree on this metric within
  ~2-5 percentage points when pooled over >= 3 synthesis seeds - the metric passed an identity
  test (same weights, independently synthesised twice) that an earlier ASR-mishearing-based
  metric failed.
- **Dominance testing is confidence-interval-aware**: a configuration is only called "better"
  or "worse" when the paired bootstrap CI of the difference excludes zero, not from point
  estimates alone (previous point-estimate-only comparisons produced false verdicts due to
  synthesis-seed noise).

## Files

```
app.py                                    Gradio inference app
config.json                               architecture + tokenizer + decoding defaults
requirements.txt
vocab.txt                                 F5-TTS custom tokenizer vocabulary
weights/f5tts_finetuned_alpha1.0.safetensors   EMA weights, alpha=1.0 (recommended default)
weights/f5tts_finetuned_alpha1.2.safetensors   EMA weights, alpha=1.2 (retention-favouring alternative)
reference/voice_reference.wav             default enrolment prompt (9.8s, DNSMOS-selected as the
                                           best of 21 candidates from a held-out validation split)
reference/voice_reference.txt             exact transcript of the reference audio
```

## License

Base model [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) is released under
CC-BY-NC-4.0 (non-commercial). This fine-tune inherits that restriction unless you have a
separate agreement with the base model's authors - do not use commercially without checking.
