#!/usr/bin/env python
"""
Script 3 - evaluation.py

Objective + neural evaluation of the fine-tuned F5-TTS voice clone on the
held-out test split, organised as a set of *conditions* that all go through
the same metric path so they can be compared with paired statistics:

  * alpha_<a>   weight interpolation (1-a)*pretrained + a*fine-tuned over the EMA
                weights (--alphas).  alpha 0 = zero-shot F5TTS_Base ("baseline"),
                alpha 1 = the fine-tuned checkpoint ("finetuned").  Sweeping alpha
                traces the fidelity / naturalness frontier without retraining.
  * copysynth   real test audio -> mel -> vocos -> audio (the vocoder ceiling)
  * gt          the real test recordings scored with the same backends

Each model condition is synthesised under two prompting modes (--prompt_mode):
  * cross   <= 12 s exact-text prompt cut from a *different* test clip (as before)
  * fixed   one fixed enrolment prompt (highest DNSMOS <= 12 s cut of the valid
            split, or --enrol_wav/--enrol_text), which removes the rate mismatch
            between prompt and target that the cross prompt carries.

Metrics per utterance (fine-tuned vs baseline deltas get paired-bootstrap 95 % CIs):
  1-2 ASR-WER / CER   Whisper (--whisper_model, default large-v3) + jiwer
  3   SIM-o           WavLM-large + ECAPA-TDNN speaker-verification cosine vs the real
                      recording of the same text (primary speaker metric); SIM-o vs the
                      prompt is reported too (the Seed-TTS / F5-TTS paper definition)
  3b  SEC             resemblyzer (GE2E) cosine, kept for continuity with past reports
  4   MCD             pymcd, DTW mode
  5   VDE             pyin voicing flags along an MFCC-DTW path
  6   FAD             Frechet distance of wav2vec2-base embeddings, set-level
  7   NMOS            UTMOS22-strong
  8   SMOS            SEC mapped to 1-5
  9   CMOS            NMOS(condition) - NMOS(baseline) per utterance
 10   MOS             DNSMOS P.835 OVRL

plus the idiosyncratic-pronunciation retention analysis per condition.

Outputs:
  output/evaluation/<condition>/<prompt_mode>/*.wav, output/evaluation/copysynth/*.wav
  output/evaluation/prompts/ (cross), output/evaluation/prompts_fixed/ (enrolment prompt)
  checkpoints/evaluation_report.md, evaluation_results.csv/.json (long-form, one row per
  condition x prompt x utterance), evaluation_metrics.png, evaluation_frontier.png

Decoding conditions (--cfg_strengths / --nfe_steps / --sways, cross product) multiply every alpha:
a non-default decoding gets a `_cfg1.3_nfe64_sway0` suffix on its condition tag.  --delta_ref
picks the condition the paired deltas are measured against ('baseline' by default, 'finetuned'
for a decoding sweep).  Retention carries a bootstrap CI over utterances in every table.

Usage:
    python scripts/evaluation.py                                   # alphas 0,1  x  cross+fixed  + copysynth
    python scripts/evaluation.py --alphas 0,0.2,0.4,0.6,0.8,1,1.2,1.5      # interpolation / extrapolation frontier
    python scripts/evaluation.py --alphas 1 --prompt_mode fixed --cfg_strengths 1,1.3,1.6,2,2.5 \
        --nfe_steps 32,64 --sways -1,0 --delta_ref finetuned         # decoding sweep on the fine-tuned weights
    python scripts/evaluation.py --limit 5 --alphas 1 --prompt_mode cross --no_copysynth   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CHECKPOINT_DIR, EVAL_GEN_DIR, TARGET_SR  # noqa: E402

log = common.get_logger("evaluation")

PROMPT_DIR = EVAL_GEN_DIR / "prompts"
FIXED_PROMPT_DIR = EVAL_GEN_DIR / "prompts_fixed"
COPYSYNTH_DIR = EVAL_GEN_DIR / "copysynth"
MODELS_DIR = CHECKPOINT_DIR / "eval_models"
PLOT_PNG = CHECKPOINT_DIR / "evaluation_metrics.png"
FRONTIER_PNG = CHECKPOINT_DIR / "evaluation_frontier.png"
REPORT_MD = CHECKPOINT_DIR / "evaluation_report.md"
RESULTS_CSV = CHECKPOINT_DIR / "evaluation_results.csv"
RESULTS_JSON = CHECKPOINT_DIR / "evaluation_results.json"
HISTORY_CSV = common.LOG_DIR / "training_metrics.csv"
TRANSCRIPT_CACHE = EVAL_GEN_DIR / "whisper_transcripts.json"
METRIC_CACHE = EVAL_GEN_DIR / "metric_cache.json"

DNSMOS_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
SIMO_CKPT = "hf://bezzam/wavlm_large_finetune_seed_tts_eval/wavlm_large_finetune.pth"  # UniSpeech WavLM-large SV mirror

# metric id -> (label, direction, unit/notes)
METRICS = {
    "wer": ("ASR-WER", "lower", "word error rate"),
    "cer": ("ASR-CER", "lower", "character error rate"),
    "sim_o": ("SIM-o", "higher", "WavLM-ECAPA cosine vs real recording"),
    "sim_o_prompt": ("SIM-o (prompt)", "higher", "WavLM-ECAPA cosine vs prompt"),
    "sec": ("SEC", "higher", "resemblyzer cosine vs real recording"),
    "mcd": ("MCD", "lower", "dB"),
    "vde": ("VDE", "lower", "voicing decision error rate"),
    "fad": ("FAD", "lower", "Frechet audio distance (set-level)"),
    "nmos": ("NMOS", "higher", "UTMOS22 neural MOS, 1-5"),
    "smos": ("SMOS", "higher", "similarity MOS proxy, 1-5"),
    "cmos": ("CMOS", "higher", "NMOS(condition) - NMOS(baseline)"),
    "mos": ("MOS", "higher", "DNSMOS P.835 OVRL, 1-5"),
}
# metrics reported with paired-bootstrap CIs against the baseline, and their sign for "better"
DELTA_METRICS = ["wer", "cer", "sim_o", "sim_o_prompt", "sec", "mcd", "vde", "nmos", "mos"]
FIDELITY_METRICS = ["sim_o", "retention"]  # y-axes of the frontier plot
NATURALNESS_METRICS = ["nmos", "mos", "wer"]  # x-axes of the frontier plot

# reference palette (dataviz skill): blue = fine-tuned, orange = baseline, aqua = ground truth
C_FT, C_BASE, C_GT, C_CS = "#2a78d6", "#eb6834", "#1baf7a", "#8a6fd6"
C_TEXT, C_TEXT2, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def free_gpu(*objs):
    """
    Drop references and return the VRAM they held.

    NOTE: `del o` here only clears this function's own parameter binding - it cannot reach the
    caller's variable.  Callers must `del` their own name as well, otherwise the model stays
    resident.  Every metric backend below used to leak this way, which left ~10.5 GiB allocated and
    made the phone-level aligner OOM on the longer clips once it ran in the same process.
    """
    import torch

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json_atomic(path: Path, obj, indent: int | None = None) -> None:
    """
    Write a cache via a temp file + atomic replace.  These caches are rewritten in full after every
    entry, and a long run that is killed mid-write would otherwise leave a truncated (or 0-byte) file
    and lose everything computed so far.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
    tmp.replace(path)


def load_json_cache(path: Path) -> dict:
    """Load a cache, starting fresh (with a warning) if it is missing or was damaged by an interrupted write."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning(f"{path.name} is unreadable ({e}); starting from an empty cache")
        return {}


def normalize_words(text: str) -> str:
    import jiwer

    tr = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    return tr(text.replace("’", "'").replace("-", " "))


DEFAULT_DEC = (2.0, 32, -1.0)  # (cfg_strength, nfe_step, sway_sampling_coef) of the F5-TTS defaults


def cond_tag(alpha: float, dec: tuple = DEFAULT_DEC) -> str:
    """'baseline' / 'finetuned' / 'alpha_0.40', with a '_cfg1.3_nfe64_sway0.0' suffix for non-default decoding."""
    tag = "baseline" if alpha == 0.0 else "finetuned" if alpha == 1.0 else f"alpha_{alpha:.2f}"
    cfg, nfe, sway = dec
    if (cfg, nfe, sway) != DEFAULT_DEC:
        tag += f"_cfg{cfg:g}_nfe{nfe}_sway{sway:g}"
    return tag


class MetricCache:
    """{metric|path|mtime_ns[|path2|mtime2] : value} so re-runs only recompute what changed."""

    def __init__(self, path: Path):
        self.path = path
        self.d = load_json_cache(path)
        self.dirty = 0

    @staticmethod
    def key(metric: str, *paths: Path) -> str:
        return metric + "|" + "|".join(f"{p.resolve()}|{p.stat().st_mtime_ns}" for p in paths)

    def get(self, metric, *paths):
        return self.d.get(self.key(metric, *paths))

    def put(self, metric, value, *paths):
        self.d[self.key(metric, *paths)] = value
        self.dirty += 1
        if self.dirty % 25 == 0:
            self.flush()

    def flush(self):
        write_json_atomic(self.path, self.d)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) and its percentile-bootstrap 95 % CI over paired utterances (NaN pairs dropped)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def checkpoint_meta(ckpt: Path) -> dict:
    """epoch / best_epoch / best_val / history of the checkpoint actually on disk (no tensors loaded)."""
    import torch

    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        meta = {k: ck.get(k) for k in ("epoch", "best_epoch", "best_val", "val_loss", "update", "planned_epochs", "exp_name")}
        meta["history"] = ck.get("history", [])
        meta["config"] = ck.get("config", {})
        del ck
        return meta
    except Exception as e:  # noqa: BLE001  (e.g. a raw F5-TTS checkpoint without our meta)
        log.warning(f"could not read training meta from {ckpt.name}: {e}")
        return {}


# --------------------------------------------------------------------------- #
# 1. prompts and generation
# --------------------------------------------------------------------------- #
def build_prompts(rows: list[dict], device: str) -> dict[str, tuple[Path, str]]:
    """cross mode: prompt for utterance i = <= 12 s exact-text cut of utterance i-1 (cyclic)."""
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    aligner = None
    prompts = {}
    for i, row in enumerate(rows):
        src = rows[(i - 1) % len(rows)]
        stem = Path(row["audio_path"]).stem
        wav_p, txt_p = PROMPT_DIR / f"{stem}.wav", PROMPT_DIR / f"{stem}.txt"
        if wav_p.exists() and txt_p.exists():
            prompts[stem] = (wav_p, txt_p.read_text(encoding="utf-8"))
            continue
        audio, sr = common.load_audio(common.abs_path(src["audio_path"]), sr=TARGET_SR)
        if src["duration"] > common.REF_MAX_SEC and aligner is None:
            aligner = common.ForcedAligner(device)
        audio, text, exact = common.cut_reference_prompt(audio, sr, src["text"], aligner)
        if not exact:
            log.warning(f"prompt for {stem}: alignment failed, F5-TTS will ASR-transcribe the prompt")
        common.save_wav(wav_p, audio, sr)
        txt_p.write_text(text, encoding="utf-8")
        prompts[stem] = (wav_p, text)
    if aligner is not None:
        free_gpu(aligner.model, aligner)
    log.info(f"{len(prompts)} cross-clip reference prompts ready in {PROMPT_DIR}")
    return prompts


def build_fixed_prompt(args, device: str, dns: "DNSMOS") -> tuple[Path, str, dict]:
    """
    fixed mode: one enrolment prompt for every utterance.  Default: every *valid*-split
    clip is cut to <= 12 s (exact text via forced alignment) and the cut with the highest
    DNSMOS OVRL is used; the valid split is never synthesised nor trained on, so the prompt
    neither leaks the target nor favours the fine-tuned model with memorised audio.
    """
    FIXED_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    wav_p, txt_p, sel_p = FIXED_PROMPT_DIR / "enrol.wav", FIXED_PROMPT_DIR / "enrol.txt", FIXED_PROMPT_DIR / "enrol_selection.json"
    if args.enrol_wav:
        src = Path(args.enrol_wav)
        text = args.enrol_text if args.enrol_text is not None else (src.with_suffix(".txt").read_text(encoding="utf-8")
                                                                    if src.with_suffix(".txt").exists() else "")
        audio, sr = common.load_audio(src, sr=TARGET_SR)
        if len(audio) / sr > common.REF_MAX_SEC + 0.5 and text:
            audio, text, _ = common.cut_reference_prompt(audio, sr, text, common.ForcedAligner(device))
        common.save_wav(wav_p, audio, sr)
        txt_p.write_text(text, encoding="utf-8")
        sel = {"source": str(src), "duration": len(audio) / sr, "user_supplied": True}
        sel_p.write_text(json.dumps(sel, indent=2), encoding="utf-8")
        return wav_p, text, sel
    if wav_p.exists() and txt_p.exists() and sel_p.exists():
        sel = json.loads(sel_p.read_text(encoding="utf-8"))
        log.info(f"fixed enrolment prompt: {sel['source']} ({sel['duration']:.1f}s, DNSMOS {sel.get('dnsmos_ovrl', float('nan')):.2f})")
        return wav_p, txt_p.read_text(encoding="utf-8"), sel

    rows = common.read_split_csv("valid")
    aligner = None
    cands = []
    tmp = FIXED_PROMPT_DIR / "_candidates"
    tmp.mkdir(exist_ok=True)
    for r in rows:
        audio, sr = common.load_audio(common.abs_path(r["audio_path"]), sr=TARGET_SR)
        if r["duration"] > common.REF_MAX_SEC and aligner is None:
            aligner = common.ForcedAligner(device)
        cut, text, exact = common.cut_reference_prompt(audio, sr, r["text"], aligner, min_sec=4.0)
        if not exact or len(cut) / sr < 4.0:
            continue
        cp = tmp / f"{Path(r['audio_path']).stem}.wav"
        common.save_wav(cp, cut, sr)
        d = dns(cp)
        cands.append({"source": r["audio_path"], "duration": len(cut) / sr, "dnsmos_ovrl": d["ovrl"],
                      "dnsmos_sig": d["sig"], "dnsmos_bak": d["bak"], "text": text, "path": cp})
    if aligner is not None:
        free_gpu(aligner.model, aligner)
    assert cands, "no usable enrolment candidate in the valid split"
    best = max(cands, key=lambda c: c["dnsmos_ovrl"])
    common.save_wav(wav_p, common.load_audio(best["path"], sr=TARGET_SR)[0], TARGET_SR)
    txt_p.write_text(best["text"], encoding="utf-8")
    sel = {k: v for k, v in best.items() if k != "path"}
    sel["candidates"] = [{k: v for k, v in c.items() if k not in ("path", "text")} for c in sorted(cands, key=lambda c: -c["dnsmos_ovrl"])]
    sel["user_supplied"] = False
    sel_p.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    for c in cands:
        c["path"].unlink(missing_ok=True)
    tmp.rmdir()
    log.info(f"fixed enrolment prompt: {best['source']} cut to {best['duration']:.1f}s, DNSMOS OVRL {best['dnsmos_ovrl']:.2f} "
             f"(best of {len(cands)} valid-split candidates) -> {wav_p}")
    return wav_p, best["text"], sel


def n_text_chunks(ref_wav: str, ref_text: str, gen_text: str, speed: float = 1.0) -> int:
    """Number of chunks f5_tts.infer_process will split gen_text into for this prompt."""
    import torchaudio
    from f5_tts.infer.utils_infer import chunk_text

    audio, sr = torchaudio.load(ref_wav)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    return len(chunk_text(gen_text, max_chars=max_chars))


def generate_set(rows, prompts, out_dir: Path, tag: str, args, device: str, model_loader, newer_than: float = 0.0,
                 dec: tuple = DEFAULT_DEC) -> dict:
    """Synthesise every test utterance into out_dir with decoding (cfg, nfe, sway); returns {stem: n_chunks}."""
    cfg_strength, nfe_step, sway = dec
    import soundfile as sf
    import torchaudio
    from f5_tts.infer.utils_infer import infer_batch_process, infer_process, preprocess_ref_audio_text
    from f5_tts.model.utils import seed_everything

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_p = out_dir / "_chunks.json"
    chunks = json.loads(chunks_p.read_text(encoding="utf-8")) if chunks_p.exists() else {}
    todo = [r for r in rows if args.regenerate or Path(r["audio_path"]).stem not in chunks
            or not (out_dir / f"{Path(r['audio_path']).stem}.wav").exists()
            or (out_dir / f"{Path(r['audio_path']).stem}.wav").stat().st_mtime < newer_than]
    log.info(f"[{tag}] {len(rows) - len(todo)} cached, {len(todo)} to synthesise")
    if not todo:
        return chunks
    model, vocoder = model_loader()
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        stem = Path(r["audio_path"]).stem
        ref_wav, ref_text = prompts[stem]
        seed_everything(args.seed)
        ref_wav_p, ref_text_p = preprocess_ref_audio_text(str(ref_wav), ref_text, show_info=lambda *_: None)
        fix_duration = None
        if args.use_truth_duration:  # total = prompt + ground-truth length, as in the F5-TTS paper's evaluation
            fix_duration = sf.info(ref_wav_p).duration + r["duration"]
        n_chunks = n_text_chunks(ref_wav_p, ref_text_p, r["text"], args.speed)
        kw = dict(mel_spec_type="vocos", nfe_step=nfe_step, cfg_strength=cfg_strength,
                  sway_sampling_coef=sway, speed=args.speed, cross_fade_duration=args.cross_fade,
                  fix_duration=fix_duration, device=device, progress=None)
        if args.no_chunk:  # whole text in one pass (single conditioning, no cross-fades), even beyond 4096 frames
            audio, sr = torchaudio.load(ref_wav_p)
            wav, sr, _ = next(infer_batch_process((audio, sr), ref_text_p, [r["text"]], model, vocoder, **kw))
            n_chunks = 1
        else:
            wav, sr, _ = infer_process(ref_wav_p, ref_text_p, r["text"], model, vocoder, show_info=lambda *_: None, **kw)
        wav = np.asarray(wav, dtype=np.float32)
        common.assert_audio_ok(wav, f"{tag}/{stem}", min_sec=0.5)
        common.save_wav(out_dir / f"{stem}.wav", wav, sr)
        chunks[stem] = n_chunks
        chunks_p.write_text(json.dumps(chunks, indent=1), encoding="utf-8")
        log.info(f"  [{tag} {k}/{len(todo)}] {stem}: {len(wav) / sr:.1f}s generated (gt {r['duration']:.1f}s, "
                 f"{n_chunks} chunk{'s' if n_chunks != 1 else ''}) | {(time.time() - t0) / k:.1f}s/utt")
    free_gpu(model, vocoder)
    return chunks


def copy_synthesis(rows, device: str, regenerate: bool) -> None:
    """Real test audio -> vocos mel -> vocos decoder -> audio: what the mel/vocoder path alone costs."""
    import torch
    from f5_tts.infer.utils_infer import load_vocoder
    from f5_tts.model.modules import MelSpec
    from finetune_tts import MEL_KWARGS

    COPYSYNTH_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if regenerate or not (COPYSYNTH_DIR / f"{Path(r['audio_path']).stem}.wav").exists()]
    if not todo:
        log.info(f"[copysynth] {len(rows)} cached")
        return
    mel_spec = MelSpec(**MEL_KWARGS).to(device)
    vocoder = load_vocoder("vocos", is_local=False, device=device)
    for r in todo:
        stem = Path(r["audio_path"]).stem
        audio, sr = common.load_audio(common.abs_path(r["audio_path"]), sr=TARGET_SR)
        with torch.inference_mode():
            mel = mel_spec(torch.from_numpy(audio)[None].to(device))  # [1, 100, T]
            wav = vocoder.decode(mel).squeeze().float().cpu().numpy()
        wav = wav[: len(audio)] if len(wav) >= len(audio) else np.pad(wav, (0, len(audio) - len(wav)))
        rms_in, rms_out = np.sqrt(np.mean(audio**2)) + 1e-9, np.sqrt(np.mean(wav**2)) + 1e-9
        common.save_wav(COPYSYNTH_DIR / f"{stem}.wav", wav * rms_in / rms_out, TARGET_SR)
    log.info(f"[copysynth] {len(todo)} utterances re-synthesised through mel -> vocos -> {COPYSYNTH_DIR}")
    free_gpu(vocoder, mel_spec)


# --------------------------------------------------------------------------- #
# 2. metric backends
# --------------------------------------------------------------------------- #
class Whisper:
    def __init__(self, name: str, device: str):
        import whisper

        log.info(f"loading Whisper '{name}' ...")
        self.model = whisper.load_model(name, device=device)
        self.fp16 = device == "cuda"
        self.cache = load_json_cache(TRANSCRIPT_CACHE)
        self.name = name

    def __call__(self, path: Path) -> str:
        key = f"{self.name}|{path.resolve()}|{path.stat().st_mtime_ns}"
        if key not in self.cache:
            audio, _ = common.load_audio(path, sr=16_000)
            res = self.model.transcribe(audio, language="en", fp16=self.fp16, temperature=0.0)
            self.cache[key] = res["text"].strip()
            write_json_atomic(TRANSCRIPT_CACHE, self.cache, indent=1)
        return self.cache[key]


class SpeakerEncoder:
    """resemblyzer GE2E - secondary speaker metric (SEC), kept for continuity with earlier reports."""

    def __init__(self, device: str):
        from resemblyzer import VoiceEncoder

        self.enc = VoiceEncoder(device=device, verbose=False)
        self._cache: dict[Path, np.ndarray] = {}

    def embed(self, path: Path) -> np.ndarray:
        from resemblyzer import preprocess_wav

        if path not in self._cache:
            self._cache[path] = self.enc.embed_utterance(preprocess_wav(str(path)))
        return self._cache[path]

    def cosine(self, a: Path, b: Path) -> float:
        ea, eb = self.embed(a), self.embed(b)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


class SimO:
    """
    SIM-o: WavLM-large fine-tuned for speaker verification + ECAPA-TDNN head (UniSpeech), the
    speaker-similarity model of the Seed-TTS / F5-TTS evaluations.  f5_tts ships the model code;
    the checkpoint is fetched through cached_path (~1.2 GB) and WavLM-large through s3prl (~1.2 GB).
    """

    def __init__(self, device: str):
        import torch
        import torch.nn.functional as F
        from cached_path import cached_path
        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

        log.info("loading SIM-o (WavLM-large + ECAPA-TDNN speaker verification) ...")
        self.torch, self.F = torch, F
        self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        sd = torch.load(str(cached_path(SIMO_CKPT)), map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd["model"], strict=False)
        assert not missing, f"SIM-o checkpoint missing keys: {missing[:5]}"
        assert set(unexpected) <= {"loss_calculator.projection.weight"}, f"unexpected keys: {unexpected[:5]}"
        self.model = self.model.to(device).eval()
        self.device = device
        self._cache: dict[Path, "torch.Tensor"] = {}

    def embed(self, path: Path):
        import torchaudio

        if path not in self._cache:
            w, sr = torchaudio.load(str(path))
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            if sr != 16_000:
                w = torchaudio.functional.resample(w, sr, 16_000)
            with self.torch.inference_mode():
                self._cache[path] = self.model(w.to(self.device)).float().cpu()
        return self._cache[path]

    def cosine(self, a: Path, b: Path) -> float:
        return float(self.F.cosine_similarity(self.embed(a), self.embed(b)).item())


def smos_from_cosine(cos: float, lo: float = 0.50, hi: float = 0.95) -> float:
    """Map resemblyzer cosine to a 1-5 similarity-MOS scale (0.5 ~ different speaker, 0.95 ~ same take)."""
    return float(1.0 + 4.0 * np.clip((cos - lo) / (hi - lo), 0.0, 1.0))


class UTMOS:
    def __init__(self, device: str):
        import torch

        log.info("loading UTMOS22-strong (torch.hub tarepan/SpeechMOS) ...")
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        self.device = device
        self.torch = torch

    def __call__(self, path: Path) -> float:
        y, sr = common.load_audio(path, sr=16_000)
        with self.torch.inference_mode():
            return float(self.model(self.torch.from_numpy(y)[None].to(self.device), sr).item())


class DNSMOS:
    """DNSMOS P.835 (non-personalised): returns OVRL MOS (also SIG/BAK) per clip."""

    SR, SEG_SAMPLES = 16_000, 144_160  # 9.01 s windows, the model's fixed input size

    def __init__(self):
        import onnxruntime as ort

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        onnx_p = MODELS_DIR / "sig_bak_ovr.onnx"
        if not onnx_p.exists():
            log.info(f"downloading DNSMOS model -> {onnx_p}")
            urllib.request.urlretrieve(DNSMOS_URL, onnx_p)
        self.sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        self.p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self.p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self.p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def __call__(self, path: Path) -> dict:
        y, _ = common.load_audio(path, sr=self.SR)
        n = self.SEG_SAMPLES
        if len(y) < n:
            y = np.tile(y, int(np.ceil(n / len(y))))[:n]
        sig, bak, ovr = [], [], []
        for start in range(0, len(y) - n + 1, self.SR):  # 9.01 s windows, 1 s hop
            seg = y[start : start + n].astype(np.float32)[None]
            s, b, o = self.sess.run(None, {"input_1": seg})[0][0]
            sig.append(self.p_sig(s)), bak.append(self.p_bak(b)), ovr.append(self.p_ovr(o))
        return {"ovrl": float(np.mean(ovr)), "sig": float(np.mean(sig)), "bak": float(np.mean(bak))}


class FADEmbedder:
    """wav2vec2-base (torchaudio) layer-6 features pooled over 1 s windows -> [n, 768]."""

    def __init__(self, device: str):
        import torchaudio

        log.info("loading wav2vec2-base for FAD embeddings ...")
        self.model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(device).eval()
        self.device = device

    def __call__(self, path: Path) -> np.ndarray:
        import torch

        y, _ = common.load_audio(path, sr=16_000)
        with torch.inference_mode():
            feats, _ = self.model.extract_features(torch.from_numpy(y)[None].to(self.device), num_layers=6)
        f = feats[-1][0].float().cpu().numpy()  # [frames(20 ms), 768]
        win = 50
        n = max(1, len(f) // win)
        return np.stack([f[i * win : (i + 1) * win].mean(0) for i in range(n)])


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eps = 1e-6 * np.eye(len(mu1))
    covmean = np.real(linalg.sqrtm((s1 + eps) @ (s2 + eps)))
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))


def voicing_decision_error(ref: Path, gen: Path, sr: int = 16_000, hop: int = 160) -> float:
    """Voicing flags (pyin) compared frame-by-frame along an MFCC DTW alignment."""
    import librosa

    def analyse(p):
        y, _ = common.load_audio(p, sr=sr)
        _, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=hop)
        n = min(len(vflag), mf.shape[1])
        return np.nan_to_num(vflag[:n]).astype(bool), mf[:, :n]

    vr, mr = analyse(ref)
    vg, mg = analyse(gen)
    _, wp = librosa.sequence.dtw(X=mr, Y=mg, metric="cosine")
    return float(np.mean([vr[i] != vg[j] for i, j in wp]))


def idiosyncrasy_analysis(gt_text: str, heard_gt: str, heard_gen: str) -> dict:
    """
    Words in the written transcript that Whisper hears *differently* on the real
    audio mark the speaker's non-standard pronunciations (the transcript spells
    them the way he says them, e.g. "smatch").  For each such word we check what
    Whisper hears on the clone:
      retained   - same as on the real audio (pronunciation reproduced)
      normalised - the dictionary form (clone "corrected" the accent)
      other      - something else
    """
    import jiwer

    ref = normalize_words(gt_text).split()
    hg = normalize_words(heard_gt).split()
    hx = normalize_words(heard_gen).split()
    if not ref or not hg or not hx:
        return {"n_idio": 0, "retained": 0, "normalised": 0, "other": 0, "examples": []}

    def heard_map(hyp):
        out = {}
        for ch in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
            if ch.type in ("equal", "substitute"):
                for k, ri in enumerate(range(ch.ref_start_idx, ch.ref_end_idx)):
                    hi = ch.hyp_start_idx + k
                    out[ri] = hyp[hi] if hi < ch.hyp_end_idx else ""
            elif ch.type == "delete":
                for ri in range(ch.ref_start_idx, ch.ref_end_idx):
                    out[ri] = ""
        return out

    m_gt, m_gen = heard_map(hg), heard_map(hx)
    idio = [i for i in range(len(ref)) if m_gt.get(i, "") != ref[i] and m_gt.get(i, "")]
    retained = normalised = other = 0
    examples = []
    for i in idio:
        g = m_gen.get(i, "")
        if g == m_gt[i]:
            retained += 1
            tag = "retained"
        elif g == ref[i]:
            normalised += 1
            tag = "normalised"
        else:
            other += 1
            tag = "other"
        if len(examples) < 4:
            examples.append({"written": ref[i], "heard_real": m_gt[i], "heard_clone": g, "outcome": tag})
    return {"n_idio": len(idio), "retained": retained, "normalised": normalised, "other": other, "examples": examples}




# --------------------------------------------------------------------------- #
# 2b. PHONE-LEVEL PRONUNCIATION RETENTION
#
# Merged in from the former scripts/phone_metrics.py + phone_retention.py +
# phone_lexicon.py.  This is the *primary* retention measurement; the ASR-proxy
# idiosyncrasy_analysis() above is kept as a secondary column for continuity with
# earlier reports.
#
# Phone-level pronunciation metrics for the aktts pipeline.
#
# The original retention metric was an ASR proxy: a word counted as an
# "idiosyncratic pronunciation marker" when Whisper mis-heard it on the real
# recording, and as "retained" when Whisper mis-heard it the same way on the
# clone.  That construct is wrong in three ways - it fires on Whisper's lexical
# priors (`suleiman` -> `sullivan`), on morphology (`ranking` -> `ranked`) and on
# homophones (`piece` -> `peace`, zero phonetic information) - and it yields only
# ~49 markers over the 21 test utterances, which is why its CIs are +-20 pp.
#
# This module measures the thing itself, at the phone level:
#
#   * word spans come from forced alignment against the **verbatim** transcript
#     (torchaudio MMS_FA, via common.ForcedAligner), so word k of the real
#     recording is compared with word k of the generated audio;
#   * the realised phone sequence inside each span comes from a CTC phone
#     recogniser (wav2vec2-lv-60-espeak-cv-ft, eSpeak IPA inventory), decoded
#     straight from the CTC vocabulary - no phonemizer/espeak backend needed;
#   * words are compared by phone edit distance (PER).
#
# **Marker definition.**  A word is an idiosyncratic-pronunciation marker when the
# speaker's realisation on the real recording differs from the zero-shot
# `F5TTS_Base` realisation of the same word in the same sentence by at least
# `marker_per` phone edits per phone.  The pretrained model *is* the
# standard-pronunciation prior this project is trying not to regress to, and it is
# measured through the same acoustic recogniser as everything else, so no
# pronunciation dictionary and no ARPAbet<->IPA mapping sits in the loop.  Words
# where speaker and prior agree carry no signal and are excluded by construction -
# which also removes homophones, inflection and tokenisation artefacts, since
# those produce identical or near-identical phone strings.
#
# **Score.**  For each marker, the condition's realisation C is compared with the
# speaker's R and the prior's B:
#
#     d_R = PER(C, R)        d_B = PER(C, B)
#     retained  <=>  d_R < d_B
#     margin    =  (d_B - d_R) / (d_B + d_R)   in [-1, 1]
#
# **Primary statistic: SPD = mean d_R over markers** (speaker phone distance; lower
# is closer to how the speaker actually says the word).  It never references the
# prior, so a condition cannot improve it merely by drifting away from the
# pretrained model - `retained` and `margin` both can, which is why a sway=0
# decode looked twice as good on `retained %` while barely moving SPD.  Anchors:
# zero-shot baseline ~0.87 (floor), copy-synthesis ~0.24 (ceiling; 90 % of markers
# "retained"), so the normalised scale 100*(0.877-SPD)/0.638 reads as "% of the
# way from the prior to the speaker".  `retained %` is kept for continuity with
# the old metric's units and `margin` as a continuous variant.
#
# Measured noise (two fresh trainings of the same config, 3 synthesis seeds each,
# 21 test utterances): within-checkpoint seed sd 0.003-0.022 SPD, between-run
# seed-pooled difference 0.008-0.012 SPD (1-3 pp normalised) with paired CIs
# containing zero.  Report SPD pooled over >= 3 seeds; single-seed pairs can differ
# by up to ~0.05 SPD (8 pp) on the same checkpoint.
#
# Edit distance rather than posterior divergence: the two recordings have
# different timing, so frame posteriors would need a DTW alignment before they
# could be compared, adding a second error source; a collapsed phone sequence is
# alignment-free, interpretable per word, and robust to the 20 ms frame grid.
# --------------------------------------------------------------------------- #
PHONE_CACHE = EVAL_GEN_DIR / "phone_cache.json"
LEXICON_JSON = CHECKPOINT_DIR / "speaker_lexicon.json"

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
PHONE_SR = 16_000


# --------------------------------------------------------------------------- #
# phone recognition
# --------------------------------------------------------------------------- #
@dataclass
class PhoneToken:
    phone: str
    start: float
    end: float


class PhoneRecognizer:
    """
    CTC phone recogniser emitting eSpeak-IPA phones with frame timings.

    The HF tokenizer for this checkpoint imports `phonemizer` at construction
    because it also supports text -> phone encoding; only id -> phone decoding is
    needed here, so the vocabulary is read directly and the tokenizer skipped.
    """

    def __init__(self, device: str = "cuda", model_id: str = PHONE_MODEL):
        import torch
        from cached_path import cached_path
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        vocab = json.loads(Path(str(cached_path(f"hf://{model_id}/vocab.json"))).read_text(encoding="utf-8"))
        self.id2p = {v: k for k, v in vocab.items()}
        self.blank = vocab.get("<pad>", 0)
        self.skip = {vocab[t] for t in ("<pad>", "<s>", "</s>", "<unk>", "|") if t in vocab}
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval().to(device)
        self.device = device

    def __call__(self, path: Path | str) -> list[PhoneToken]:
        """CTC-collapsed phone tokens with (start, end) seconds."""
        import torchaudio

        w, sr = torchaudio.load(str(path))
        w = w.mean(0)
        if sr != PHONE_SR:
            w = torchaudio.functional.resample(w, sr, PHONE_SR)
        dur = len(w) / PHONE_SR
        feats = self.fe(w.numpy(), sampling_rate=PHONE_SR, return_tensors="pt").input_values.to(self.device)
        with self.torch.inference_mode():
            ids = self.model(feats).logits[0].argmax(-1).cpu().tolist()
        if not ids:
            return []
        step = dur / len(ids)
        out: list[PhoneToken] = []
        prev = None
        for i, tok in enumerate(ids):
            if tok != prev and tok not in self.skip:
                out.append(PhoneToken(self.id2p[tok], i * step, (i + 1) * step))
            elif out and tok == prev and tok not in self.skip:
                out[-1].end = (i + 1) * step
            prev = tok
        return out

    def release(self):
        import gc

        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def phones_in_span(tokens: list[PhoneToken], t0: float, t1: float, pad: float = 0.02) -> list[str]:
    """Phones whose centre falls inside [t0, t1] (a small pad absorbs alignment jitter)."""
    return [t.phone for t in tokens if t0 - pad <= (t.start + t.end) / 2 <= t1 + pad]


# --------------------------------------------------------------------------- #
# per-word realisations
# --------------------------------------------------------------------------- #
def word_realisations(path: Path | str, text: str, aligner, recognizer: PhoneRecognizer,
                      min_phones: int = 2) -> dict[int, dict]:
    """
    {word_index: {word, start, end, phones}} for one audio file.

    Alignment is against the verbatim transcript, so indices are shared between
    the real recording and every generated version of the same sentence.
    Spelling is never normalised; ForcedAligner._norm_word only lower-cases for
    the aligner's own romanised-character inventory.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    try:
        spans = aligner.align(audio, sr, text)
    except Exception as e:  # noqa: BLE001  (degenerate audio, or CUDA OOM on a long clip)
        log.warning(f"forced alignment failed for {Path(path).name}: {type(e).__name__}: {e}")
        return {}
    toks = recognizer(path)
    out = {}
    for i, sp in enumerate(spans):
        ph = phones_in_span(toks, sp["start"], sp["end"])
        if len(ph) >= min_phones:
            out[i] = {"word": sp["word"], "start": sp["start"], "end": sp["end"], "phones": ph}
    return out


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def per(a: list[str], b: list[str]) -> float:
    """Phone error rate between two realisations, normalised by the longer sequence."""
    n = max(len(a), len(b))
    return edit_distance(a, b) / n if n else 0.0


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
@dataclass
class MarkerScore:
    stem: str
    idx: int
    word: str
    real: list[str]
    base: list[str]
    cond: list[str]
    d_real: float
    d_base: float
    marker_per: float

    @property
    def retained(self) -> bool:
        return self.d_real < self.d_base

    @property
    def margin(self) -> float:
        s = self.d_real + self.d_base
        return (self.d_base - self.d_real) / s if s else 0.0

    def as_dict(self) -> dict:
        return {"stem": self.stem, "idx": self.idx, "word": self.word,
                "real": " ".join(self.real), "base": " ".join(self.base), "cond": " ".join(self.cond),
                "d_real": round(self.d_real, 4), "d_base": round(self.d_base, 4),
                "marker_per": round(self.marker_per, 4), "retained": self.retained,
                "margin": round(self.margin, 4)}


@dataclass
class RetentionResult:
    markers: list[MarkerScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.markers)

    @property
    def retained_pct(self) -> float:
        return 100 * sum(m.retained for m in self.markers) / self.n if self.n else float("nan")

    @property
    def mean_margin(self) -> float:
        return float(np.mean([m.margin for m in self.markers])) if self.n else float("nan")

    def by_utterance(self) -> dict[str, list[MarkerScore]]:
        out: dict[str, list[MarkerScore]] = {}
        for m in self.markers:
            out.setdefault(m.stem, []).append(m)
        return out


def find_markers(real: dict[int, dict], base: dict[int, dict], marker_per: float = 0.34,
                 min_phones: int = 2) -> dict[int, float]:
    """
    Word indices where the speaker and the zero-shot prior genuinely differ, with
    the size of that difference.  `marker_per` of 0.34 means at least roughly one
    phone edit in three - enough to exclude recogniser jitter on a short word
    while keeping real segmental differences.
    """
    out = {}
    for i, r in real.items():
        b = base.get(i)
        if not b or len(r["phones"]) < min_phones or len(b["phones"]) < min_phones:
            continue
        d = per(r["phones"], b["phones"])
        if d >= marker_per:
            out[i] = d
    return out


def score_condition(stem: str, markers: dict[int, float], real: dict[int, dict], base: dict[int, dict],
                    cond: dict[int, dict]) -> list[MarkerScore]:
    out = []
    for i, mper in markers.items():
        c = cond.get(i)
        if not c:
            continue
        out.append(MarkerScore(stem=stem, idx=i, word=real[i]["word"], real=real[i]["phones"],
                               base=base[i]["phones"], cond=c["phones"],
                               d_real=per(c["phones"], real[i]["phones"]),
                               d_base=per(c["phones"], base[i]["phones"]), marker_per=mper))
    return out


def score_condition_pooled(stem: str, markers: dict[int, float], real: dict[int, dict], base: dict[int, dict],
                           cond_seeds: list[dict[int, dict]]) -> list[MarkerScore]:
    """
    Like score_condition, but averages d_real/d_base over several synthesis seeds of the same
    condition before scoring.  Measured seed noise (same checkpoint, same decoding) is 0.003-0.022
    SPD and can occasionally reach paired significance on its own (see the section-2b module
    docstring), so every Phase-2+ comparison pools >= 3 seeds rather than reading a single one.
    real/base (the ground-truth recording and the zero-shot prior) do not vary with seed - only
    `cond` does - so only the condition side needs averaging.
    """
    out = []
    for i, mper in markers.items():
        per_seed = [c[i]["phones"] for c in cond_seeds if i in c]
        if not per_seed:
            continue
        dr = float(np.mean([per(ph, real[i]["phones"]) for ph in per_seed]))
        db = float(np.mean([per(ph, base[i]["phones"]) for ph in per_seed]))
        out.append(MarkerScore(stem=stem, idx=i, word=real[i]["word"], real=real[i]["phones"],
                               base=base[i]["phones"], cond=per_seed[0],
                               d_real=dr, d_base=db, marker_per=mper))
    return out


def bootstrap_by_utterance(markers: list[MarkerScore], stat: str = "retained", n_boot: int = 10000,
                           seed: int = 0) -> tuple[float, float, float]:
    """
    Percentile bootstrap resampling **utterances**, not markers: markers within an
    utterance are correlated (same prompt, same synthesis run), so resampling
    markers directly would understate the CI.
    """
    if not markers:
        return float("nan"), float("nan"), float("nan")
    groups = list(RetentionResult(markers).by_utterance().values())
    if stat == "retained":
        num = np.array([sum(m.retained for m in g) for g in groups], float)
        den = np.array([len(g) for g in groups], float)
        point = 100 * num.sum() / den.sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        b = 100 * num[idx].sum(1) / den[idx].sum(1)
    elif stat == "d_real":
        # distance to the speaker's own realisation.  Unlike `retained` / `margin`, this never
        # references the prior, so it cannot be inflated by a condition simply drifting away from
        # the prior - the confound that makes sway=0 look better than it is.
        num = np.array([sum(m.d_real for m in g) for g in groups], float)
        den = np.array([len(g) for g in groups], float)
        point = num.sum() / den.sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        b = num[idx].sum(1) / den[idx].sum(1)
    else:
        vals = [np.array([m.margin for m in g], float) for g in groups]
        num = np.array([v.sum() for v in vals])
        den = np.array([len(v) for v in vals], float)
        point = num.sum() / den.sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        b = num[idx].sum(1) / den[idx].sum(1)
    return float(point), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_bootstrap_by_utterance(a: list[MarkerScore], b: list[MarkerScore], stat: str = "retained",
                                  n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired delta (a - b) over the shared markers, resampling utterances."""
    ka = {(m.stem, m.idx): m for m in a}
    kb = {(m.stem, m.idx): m for m in b}
    shared = sorted(set(ka) & set(kb))
    if not shared:
        return float("nan"), float("nan"), float("nan")
    stems = sorted({s for s, _ in shared})
    si = {s: i for i, s in enumerate(stems)}
    na, nb, den = np.zeros(len(stems)), np.zeros(len(stems)), np.zeros(len(stems))
    for k in shared:
        i = si[k[0]]
        if stat == "retained":
            na[i] += ka[k].retained
            nb[i] += kb[k].retained
        elif stat == "d_real":
            na[i] += ka[k].d_real
            nb[i] += kb[k].d_real
        else:
            na[i] += ka[k].margin
            nb[i] += kb[k].margin
        den[i] += 1
    scale = 100 if stat == "retained" else 1  # d_real / margin are reported in PER units
    point = scale * (na.sum() - nb.sum()) / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stems), size=(n_boot, len(stems)))
    boot = scale * (na[idx].sum(1) - nb[idx].sum(1)) / den[idx].sum(1)
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


class Realisations:
    """word_realisations() with a persistent (path, mtime) cache."""

    def __init__(self, aligner, recognizer):
        self.aligner, self.rec = aligner, recognizer
        self.d = {}
        if PHONE_CACHE.exists():
            try:
                self.d = json.loads(PHONE_CACHE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                log.warning(f"{PHONE_CACHE.name} unreadable; starting fresh")
        self.new = 0

    def get(self, path: Path, text: str) -> dict[int, dict]:
        key = f"{path.resolve()}|{path.stat().st_mtime_ns}"
        if key not in self.d:
            r = word_realisations(path, text, self.aligner, self.rec)
            if not r:
                # word_realisations() returns {} when forced alignment fails, which can be
                # transient (GPU pressure, a released recogniser mid-run).  Caching that would
                # turn a transient failure into a permanent wrong answer - it once silently cut
                # a condition's marker count from ~200 to 36 - so empty results are never cached.
                log.warning(f"no word realisations for {path.name} (alignment failed); not caching")
                return {}
            self.d[key] = {str(i): v for i, v in r.items()}
            self.new += 1
            if self.new % 20 == 0:
                self.flush()
        return {int(i): v for i, v in self.d[key].items()}

    def flush(self):
        PHONE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PHONE_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d), encoding="utf-8")
        tmp.replace(PHONE_CACHE)


def norm_word(w: str) -> str:
    """Lexicon key: lower-cased, punctuation stripped.  Spelling itself is never altered."""
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


def build_speaker_lexicon(R: "Realisations", splits: str = "train,valid,test", out: Path = None) -> dict:
    """
    word -> {n, mode, mode_frac, realisations} over every real recording in the dataset.

    Markers are discovered from ~270 train+valid+test clips rather than from the 21 clips being
    scored, which is what lifts the marker base from the ASR-proxy's 49 to ~210 per prompt mode.
    Honest limitation: the CTC recogniser's per-token output varies, so for polysyllabic words the
    modal string is a weak summary (median modal share ~0.33) - the lexicon works as a marker
    *identity* filter (--phone_lexicon), not as a noise reducer.
    """
    lex: dict[str, Counter] = defaultdict(Counter)
    tokens_by_split, n_files = Counter(), 0
    for split in splits.split(","):
        for r in common.read_split_csv(split):
            for v in R.get(common.abs_path(r["audio_path"]), r["text"]).values():
                lex[norm_word(v["word"])][" ".join(v["phones"])] += 1
                tokens_by_split[split] += 1
            n_files += 1
            if n_files % 25 == 0:
                R.flush()
    R.flush()
    entries = {}
    for w, c in lex.items():
        total = sum(c.values())
        mode, mode_n = c.most_common(1)[0]
        entries[w] = {"n": total, "mode": mode, "mode_frac": mode_n / total,
                      "realisations": dict(c.most_common(8))}
    doc = {"splits": splits, "n_files": n_files, "tokens": dict(tokens_by_split), "lexicon": entries}
    log.info(f"speaker lexicon: {len(entries)} word types from {sum(tokens_by_split.values())} tokens "
             f"in {n_files} clips; {sum(1 for v in entries.values() if v['n'] >= 3)} types with >= 3 tokens")
    if out:
        write_json_atomic(out, doc, indent=1)
    return doc


def phone_retention_analysis(rows: list[dict], conds: list[str], prompt: str, device: str, args,
                             reference: str = "baseline", delta_ref: str | None = None,
                             aligner=None, rec=None) -> dict:
    """
    Phone-level retention for every condition, for one prompt mode.

    Returns {"n_markers", "n_aligned_words", "conditions": {cond: {spd, retained_pct, CIs, paired
    deltas...}}, "per_marker": [...], "markers_by_utterance": {...}} - the same structure the
    former phone_retention.py wrote to JSON, so archived outputs stay comparable.
    """
    stems = [Path(r["audio_path"]).stem for r in rows]
    text = {s: r["text"] for s, r in zip(stems, rows)}
    real_path = {s: common.abs_path(r["audio_path"]) for s, r in zip(stems, rows)}
    n_seeds = max(getattr(args, "phone_seeds", 1), 1)

    # One aligner + recogniser can serve every prompt mode; when the caller supplies them we must
    # not release them here.  Free whatever the earlier metric backends left on the GPU first - the
    # aligner OOMs on the longer clips otherwise, and that surfaced as 13 of 21 files silently
    # losing their word spans.
    owned = aligner is None or rec is None
    if owned:
        free_gpu()
        aligner = aligner or common.ForcedAligner(device)
        rec = rec or PhoneRecognizer(device)
    R = Realisations(aligner, rec)
    try:
        if getattr(args, "build_lexicon", False):
            build_speaker_lexicon(R, out=LEXICON_JSON)

        real = {s: R.get(real_path[s], text[s]) for s in stems}
        base = {s: R.get(EVAL_GEN_DIR / reference / prompt / f"{s}.wav", text[s]) for s in stems}

        markers = {s: find_markers(real[s], base[s], args.marker_per, args.phone_min_phones) for s in stems}
        n_raw = sum(len(m) for m in markers.values())
        n_words = sum(len(r) for r in real.values())

        if getattr(args, "phone_lexicon", False) and LEXICON_JSON.exists():
            lex = json.loads(LEXICON_JSON.read_text(encoding="utf-8"))["lexicon"]
            for s in stems:
                for i in list(markers[s]):
                    e = lex.get(norm_word(real[s][i]["word"]))
                    if not e or e["n"] < 3:
                        continue
                    mode = e["mode"].split()
                    if per(real[s][i]["phones"], mode) > 0.5:
                        del markers[s][i]   # this token disagrees with the speaker's usual realisation
                        continue
                    real[s][i] = dict(real[s][i], phones=mode)

        # one marker per word type per utterance; repeated tokens fold into the first
        dedup_groups: dict[tuple, list[int]] = {}
        for s in stems:
            seen: dict[str, int] = {}
            for i in sorted(markers[s]):
                w = real[s][i]["word"].lower()
                if w in seen:
                    markers[s].pop(i)
                    dedup_groups.setdefault((s, seen[w]), []).append(i)
                else:
                    seen[w] = i
        n_markers = sum(len(m) for m in markers.values())
        log.info(f"[phone/{prompt}] markers: {n_markers} of {n_words} aligned words "
                 f"({100 * n_markers / max(n_words, 1):.1f} %) across {sum(1 for s in stems if markers[s])} "
                 f"utterances (dedup: {n_raw} tokens -> {n_markers} word types) | seeds pooled: {n_seeds}")

        results, per_marker, keep = {}, [], {}
        for cond in conds:
            seed_range = range(1) if cond in ("copysynth", reference) else range(n_seeds)
            scored: list[MarkerScore] = []
            for s in stems:
                if not markers[s]:
                    continue
                paths = []
                for sd in seed_range:
                    tag = cond if sd == 0 else f"{cond}_seed{sd}"
                    p = (COPYSYNTH_DIR / f"{s}.wav") if cond == "copysynth" else EVAL_GEN_DIR / tag / prompt / f"{s}.wav"
                    if p.exists():
                        paths.append(p)
                cond_seeds = [c for c in (R.get(p, text[s]) for p in paths) if c]
                if not cond_seeds:
                    continue
                sc = score_condition_pooled(s, markers[s], real[s], base[s], cond_seeds)
                for m in sc:
                    extra = dedup_groups.get((s, m.idx), [])
                    if extra:
                        dr = [m.d_real] + [per(cs[j]["phones"], real[s][m.idx]["phones"])
                                           for cs in cond_seeds for j in extra if j in cs]
                        db = [m.d_base] + [per(cs[j]["phones"], base[s][j]["phones"])
                                           for cs in cond_seeds for j in extra if j in cs and j in base[s]]
                        m.d_real, m.d_base = float(np.mean(dr)), float(np.mean(db))
                scored += sc
            if not scored:
                continue
            ret, rlo, rhi = bootstrap_by_utterance(scored, "retained")
            spd, slo, shi = bootstrap_by_utterance(scored, "d_real")
            mar, mlo, mhi = bootstrap_by_utterance(scored, "margin")
            results[cond] = {"n_markers": len(scored), "spd": spd, "spd_lo": slo, "spd_hi": shi,
                             "retained_pct": ret, "retained_lo": rlo, "retained_hi": rhi,
                             "margin": mar, "margin_lo": mlo, "margin_hi": mhi}
            per_marker += [dict(m.as_dict(), cond=cond) for m in scored]
            keep[cond] = scored
            log.info(f"  [phone/{prompt}] {cond:34s} SPD {spd:.3f} [{slo:.3f}, {shi:.3f}]  "
                     f"retained {ret:5.1f} % [{rlo:5.1f}, {rhi:5.1f}]  n={len(scored)}")

        ref_c = delta_ref if delta_ref in keep else ("finetuned" if "finetuned" in keep else (conds[0] if conds else None))
        for cond in keep:
            if cond == ref_c or ref_c is None:
                continue
            for stat, pfx in (("retained", "d_retained"), ("d_real", "d_spd"), ("margin", "d_margin")):
                d, lo, hi = paired_bootstrap_by_utterance(keep[cond], keep[ref_c], stat)
                results[cond].update({pfx: d, f"{pfx}_lo": lo, f"{pfx}_hi": hi})

        R.flush()
        return {"prompt": prompt, "reference": reference, "delta_reference": ref_c,
                "marker_per": args.marker_per, "seeds_pooled": n_seeds,
                "n_markers": n_markers, "n_aligned_words": n_words, "conditions": results,
                "per_marker": per_marker,
                "markers_by_utterance": {s: {str(i): round(d, 4) for i, d in m.items()} for s, m in markers.items()}}
    finally:
        if owned:
            rec.release()
            free_gpu(aligner)


# --------------------------------------------------------------------------- #
# 3. aggregation, plots & report
# --------------------------------------------------------------------------- #
def retention_bootstrap(ret_a: np.ndarray, n_a: np.ndarray, ret_b: np.ndarray | None = None,
                        n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """
    Retention = sum(retained) / sum(markers) over utterances (markers come from the real audio, so they
    are identical across conditions).  Bootstrap over utterances; with ret_b it is the paired delta.
    """
    ret_a, n_a = np.asarray(ret_a, float), np.asarray(n_a, float)
    keep = n_a > 0
    ret_a, n_a = ret_a[keep], n_a[keep]
    ret_b = None if ret_b is None else np.asarray(ret_b, float)[keep]
    if n_a.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    def stat(idx):
        v = 100 * ret_a[idx].sum(1) / n_a[idx].sum(1)
        return v if ret_b is None else v - 100 * ret_b[idx].sum(1) / n_a[idx].sum(1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n_a), size=(n_boot, len(n_a)))
    point = 100 * ret_a.sum() / n_a.sum() - (0 if ret_b is None else 100 * ret_b.sum() / n_a.sum())
    b = stat(idx)
    return float(point), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def condition_summary(df: pd.DataFrame, fad: dict, retention: dict, ref_cond: str = "baseline") -> pd.DataFrame:
    """One row per (condition, prompt): metric means (+ CIs for retention) and paired-bootstrap deltas vs ref_cond."""
    rows = []
    for (cond, prompt), g in df.groupby(["cond", "prompt"], sort=False):
        row = {"cond": cond, "prompt": prompt, "alpha": g["alpha"].iloc[0], "n": len(g)}
        for k in ("cfg", "nfe", "sway"):
            row[k] = g[k].iloc[0] if k in g else float("nan")
        for m in METRICS:
            if m == "fad":
                row["fad"] = fad.get((cond, prompt), float("nan"))
            elif m in g:
                row[m] = float(g[m].astype(float).mean())
        ret = retention.get((cond, prompt))
        row["normalised"] = 100 * ret["normalised"] / ret["n_idio"] if ret and ret["n_idio"] else float("nan")
        if "idio_words" in g and g["idio_words"].notna().any():
            row["retention"], row["retention_lo"], row["retention_hi"] = retention_bootstrap(g["idio_retained"].values, g["idio_words"].values)
            row["n_markers"] = int(g["idio_words"].sum())
        else:
            row["retention"] = row["retention_lo"] = row["retention_hi"] = float("nan")
        base = df[(df["cond"] == ref_cond) & (df["prompt"] == prompt)].set_index("stem")
        if cond not in (ref_cond, "gt", "copysynth") and len(base):
            gi = g.set_index("stem").reindex(base.index)
            for m in DELTA_METRICS:
                if m in gi and m in base:
                    d, lo, hi = paired_bootstrap(gi[m].values, base[m].values)
                    row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = d, lo, hi
            if "idio_words" in gi and "idio_retained" in base:
                row["d_retention"], row["d_retention_lo"], row["d_retention_hi"] = retention_bootstrap(
                    gi["idio_retained"].values, gi["idio_words"].values, base["idio_retained"].values)
        rows.append(row)
    return pd.DataFrame(rows)


# the project objective: retention up, WER down, speaker identity not given back.  NMOS / DNSMOS are *not*
# axes: copy-synthesis shows the recordings themselves bound them, and the baseline scores higher on them
# precisely by sounding less like this speaker.
OBJECTIVE_AXES = [("retention", 1), ("wer", -1), ("sim_o", 1)]


# Axes where the paired-delta CI, not the point estimate, decides "worse".  Retention is the one
# axis measured with enough noise that point estimates alone produce false verdicts: two independent
# syntheses of the *same* weights differ by up to 10 pp on `retention` with a paired CI that excludes
# zero (see section 2b).  A 2 pp point-estimate deficit on this axis is not evidence of
# anything, and previously vetoed conditions that won everywhere else.
CI_AWARE_AXES = {"retention"}


def dominance(summary: pd.DataFrame, prompt: str, ref_cond: str, axes=OBJECTIVE_AXES) -> list[str]:
    """
    Conditions (same prompt mode) that are >= the reference on every objective axis and > on at least one.

    For axes in CI_AWARE_AXES the comparison uses the paired bootstrap CI carried in the summary as
    `d_<metric>_lo` / `d_<metric>_hi` (deltas vs the report's delta reference): a condition counts as
    "not worse" when that CI contains zero, and as "strictly better" only when the CI excludes zero in
    its favour.  Axes without a CI in the frame fall back to the point estimate.
    """
    s = summary[summary["prompt"] == prompt].set_index("cond")
    if ref_cond not in s.index:
        return []
    ref = s.loc[ref_cond]
    out = []
    for cond, r in s.iterrows():
        if cond in (ref_cond, "gt", "copysynth"):
            continue
        ge, gt = True, False
        for m, sgn in axes:
            if np.isnan(r.get(m, np.nan)) or np.isnan(ref.get(m, np.nan)):
                continue
            lo, hi = r.get(f"d_{m}_lo", np.nan), r.get(f"d_{m}_hi", np.nan)
            if m in CI_AWARE_AXES and not (np.isnan(lo) or np.isnan(hi)):
                # CI is on (cond - delta_ref); re-express against ref_cond only when they coincide
                rlo, rhi = ref.get(f"d_{m}_lo", 0.0), ref.get(f"d_{m}_hi", 0.0)
                rlo = 0.0 if np.isnan(rlo) else rlo
                rhi = 0.0 if np.isnan(rhi) else rhi
                d_lo, d_hi = sgn * (lo - rhi), sgn * (hi - rlo)
                if d_hi < 0:          # CI entirely on the worse side
                    ge = False
                elif d_lo > 0:        # CI entirely on the better side
                    gt = True
            else:
                diff = sgn * (r[m] - ref[m])
                if diff < -1e-9:
                    ge = False
                elif diff > 1e-9:
                    gt = True
        if ge and gt:
            out.append(cond)
    return out


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT2,
                         "xtick.color": C_TEXT2, "ytick.color": C_TEXT2, "text.color": C_TEXT,
                         "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE})
    return plt


def make_plots(df: pd.DataFrame, fad: dict, prompt: str):
    """Box/strip plots: fine-tuned vs baseline vs ground truth (+ copy-synthesis) for one prompt mode."""
    plt = _style()
    fig, axes = plt.subplots(2, 6, figsize=(26, 9.5))
    rng = np.random.default_rng(0)
    sub = {c: df[(df["cond"] == c) & (df["prompt"] == prompt)].set_index("stem") for c in ("finetuned", "baseline")}
    ref = {c: df[df["cond"] == c].set_index("stem") for c in ("gt", "copysynth")}

    def box(ax, groups, colors, title, unit, better):
        data = [np.asarray(g, dtype=float) for g in groups.values()]
        data = [d[~np.isnan(d)] for d in data]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C_TEXT, lw=1.5), whiskerprops=dict(color=C_TEXT2, lw=1),
                        capprops=dict(color=C_TEXT2, lw=1), boxprops=dict(lw=1))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c + "33")
            patch.set_edgecolor(c)
        for k, (d, c) in enumerate(zip(data, colors), 1):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=c, alpha=0.75, edgecolor=C_SURFACE, lw=0.6, zorder=3)
            if len(d):
                ax.annotate(f"{np.mean(d):.3g}", (k, np.max(d)), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=C_TEXT2)
        ax.set_xticks(range(1, len(data) + 1), list(groups.keys()), fontsize=8)
        ax.set_title(f"{title}  ({unit}, {better} is better)", fontsize=9)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    order = ["wer", "cer", "sim_o", "sim_o_prompt", "sec", "mcd", "vde", "nmos", "smos", "cmos", "mos"]
    for ax, m in zip(axes.flat, order):
        label, better, unit = METRICS[m]
        groups, colors = {}, []
        for c, col in (("finetuned", C_FT), ("baseline", C_BASE)):
            if c in sub and m in sub[c] and not (c == "baseline" and m == "cmos"):
                groups[c], colors = sub[c][m], colors + [col]
        for c, col in (("copysynth", C_CS), ("gt", C_GT)):
            if len(ref[c]) and m in ref[c] and ref[c][m].notna().any():
                groups[c], colors = ref[c][m], colors + [col]
        box(ax, groups, colors, label, unit, better)

    ax = axes.flat[-1]
    names = [c for c in ("finetuned", "baseline") if (c, prompt) in fad] + [c for c in ("copysynth",) if (c, "-") in fad]
    vals = [fad.get((c, prompt), fad.get((c, "-"))) for c in names]
    cols = {"finetuned": C_FT, "baseline": C_BASE, "copysynth": C_CS}
    bars = ax.bar(names, vals, color=[cols[n] for n in names], width=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=C_TEXT2)
    ax.set_title("FAD  (vs real test set, lower is better)", fontsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    n = len(sub["finetuned"]) if len(sub["finetuned"]) else len(sub["baseline"])
    fig.suptitle(f"F5-TTS voice clone - test-set evaluation, '{prompt}' prompt (n = {n} utterances)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOT_PNG, dpi=130)
    plt.close(fig)
    log.info(f"plots -> {PLOT_PNG}")


def _short(cond: str) -> str:
    return cond.replace("finetuned", "ft").replace("baseline", "base").replace("alpha_", "α").replace("_cfg", " cfg").replace("_nfe", " nfe").replace("_sway", " sw")


def make_frontier(summary: pd.DataFrame, prompts: list[str]):
    """
    The objective frontier: retention vs WER, SIM-o vs WER and retention vs SIM-o for every model condition,
    coloured by alpha (or by CFG when only decoding varies), with the real-audio WER and the same-speaker
    SIM-o reference as dotted lines.
    """
    plt = _style()
    model_rows = summary[~summary["cond"].isin(["gt", "copysynth"])].copy()
    if len(model_rows) < 3:
        return
    color_key = "alpha" if model_rows["alpha"].nunique() > 1 else "cfg"
    panels = [("wer", "retention"), ("wer", "sim_o"), ("sim_o", "retention")]
    lab = {"wer": "ASR-WER (lower is better)", "retention": "pronunciation retained (%)", "sim_o": "SIM-o vs real recording"}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.2))
    gt = summary[summary["cond"] == "gt"].iloc[0] if (summary["cond"] == "gt").any() else None
    markers = {"cross": "o", "fixed": "s"}
    vmin, vmax = float(model_rows[color_key].min()), float(model_rows[color_key].max())
    sc = None
    for ax, (xm, ym) in zip(axes, panels):
        for prompt in prompts:
            s = model_rows[model_rows["prompt"] == prompt].sort_values(color_key)
            if not len(s):
                continue
            if color_key == "alpha":
                ax.plot(s[xm], s[ym], "-", color=C_GRID, lw=1.2, zorder=1)
            sc = ax.scatter(s[xm], s[ym], c=s[color_key], cmap="viridis", vmin=vmin, vmax=vmax, s=70,
                            marker=markers.get(prompt, "o"), edgecolor=C_TEXT, lw=0.6, zorder=3, label=f"{prompt} prompt")
            if ym == "retention" and "retention_lo" in s:
                ax.errorbar(s[xm], s[ym], yerr=[s[ym] - s["retention_lo"], s["retention_hi"] - s[ym]], fmt="none",
                            ecolor=C_GRID, elinewidth=0.8, capsize=2, zorder=2)
            for _, r in s.iterrows():
                ax.annotate(_short(r["cond"]), (r[xm], r[ym]), xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=C_TEXT2)
        if gt is not None and xm == "wer" and not np.isnan(gt["wer"]):
            ax.axvline(gt["wer"], color=C_GT, ls=":", lw=1.2, label="real-audio WER")
        ax.set_xlabel(lab[xm])
        ax.set_ylabel(lab[ym])
        ax.grid(color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, loc="best")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label={"alpha": "alpha (0 = pretrained, 1 = fine-tuned)", "cfg": "CFG strength"}[color_key], shrink=0.8)
    fig.suptitle("Objective frontier: retention / WER / SIM-o (real-audio WER dotted; same-speaker SIM-o reference = 0.60)", fontsize=12)
    fig.savefig(FRONTIER_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"frontier plot -> {FRONTIER_PNG}")


def make_decoding_plot(summary: pd.DataFrame, prompt: str):
    """retention / WER / SIM-o / NMOS against CFG, one line per (NFE, sway), for one alpha and prompt mode."""
    s = summary[(summary["prompt"] == prompt) & (~summary["cond"].isin(["gt", "copysynth"]))]
    if s["cfg"].nunique() < 2:
        return
    plt = _style()
    metrics = [("retention", "pronunciation retained (%)"), ("wer", "ASR-WER"), ("sim_o", "SIM-o vs real"), ("nmos", "NMOS (UTMOS22)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4))
    styles = {(32, -1.0): ("-", "o"), (64, -1.0): ("-", "s"), (32, 0.0): ("--", "o"), (64, 0.0): ("--", "s")}
    for ax, (m, label) in zip(axes, metrics):
        for (nfe, sway), g in s.groupby(["nfe", "sway"]):
            g = g.sort_values("cfg")
            ls, mk = styles.get((int(nfe), float(sway)), ("-", "o"))
            ax.plot(g["cfg"], g[m], ls, marker=mk, color=C_FT if sway == -1.0 else C_BASE, lw=1.4, ms=6, label=f"NFE {int(nfe)}, sway {sway:g}")
            if m == "retention" and "retention_lo" in g:
                ax.fill_between(g["cfg"], g["retention_lo"], g["retention_hi"], color=C_FT if sway == -1.0 else C_BASE, alpha=0.08)
        gt = summary[summary["cond"] == "gt"]
        if m == "wer" and len(gt):
            ax.axhline(gt.iloc[0]["wer"], color=C_GT, ls=":", lw=1.2, label="real audio")
        if m == "sim_o":
            ax.axhline(0.603, color=C_GT, ls=":", lw=1.2, label="same-speaker ref")
        ax.set_xlabel("CFG strength")
        ax.set_title(label, fontsize=10)
        ax.grid(color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Decoding sweep, `{prompt}` prompt", fontsize=12)
    fig.tight_layout()
    out = CHECKPOINT_DIR / "evaluation_decoding.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info(f"decoding plot -> {out}")


def training_analysis(meta: dict) -> str:
    """Training summary read from the checkpoint that was evaluated (its own best_epoch, not a CSV argmin)."""
    if not meta or not meta.get("history"):
        return "_Checkpoint carries no training history (not produced by finetune_tts.py) - training analysis unavailable._"
    h = pd.DataFrame(meta["history"])
    last = h.iloc[-1]
    best_epoch, best_val = int(meta.get("best_epoch") or 0), float(meta.get("best_val") or float("nan"))
    raw_min = h.loc[h["validation_loss"].idxmin()]
    tail = h.tail(10)
    cfg = meta.get("config", {})
    note = ""
    if int(raw_min["epoch"]) != best_epoch:
        note = (f" (the raw CSV minimum is {raw_min['validation_loss']:.4f} at epoch {int(raw_min['epoch'])}, "
                f"but it is within `--min_delta` {cfg.get('min_delta', 1e-4)} of the epoch-{best_epoch} value, so the "
                f"trainer kept epoch {best_epoch} as best and that is the checkpoint on disk)")
    lines = [
        f"* Checkpoint on disk: **epoch {meta.get('epoch')}** = trainer's best epoch **{best_epoch}**, validation loss "
        f"**{best_val:.4f}**{note}.",
        f"* Run: {int(last['epoch'])} epochs, {int(last['optimizer_updates'])} optimizer updates (planned "
        f"{meta.get('planned_epochs')}, `--grad_accum` {cfg.get('grad_accum')}, `--lr` {cfg.get('lr')}, "
        f"`--val_repeats` {cfg.get('val_repeats')}).",
        f"* First epoch: train {h.iloc[0]['training_loss']:.4f} / val {h.iloc[0]['validation_loss']:.4f} -> "
        f"final epoch: train {last['training_loss']:.4f} / val {last['validation_loss']:.4f} "
        f"(val change {100 * (best_val / h.iloc[0]['validation_loss'] - 1):+.1f} % to the best epoch).",
        f"* Last {len(tail)} epochs: val-loss std {tail['validation_loss'].std():.4f}, train-loss std "
        f"{tail['training_loss'].std():.4f}; the CFM loss is an MSE against a random (timestep, mask, cond-drop) "
        f"draw, so its level and flatness say little about clone quality - see the metric tables instead.",
        f"* Learning rate: {h['learning_rate'].max():.2e} peak -> {last['learning_rate']:.2e} final; peak VRAM "
        f"{h['peak_vram_gb'].max():.1f} GiB; mean epoch time {h['epoch_time_s'].mean() / 60:.1f} min.",
        "* Curves: `logs/training_convergence.png`; metrics table: `logs/training_metrics.csv` / `.xlsx`.",
    ]
    return "\n".join(lines)


def fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def fmt_delta(row, m, nd=3):
    if f"d_{m}" not in row or np.isnan(row[f"d_{m}"]):
        return "-"
    lo, hi = row[f"d_{m}_lo"], row[f"d_{m}_hi"]
    sig = "" if lo <= 0 <= hi else " *"
    return f"{row[f'd_{m}']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]{sig}"


def phone_section_md(phone: dict, prompts: list[str]) -> str:
    """Section 6a: the phone-level retention tables, one per prompt mode."""
    if not phone:
        return ("## 6a. Phone-level pronunciation retention\n\n"
                "_Not computed for this run (`--no_phone_metric`, or no `baseline` condition was present)._\n")
    out = ["## 6a. Phone-level pronunciation retention (primary measurement)", "",
           "Markers are words where the **speaker's** realised phone sequence differs from the **zero-shot "
           "pretrained model's** realisation of the same word in the same sentence by at least "
           f"`{next(iter(phone.values()))['marker_per']}` PER (phone edit distance, forced-aligned word spans, "
           "CTC phone recogniser). Homophones, inflection and tokenisation splits are excluded *by construction* "
           "- they produce near-identical phone strings and so never clear the threshold.", "",
           "`SPD` = mean phone distance from the condition to the **speaker** (lower = closer to how he actually "
           "says the word). It never references the prior, so unlike `retained %` it cannot be inflated by a "
           "condition merely drifting away from the pretrained model.", ""]
    for prompt in prompts:
        d = phone.get(prompt)
        if not d:
            continue
        conds = d["conditions"]
        floor = conds.get(d["reference"], {}).get("spd")
        ceil = conds.get("copysynth", {}).get("spd")
        def spa(v):
            if floor is None or ceil is None or floor == ceil or v is None:
                return ""
            return f"{100 * (floor - v) / (floor - ceil):5.1f} %"
        out += [f"### `{prompt}` prompt", "",
                f"* Markers: **{d['n_markers']}** word types of {d['n_aligned_words']} aligned words "
                f"(one marker per word type per utterance; repeated tokens averaged).",
                f"* Synthesis seeds pooled per condition: **{d['seeds_pooled']}**"
                + ("  _(single seed - see the caveat below)_" if d["seeds_pooled"] < 3 else ""),
                f"* Anchors: zero-shot `{d['reference']}` SPD {floor:.3f} (floor)"
                + (f", copy-synthesis SPD {ceil:.3f} (ceiling)" if ceil is not None else "") + ".", "",
                "| condition | SPD | 95 % CI | % of the way to the speaker | retained % | 95 % CI | n |",
                "|---|---|---|---|---|---|---|"]
        for c, v in sorted(conds.items(), key=lambda kv: kv[1]["spd"]):
            out.append(f"| `{c}` | {v['spd']:.3f} | [{v['spd_lo']:.3f}, {v['spd_hi']:.3f}] | {spa(v['spd'])} "
                       f"| {v['retained_pct']:.1f} % | [{v['retained_lo']:.1f}, {v['retained_hi']:.1f}] "
                       f"| {v['n_markers']} |")
        ref_c = d.get("delta_reference")
        deltas = [(c, v) for c, v in conds.items() if "d_spd" in v]
        if ref_c and deltas:
            out += ["", f"Paired deltas vs `{ref_c}` (CI excludes zero = the difference is real):", "",
                    "| condition | dSPD | 95 % CI | separates? |", "|---|---|---|---|"]
            for c, v in sorted(deltas, key=lambda kv: kv[1]["d_spd"]):
                sig = "yes" if (v["d_spd_lo"] > 0 or v["d_spd_hi"] < 0) else "no"
                out.append(f"| `{c}` | {v['d_spd']:+.3f} | [{v['d_spd_lo']:+.3f}, {v['d_spd_hi']:+.3f}] | {sig} |")
        out.append("")
    out += ["**Reliability caveat.** `retained %` is a thresholded statistic and is noisy at a single seed: two "
            "independent syntheses of *identical weights* have differed by 10 pp on the `fixed` prompt, with a "
            "paired CI excluding zero. `SPD` passed that same identity test. Pooled over >= 3 seeds both statistics "
            "agree between independent training runs to within ~2-5 pp. Read single-seed `retained %` as indicative "
            "only; use `SPD` with >= 3 pooled seeds for any comparison that decides something.", ""]
    return "\n".join(out)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, fad: dict, retention: dict, sim_sanity: dict,
                 args, ckpt: Path, meta: dict, prompts: list[str], enrol: dict | None, chunks: dict,
                 phone: dict | None = None):
    n_test = df[df["cond"] == "gt"]["stem"].nunique()
    gt = summary[summary["cond"] == "gt"].iloc[0] if (summary["cond"] == "gt").any() else None
    cs = summary[summary["cond"] == "copysynth"].iloc[0] if (summary["cond"] == "copysynth").any() else None
    alphas = sorted(summary[~summary["cond"].isin(["gt", "copysynth"])]["alpha"].unique())
    ckpt_epoch = meta.get("epoch")

    # ---- ceiling table -----------------------------------------------------
    ceil_rows = []
    for m in ("nmos", "mos", "wer", "cer", "sim_o", "mcd", "vde"):
        label = METRICS[m][0]
        cells = [fmt(gt[m]) if gt is not None and m in gt else "-", fmt(cs[m]) if cs is not None and m in cs else "-"]
        for prompt in prompts:
            for c in ("baseline", "finetuned"):
                r = summary[(summary["cond"] == c) & (summary["prompt"] == prompt)]
                cells.append(fmt(r.iloc[0][m]) if len(r) and m in r else "-")
        ceil_rows.append(f"| {label} ({METRICS[m][1]} better) | " + " | ".join(cells) + " |")
    ceil_head = "| Metric | real audio | copy-synthesis (mel->vocos) | " + " | ".join(
        f"{c} / {p}" for p in prompts for c in ("baseline", "finetuned")) + " |"
    ceil_sep = "|---|" + "---:|" * (2 + 2 * len(prompts))

    # ---- main tables per prompt mode ---------------------------------------
    ref = args.delta_ref
    main_sections = []
    for prompt in prompts:
        s = summary[(summary["prompt"] == prompt)].sort_values(["alpha", "cfg", "nfe", "sway"])
        head = ("| condition | alpha | CFG | NFE | sway | retained % [95 % CI] | WER | SIM-o | CER | SIM-o(prompt) | SEC | MCD | VDE | FAD | NMOS | SMOS | CMOS | MOS | normalised % |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        body = []
        for _, r in s.iterrows():
            ret = f"{fmt(r['retention'], 1)} [{fmt(r.get('retention_lo'), 1)}, {fmt(r.get('retention_hi'), 1)}]"
            body.append(f"| {r['cond']} | {r['alpha']:.2f} | {fmt(r['cfg'], 1)} | {fmt(r['nfe'], 0)} | {fmt(r['sway'], 1)} | {ret} | "
                        f"**{fmt(r['wer'])}** | **{fmt(r['sim_o'])}** | {fmt(r['cer'])} | "
                        f"{fmt(r.get('sim_o_prompt'))} | {fmt(r['sec'])} | {fmt(r['mcd'], 2)} | {fmt(r['vde'])} | "
                        f"{fmt(r['fad'], 2)} | {fmt(r['nmos'])} | {fmt(r['smos'], 2)} | {fmt(r.get('cmos'))} | "
                        f"{fmt(r['mos'])} | {fmt(r['normalised'], 1)} |")
        dhead = ("| condition | Δretained (pp) | ΔWER | ΔSIM-o | ΔCER | ΔSIM-o(prompt) | ΔSEC | ΔMCD | ΔVDE | ΔNMOS | ΔMOS |\n"
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        dbody = []
        for _, r in s.iterrows():
            if r["cond"] == ref:
                continue
            dbody.append(f"| {r['cond']} | {fmt_delta(r, 'retention', 1)} | " + " | ".join(
                fmt_delta(r, m, 2 if m == "mcd" else 3) for m in DELTA_METRICS) + " |")
        dom_ft = dominance(summary, prompt, "finetuned")
        dom_base = dominance(summary, prompt, "baseline")
        main_sections.append(
            f"### `{prompt}` prompt\n\n{head}\n" + "\n".join(body) + "\n\n"
            f"Paired deltas vs `{ref}` over the same {n_test} utterances, mean [95 % bootstrap CI, "
            f"10 000 resamples]; `*` = CI excludes 0. Retention deltas are in percentage points over the shared markers:\n\n"
            f"{dhead}\n" + "\n".join(dbody) + "\n\n"
            f"* Conditions that **dominate the fine-tuned checkpoint at default decoding** on the objective axes "
            f"(>= retention, <= WER, >= SIM-o, strictly better on at least one): {', '.join(dom_ft) if dom_ft else 'none'}.\n"
            f"* Conditions that dominate the zero-shot baseline on the same axes: {', '.join(dom_base) if dom_base else 'none'}.\n")

    # ---- retention examples (fine-tuned, first prompt mode) ----------------
    ret_ft = retention.get(("finetuned", prompts[0])) or {"n_idio": 0, "retained": 0, "normalised": 0, "other": 0, "examples": []}
    ret_base = retention.get(("baseline", prompts[0]))
    n_idio = ret_ft["n_idio"]
    ex_lines = [f"| {e['stem']} | {e['written']} | {e['heard_real']} | {e['heard_clone']} | {e['outcome']} |" for e in ret_ft["examples"][:12]]

    # ---- per-utterance (fine-tuned, first prompt mode) ---------------------
    per = df[(df["cond"] == "finetuned") & (df["prompt"] == prompts[0])]
    cols = [c for c in ["stem", "duration", "n_chunks", "wer", "cer", "sim_o", "sec", "mcd", "vde", "nmos", "cmos", "mos"] if c in per]
    per_md = per[cols].to_markdown(index=False, floatfmt=".3f") if len(per) else "_none_"

    # ---- chunking ------------------------------------------------------------
    chunk_lines = []
    for (cond, prompt), ch in sorted(chunks.items()):
        multi = {k: v for k, v in ch.items() if v > 1}
        chunk_lines.append(f"* `{cond}` / `{prompt}`: {len(multi)} of {len(ch)} utterances synthesised in >1 chunk"
                           + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(multi.items(), key=lambda kv: -kv[1]))})" if multi else ""))

    enrol_txt = ""
    if enrol:
        enrol_txt = (f"* `fixed` prompt: `{enrol['source']}` cut to {enrol['duration']:.1f} s"
                     + (f", DNSMOS OVRL {enrol['dnsmos_ovrl']:.2f} (best of {len(enrol.get('candidates', []))} valid-split candidates)"
                        if "dnsmos_ovrl" in enrol else "") + ", used for every utterance.\n")

    md = f"""# F5-TTS Voice-Cloning Evaluation Report

*Generated {time.strftime('%Y-%m-%d %H:%M')} - checkpoint `{ckpt.name}` (epoch {ckpt_epoch}, {ckpt.stat().st_size / 1024**3:.2f} GB), Whisper `{args.whisper_model}`, CFG {{{args.cfg_strengths}}}, NFE {{{args.nfe_steps}}}, sway {{{args.sways}}}, speed {args.speed}, cross-fade {args.cross_fade} s, seed {args.seed}, chunking {'off' if args.no_chunk else 'on'}.*

## 1. Setup

* Test split: **{n_test}** held-out utterances ({df[df['cond'] == 'gt']['duration'].sum() / 60:.1f} min of real speech), none of which share a recording with the training data (duplicates were removed by fingerprint + waveform cross-correlation before splitting).
* Conditions: `alpha_a` = weight interpolation (1-a)·pretrained + a·fine-tuned over the EMA weights ({', '.join(f'{a:.1f}' for a in alphas)}); `baseline` is alpha 0 (zero-shot `F5TTS_Base`), `finetuned` alpha 1 (`{ckpt.name}`); `copysynth` = the real recordings passed through mel -> vocos (the vocoder ceiling); `gt` = the real recordings.
* Prompting: `cross` = <= 12 s exact-text prompt cut from a *different* test clip (forced alignment keeps the text exact). {"`fixed` = one enrolment prompt for all utterances." if "fixed" in prompts else ""} The total duration is fixed to prompt + ground-truth length ({'on' if args.use_truth_duration else 'off'}).
{enrol_txt}* Generated audio: `output/evaluation/<condition>/<prompt>/`, copy-synthesis: `output/evaluation/copysynth/`, prompts: `output/evaluation/prompts/`, `prompts_fixed/` (kept out of `output/generated_audio/`).

## 2. Metric sanity: SIM-o separates speakers, resemblyzer does not

WavLM-large + ECAPA-TDNN cosine (the Seed-TTS / F5-TTS "SIM-o" model):

| pairs | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| same speaker (real test clip vs real test clip) | {sim_sanity['same_n']} | {sim_sanity['same_mean']:.3f} | {sim_sanity['same_median']:.3f} | {sim_sanity['same_min']:.3f} | {sim_sanity['same_max']:.3f} |
| different speaker (real test clip vs F5-TTS example speakers) | {sim_sanity['diff_n']} | {sim_sanity['diff_mean']:.3f} | {sim_sanity['diff_median']:.3f} | {sim_sanity['diff_min']:.3f} | {sim_sanity['diff_max']:.3f} |
| same speaker, resemblyzer (for comparison) | {sim_sanity['sec_same_n']} | {sim_sanity['sec_same_mean']:.3f} | {sim_sanity['sec_same_median']:.3f} | {sim_sanity['sec_same_min']:.3f} | {sim_sanity['sec_same_max']:.3f} |
| different speaker, resemblyzer | {sim_sanity['sec_diff_n']} | {sim_sanity['sec_diff_mean']:.3f} | {sim_sanity['sec_diff_median']:.3f} | {sim_sanity['sec_diff_min']:.3f} | {sim_sanity['sec_diff_max']:.3f} |

The same-speaker SIM-o spread across this speaker's own recordings is the reference for what a "perfect" clone can score against a different real recording: the target is the same-speaker distribution, not 1.0.

## 3. Ceiling: real audio vs the vocoder path vs the models

{ceil_head}
{ceil_sep}
{chr(10).join(ceil_rows)}

If copy-synthesis sits at the real-audio level, the recordings themselves bound the naturalness scores and no training can lift a clone above the pretrained baseline on NMOS/DNSMOS; if copy-synthesis is well above real audio, the models are losing something the mel/vocoder path preserves.

## 4. Results per prompting mode

{chr(10).join(main_sections)}
Chunking (f5_tts splits long text at `max_chars` derived from the prompt; 1.1.22 allocates the fixed duration across chunks by text length):

{chr(10).join(chunk_lines) if chunk_lines else '* n/a'}

Metric notes:

* **ASR-WER / CER** - Whisper `{args.whisper_model}` transcripts vs the written transcript after lower-casing and punctuation removal. Because the transcript deliberately spells non-standard pronunciations, the *real* recordings do not score 0 either: the ground-truth WER above is the ceiling for this speaker.
* **SIM-o** - WavLM-large + ECAPA-TDNN speaker-verification cosine between the generated utterance and the real recording of the same text; **SIM-o (prompt)** is the same model against the prompt audio (the paper definition). **SEC** is resemblyzer (GE2E), kept for continuity; it saturates on same-speaker pairs (section 2).
* **MCD** - `pymcd` mel-cepstral distortion (dB) with DTW alignment. **VDE** - fraction of DTW-aligned frames whose pyin voicing decision differs.
* **FAD** - Frechet distance between wav2vec2-base layer-6 embeddings (1 s windows) of the generated set and the real test set; set-level, indicative only at n = {n_test}.
* **NMOS** - UTMOS22-strong; **MOS** - DNSMOS P.835 OVRL. The real clips are RoFormer-isolated vocals from music-backed recordings, so their scores are a reference point, not an upper bound.
* **SMOS** - SEC mapped from [0.50, 0.95] to [1, 5]. **CMOS** - NMOS(condition) - NMOS(baseline) per utterance (> 0 = clone preferred).

## 5. Training convergence and stability

{training_analysis(meta)}

{phone_section_md(phone or {}, prompts)}
## 6b. Idiosyncratic pronunciation assessment - legacy ASR proxy (`finetuned`, `{prompts[0]}` prompt, Whisper `{args.whisper_model}`)

The speaker's transcript spells words the way he says them. Whisper therefore *mis-hears* those words on the **real** audio, which marks every non-standard pronunciation in the test set; for each marker we check what Whisper hears on the clone:

* Markers found in the real test audio: **{n_idio}** words across {n_test} utterances.
* **Retained** (clone mis-heard the same way -> pronunciation reproduced): **{ret_ft['retained']} ({100 * ret_ft['retained'] / n_idio if n_idio else float('nan'):.1f} %)**
* **Normalised** (Whisper hears the dictionary spelling -> clone "corrected" the accent): {ret_ft['normalised']} ({100 * ret_ft['normalised'] / n_idio if n_idio else float('nan'):.1f} %)
* Other: {ret_ft['other']}.{f" The zero-shot baseline retains {100 * ret_base['retained'] / n_idio:.1f} %." if ret_base and n_idio else ""}

| utterance | written | heard on real audio | heard on clone | outcome |
|---|---|---|---|---|
{chr(10).join(ex_lines) if ex_lines else '| - | - | - | - | - |'}

Retention for every condition is in the tables of section 4 (`retained %`, with bootstrap CIs over utterances).

Caveat on the marker definition: a marker is "Whisper hears a different word on the real audio", which also fires on Whisper's own lexical priors (rare proper nouns such as `suleiman` -> `sullivan`), on morphology (`ranking` -> `ranked`, `university` -> `universities`) and on tokenisation (`post` -> `postgraduate`). Those are counted alongside genuine segmental idiosyncrasies, and a clone that pronounces a name *correctly per the transcript* is scored "normalised". Treat the aggregate as an ASR-proxy, not a phonetic measurement - section 6a measures the phones directly and supersedes it. This section is retained only for continuity with earlier reports.

## 7. Per-utterance results (`finetuned`, `{prompts[0]}` prompt)

{per_md}

## 8. Files

* `checkpoints/evaluation_frontier.png` - SIM-o / retention against NMOS / DNSMOS / WER across alpha (both prompt modes).
* `checkpoints/evaluation_metrics.png` - box/strip plots of every metric (fine-tuned vs baseline vs copy-synthesis vs ground truth) and the FAD bars.
* `checkpoints/evaluation_results.csv` / `.json` - long-form per-utterance numbers (one row per condition x prompt x utterance), transcripts and pronunciation markers, plus the condition summary with CIs.
* `logs/training_convergence.png`, `logs/training_metrics.csv` / `.xlsx`, `logs/finetune_tts.log` - training curves, metrics and log.
"""
    REPORT_MD.write_text(md, encoding="utf-8")
    log.info(f"report -> {REPORT_MD}")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the fine-tuned F5-TTS model on the test split.")
    p.add_argument("--ckpt", default=None, help="fine-tuned checkpoint (default: checkpoints/best_model.pt)")
    p.add_argument("--pretrain", default=None, help="local pretrained checkpoint for interpolation (default: HF hub cache)")
    p.add_argument("--alphas", default="0,1",
                   help="comma-separated interpolation weights; 0 = zero-shot baseline, 1 = fine-tuned checkpoint")
    p.add_argument("--prompt_mode", default="both", choices=["cross", "fixed", "both"])
    p.add_argument("--enrol_wav", default=None, help="fixed-mode enrolment prompt (default: best-DNSMOS valid-split cut)")
    p.add_argument("--enrol_text", default=None, help="exact transcript of --enrol_wav (default: sidecar .txt)")
    p.add_argument("--whisper_model", default="large-v3", help="base | small | medium | large-v3")
    p.add_argument("--cfg_strengths", default="2.0", help="comma-separated CFG strengths to sweep (F5-TTS default 2.0)")
    p.add_argument("--nfe_steps", default="32", help="comma-separated ODE step counts to sweep (default 32)")
    p.add_argument("--sways", default="-1.0", help="comma-separated sway-sampling coefficients to sweep (default -1.0)")
    p.add_argument("--delta_ref", default="baseline",
                   help="condition the paired deltas are computed against (e.g. 'finetuned' for a decoding sweep)")
    p.add_argument("--cross_fade", type=float, default=0.15, help="cross-fade between text chunks (s)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N test utterances (smoke test)")
    p.add_argument("--no_copysynth", action="store_true", help="skip the mel -> vocos copy-synthesis ceiling")
    p.add_argument("--no_chunk", action="store_true",
                   help="synthesise each utterance's full text in one pass instead of f5_tts's max_chars chunks")
    p.add_argument("--regenerate", action="store_true", help="ignore cached generated audio")
    p.add_argument("--no_truth_duration", dest="use_truth_duration", action="store_false",
                   help="let F5-TTS estimate the duration from the prompt's speaking rate instead of using the "
                        "ground-truth duration (the speaker's rate varies ~3x between clips)")
    p.add_argument("--skip_cleanup", action="store_true", help="skip the pre-run disk sweep at startup")
    # phone-level retention (section 2b) - the primary retention measurement
    p.add_argument("--no_phone_metric", dest="phone_metric", action="store_false",
                   help="skip the phone-level retention analysis and report only the legacy ASR-proxy retention")
    p.add_argument("--marker_per", type=float, default=0.70,
                   help="min PER(real, prior) for a word to count as an idiosyncratic-pronunciation marker. "
                        "Calibrated against the copy-synthesis noise floor: at 0.70 only 3.1 %% of noise-floor word "
                        "pairs clear the bar against 34.4 %% of real-vs-prior pairs (10.9x separation)")
    p.add_argument("--phone_min_phones", type=int, default=2, help="min phones in a word span for it to be scored")
    p.add_argument("--phone_seeds", type=int, default=1,
                   help="pool this many synthesis seeds per condition (reads <cond>_seed1/, _seed2/, ...). Seed noise "
                        "alone can reach paired significance, so sweeps should use >= 3")
    p.add_argument("--phone_lexicon", action="store_true",
                   help="use the whole-dataset modal realisation as the speaker reference and drop one-off tokens "
                        "(requires checkpoints/speaker_lexicon.json; see --build_lexicon)")
    p.add_argument("--build_lexicon", action="store_true",
                   help="(re)build checkpoints/speaker_lexicon.json from train+valid+test before scoring")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    common.ensure_layout()
    common.run_cleanup(log, skip=args.skip_cleanup, require_gb=5)
    common.ensure_ffmpeg(log)
    device = args.device or common.gpu_report(log, min_free_gb=4.0)

    from cached_path import cached_path
    from finetune_tts import MODEL_CONFIGS
    from inference import load_tts, resolve_checkpoint

    ckpt = resolve_checkpoint(args.ckpt)
    meta = checkpoint_meta(ckpt)
    alphas = sorted({round(float(a), 4) for a in args.alphas.split(",")})
    import itertools

    decodings = sorted({(round(float(c), 3), int(n), round(float(s), 3)) for c, n, s in itertools.product(
        args.cfg_strengths.split(","), args.nfe_steps.split(","), args.sways.split(","))})
    prompts = ["cross", "fixed"] if args.prompt_mode == "both" else [args.prompt_mode]
    rows = common.read_split_csv("test")
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        assert common.abs_path(r["audio_path"]).exists(), f"missing test audio {r['audio_path']}"
    stems = [Path(r["audio_path"]).stem for r in rows]
    gt = {s: common.abs_path(r["audio_path"]) for s, r in zip(stems, rows)}
    text = {s: r["text"] for s, r in zip(stems, rows)}
    dur = {s: r["duration"] for s, r in zip(stems, rows)}
    log.info(f"evaluating {len(rows)} test utterances | ckpt {ckpt.name} (epoch {meta.get('epoch')}) | "
             f"alphas {alphas} | decodings (cfg, nfe, sway) {decodings} | prompts {prompts} | whisper {args.whisper_model} "
             f"| deltas vs '{args.delta_ref}'")
    if args.no_chunk:
        log.info("chunking disabled: every utterance is synthesised in a single pass")

    # ---- prompts ---------------------------------------------------------------
    dns = DNSMOS()
    prompt_sets: dict[str, dict[str, tuple[Path, str]]] = {}
    enrol = None
    if "cross" in prompts:
        prompt_sets["cross"] = build_prompts(rows, device)
    if "fixed" in prompts:
        wav_p, enrol_text, enrol = build_fixed_prompt(args, device, dns)
        prompt_sets["fixed"] = {s: (wav_p, enrol_text) for s in stems}

    # ---- generation ------------------------------------------------------------
    pretrain = args.pretrain or str(cached_path(MODEL_CONFIGS["F5TTS_Base"]["ckpt"]))
    gen: dict[tuple[str, str], dict[str, Path]] = {}  # (cond, prompt) -> {stem: wav}
    chunks: dict[tuple[str, str], dict] = {}
    cond_meta: dict[str, dict] = {}
    for alpha in alphas:
        for dec in decodings:
            cond = cond_tag(alpha, dec)
            cond_meta[cond] = {"alpha": alpha, "cfg": dec[0], "nfe": dec[1], "sway": dec[2]}
            for prompt in prompts:
                out_dir = EVAL_GEN_DIR / cond / (prompt + ("_nochunk" if args.no_chunk else "")
                                                 + (f"_xf{args.cross_fade:g}" if args.cross_fade != 0.15 else ""))
                newer = 0.0 if alpha == 0.0 else ckpt.stat().st_mtime
                chunks[(cond, prompt)] = generate_set(
                    rows, prompt_sets[prompt], out_dir, f"{cond}/{prompt}", args, device,
                    model_loader=lambda a=alpha: load_tts(ckpt, device, alpha=a, pretrained=pretrain), newer_than=newer, dec=dec)
                gen[(cond, prompt)] = {s: out_dir / f"{s}.wav" for s in stems}
    if not args.no_copysynth:
        copy_synthesis(rows, device, args.regenerate)
        gen[("copysynth", "-")] = {s: COPYSYNTH_DIR / f"{s}.wav" for s in stems}
    gen[("gt", "-")] = dict(gt)

    # ---- per-utterance metrics ---------------------------------------------------
    cache = MetricCache(METRIC_CACHE)
    res: dict[tuple[str, str, str], dict] = {}
    for (cond, prompt), paths in gen.items():
        for s in stems:
            cm = cond_meta.get(cond, {})
            res[(cond, prompt, s)] = {"cond": cond, "prompt": prompt, "alpha": cm.get("alpha", float("nan")),
                                      "cfg": cm.get("cfg", float("nan")), "nfe": cm.get("nfe", float("nan")),
                                      "sway": cm.get("sway", float("nan")), "stem": s,
                                      "duration": dur[s], "text": text[s], "n_chunks": chunks.get((cond, prompt), {}).get(s)}

    import jiwer

    log.info(f"[1-2] ASR-WER / CER (Whisper {args.whisper_model}) ...")
    asr = Whisper(args.whisper_model, device)
    for (cond, prompt), paths in gen.items():
        for s in stems:
            hyp = asr(paths[s])
            ref, hn = normalize_words(text[s]), normalize_words(hyp)
            res[(cond, prompt, s)].update({"asr": hyp, "wer": jiwer.wer(ref, hn) if hn else 1.0, "cer": jiwer.cer(ref, hn) if hn else 1.0})
    free_gpu(asr.model, asr)
    del asr
    free_gpu()

    log.info("[3] SIM-o (WavLM-large + ECAPA-TDNN) vs real recording and vs prompt ...")
    simo = SimO(device)
    for (cond, prompt), paths in gen.items():
        if cond == "gt":
            continue
        for s in stems:
            v = cache.get("sim_o", paths[s], gt[s])
            if v is None:
                v = simo.cosine(paths[s], gt[s])
                cache.put("sim_o", v, paths[s], gt[s])
            res[(cond, prompt, s)]["sim_o"] = v
            if prompt in prompt_sets:
                pw = prompt_sets[prompt][s][0]
                vp = cache.get("sim_o", paths[s], pw)
                if vp is None:
                    vp = simo.cosine(paths[s], pw)
                    cache.put("sim_o", vp, paths[s], pw)
                res[(cond, prompt, s)]["sim_o_prompt"] = vp
    # sanity distributions: same speaker (real vs real) and different speaker (real vs f5 example speakers)
    import glob
    import itertools
    from importlib.resources import files

    ex_dir = str(files("f5_tts").joinpath("infer/examples"))
    others = sorted(glob.glob(f"{ex_dir}/**/*.wav", recursive=True) + glob.glob(f"{ex_dir}/**/*.flac", recursive=True))
    same = [simo.cosine(gt[a], gt[b]) for a, b in itertools.combinations(stems, 2)]
    diff = [simo.cosine(gt[a], Path(o)) for a in stems for o in others]
    free_gpu(simo.model, simo)
    del simo
    free_gpu()

    log.info("[3b, 8] resemblyzer SEC / SMOS ...")
    spk = SpeakerEncoder(device)
    for (cond, prompt), paths in gen.items():
        if cond == "gt":
            continue
        for s in stems:
            v = cache.get("sec", paths[s], gt[s])
            if v is None:
                v = spk.cosine(paths[s], gt[s])
                cache.put("sec", v, paths[s], gt[s])
            res[(cond, prompt, s)].update({"sec": v, "smos": smos_from_cosine(v)})
    sec_same = [spk.cosine(gt[a], gt[b]) for a, b in itertools.combinations(stems, 2)]
    sec_diff = [spk.cosine(gt[a], Path(o)) for a in stems for o in others]
    free_gpu(spk.enc, spk)
    del spk
    free_gpu()
    sim_sanity = {}
    for name, vals in (("same", same), ("diff", diff), ("sec_same", sec_same), ("sec_diff", sec_diff)):
        v = np.asarray(vals) if vals else np.array([np.nan])
        sim_sanity.update({f"{name}_n": len(vals), f"{name}_mean": float(np.mean(v)), f"{name}_median": float(np.median(v)),
                           f"{name}_min": float(np.min(v)), f"{name}_max": float(np.max(v))})
    log.info(f"  SIM-o same-speaker {sim_sanity['same_mean']:.3f} [{sim_sanity['same_min']:.2f}, {sim_sanity['same_max']:.2f}] "
             f"vs different-speaker {sim_sanity['diff_mean']:.3f} [{sim_sanity['diff_min']:.2f}, {sim_sanity['diff_max']:.2f}] | "
             f"resemblyzer same {sim_sanity['sec_same_mean']:.3f} vs diff {sim_sanity['sec_diff_mean']:.3f}")

    log.info("[4] MCD-DTW (pymcd) ...")
    from pymcd.mcd import Calculate_MCD

    mcd = Calculate_MCD(MCD_mode="dtw")
    for (cond, prompt), paths in gen.items():
        if cond == "gt":
            continue
        for s in stems:
            v = cache.get("mcd", paths[s], gt[s])
            if v is None:
                v = float(mcd.calculate_mcd(str(gt[s]), str(paths[s])))
                cache.put("mcd", v, paths[s], gt[s])
            res[(cond, prompt, s)]["mcd"] = v

    log.info("[5] VDE (pyin voicing along MFCC-DTW) ...")
    for (cond, prompt), paths in gen.items():
        if cond == "gt":
            continue
        for s in stems:
            v = cache.get("vde", paths[s], gt[s])
            if v is None:
                v = voicing_decision_error(gt[s], paths[s])
                cache.put("vde", v, paths[s], gt[s])
            res[(cond, prompt, s)]["vde"] = v
    cache.flush()

    log.info("[7, 9] UTMOS neural MOS / CMOS ...")
    utmos = UTMOS(device)
    for (cond, prompt), paths in gen.items():
        for s in stems:
            v = cache.get("nmos", paths[s])
            if v is None:
                v = utmos(paths[s])
                cache.put("nmos", v, paths[s])
            res[(cond, prompt, s)]["nmos"] = v
    free_gpu(utmos.model, utmos)
    del utmos
    free_gpu()
    for (cond, prompt) in gen:
        base_key = ("baseline", prompt)
        for s in stems:
            if cond not in ("gt", "copysynth", "baseline") and base_key in gen:
                res[(cond, prompt, s)]["cmos"] = res[(cond, prompt, s)]["nmos"] - res[("baseline", prompt, s)]["nmos"]

    log.info("[10] DNSMOS P.835 ...")
    for (cond, prompt), paths in gen.items():
        for s in stems:
            d = cache.get("dnsmos", paths[s])
            if d is None:
                d = dns(paths[s])
                cache.put("dnsmos", d, paths[s])
            res[(cond, prompt, s)].update({"mos": d["ovrl"], "dnsmos_sig": d["sig"], "dnsmos_bak": d["bak"]})
    cache.flush()

    log.info("[6] FAD (wav2vec2 embeddings) ...")
    emb = FADEmbedder(device)
    e_gt = np.concatenate([emb(gt[s]) for s in stems])
    fad = {}
    for (cond, prompt), paths in gen.items():
        if cond == "gt":
            continue
        fad[(cond, prompt)] = frechet_distance(np.concatenate([emb(paths[s]) for s in stems]), e_gt)
    free_gpu(emb.model, emb)
    del emb
    free_gpu()

    log.info("idiosyncratic pronunciation analysis ...")
    retention = {}
    for (cond, prompt) in gen:
        if cond == "gt":
            continue
        tot = {"n_idio": 0, "retained": 0, "normalised": 0, "other": 0, "examples": []}
        for s in stems:
            a = idiosyncrasy_analysis(text[s], res[("gt", "-", s)]["asr"], res[(cond, prompt, s)]["asr"])
            res[(cond, prompt, s)].update({"idio_words": a["n_idio"], "idio_retained": a["retained"], "idio_normalised": a["normalised"]})
            for k in ("n_idio", "retained", "normalised", "other"):
                tot[k] += a[k]
            tot["examples"] += [dict(e, stem=s) for e in a["examples"]]
        retention[(cond, prompt)] = tot

    phone = {}
    if args.phone_metric:
        log.info("phone-level pronunciation retention ...")
        free_gpu()
        ph_aligner, ph_rec = common.ForcedAligner(device), PhoneRecognizer(device)
        try:
            for prompt in prompts:
                conds_p = [c for (c, pp) in gen if pp == prompt and c != "gt"]
                if COPYSYNTH_DIR.is_dir() and "copysynth" not in conds_p:
                    conds_p.append("copysynth")
                if "baseline" not in conds_p:
                    log.warning(f"[phone/{prompt}] no 'baseline' condition present - the phone metric needs the "
                                "zero-shot prior as its reference; skipping")
                    continue
                try:
                    phone[prompt] = phone_retention_analysis(rows, conds_p, prompt, device, args,
                                                             delta_ref=args.delta_ref,
                                                             aligner=ph_aligner, rec=ph_rec)
                except Exception as e:  # noqa: BLE001 - a metric failure must not lose the rest of the report
                    log.warning(f"[phone/{prompt}] phone-level retention failed: {type(e).__name__}: {e}")
        finally:
            ph_rec.release()
            free_gpu(ph_aligner)

    # ---- outputs -----------------------------------------------------------------
    df = pd.DataFrame(list(res.values()))
    summary = condition_summary(df, fad, retention, ref_cond=args.delta_ref)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps({
        "per_utterance": list(res.values()), "summary": summary.to_dict(orient="records"),
        "fad": {f"{c}/{p}": v for (c, p), v in fad.items()},
        "retention": {f"{c}/{p}": v for (c, p), v in retention.items()}, "sim_sanity": sim_sanity,
        "phone_retention": phone,
        "chunks": {f"{c}/{p}": v for (c, p), v in chunks.items()}, "enrol": enrol,
        "args": vars(args), "checkpoint": str(ckpt), "checkpoint_meta": {k: v for k, v in meta.items() if k != "history"},
    }, indent=2, default=str), encoding="utf-8")
    if ("finetuned", prompts[0]) in gen or ("baseline", prompts[0]) in gen:
        make_plots(df, fad, prompts[0])
    make_frontier(summary, prompts)
    for prompt in prompts:
        make_decoding_plot(summary, prompt)
    write_report(df, summary, fad, retention, sim_sanity, args, ckpt, meta, prompts, enrol, chunks, phone)

    for _, r in summary.iterrows():
        log.info(f"summary {r['cond']:>10s}/{r['prompt']:<5s}: " + " | ".join(
            f"{METRICS[m][0]} {r[m]:.3f}" for m in ("wer", "sim_o", "nmos", "mos", "fad") if m in r and not np.isnan(r[m]))
            + (f" | retained {r['retention']:.1f}%" if not np.isnan(r["retention"]) else ""))
    log.info(f"evaluation finished in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
