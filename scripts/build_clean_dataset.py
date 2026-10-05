#!/usr/bin/env python
"""
Script 1 - build_clean_dataset.py

    raw_data/*.mp3|wav  --(MelBand RoFormer vocals + MelBand RoFormer denoise)-->  data/clean_audios/<stem>.wav
                        --(fingerprint dedup >0.95)-------------------------------->  duplicates dropped & logged
                        --(80/10/10 stratified)------------------------------------>  data/splits/{train,valid,test}/ + *.csv

Vocal isolation uses the `audio-separator` package (the same MelBand RoFormer
checkpoints as Kijai's ComfyUI-MelBandRoFormer node): stage 1 extracts the
vocals stem, stage 2 runs the MelBand RoFormer denoiser on it.  The previous
dataset in data/clean_audios/ and data/splits/ is DELETED and rebuilt from
scratch on every full run (use --skip_separation to only redo dedup + split).

Metadata CSVs use the pipe-separated format `audio_path|text|duration`.
Transcript spelling is preserved verbatim (idiosyncratic pronunciations such as
"smatch" are exactly what the model must learn); only byte-level encoding
damage in the source CSV is repaired, and rows whose text is unrecoverable are
excluded and logged.

Usage:
    python scripts/build_clean_dataset.py                 # full pipeline
    python scripts/build_clean_dataset.py --skip_separation   # reuse clean_audios/
    python scripts/build_clean_dataset.py --separator_model model_bs_roformer_ep_317_sdr_12.9755.ckpt
    python scripts/build_clean_dataset.py --no_denoise      # vocals stage only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    CLEAN_DIR,
    SPLITS_DIR,
    SPLIT_NAMES,
    TARGET_SR,
    assert_audio_ok,
    load_audio,
    rms_normalize,
    save_wav,
)

log = common.get_logger("build_clean_dataset")


# --------------------------------------------------------------------------- #
# 1. Vocal isolation + denoising with MelBand RoFormer (audio-separator)
# --------------------------------------------------------------------------- #
def _post_process(vocals: np.ndarray, sr: int, normalize: bool, trim_db: float) -> np.ndarray:
    """High-pass, resample to TARGET_SR, trim leading/trailing silence, RMS-normalise."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    # remove sub-60 Hz rumble left over from the separation
    sos = butter(4, 60.0, btype="highpass", fs=sr, output="sos")
    vocals = sosfiltfilt(sos, vocals).astype(np.float32)

    if sr != TARGET_SR:
        vocals = librosa.resample(vocals, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")

    if trim_db > 0:
        _, (st, en) = librosa.effects.trim(vocals, top_db=trim_db, frame_length=2048, hop_length=512)
        pad = int(0.10 * TARGET_SR)  # keep 100 ms of context each side
        vocals = vocals[max(0, st - pad) : min(len(vocals), en + pad)]

    if normalize:
        vocals = rms_normalize(vocals, target_rms=0.1)
    return vocals.astype(np.float32)


def reset_dataset_dirs() -> None:
    """Delete the previous clean audio and splits so the dataset is rebuilt from scratch."""
    for d in (CLEAN_DIR, SPLITS_DIR):
        if d.exists():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d)
            log.info(f"Deleted previous dataset folder {d} ({n} files)")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (SPLITS_DIR / split).mkdir(parents=True, exist_ok=True)


def separate_vocals(raw_files: list[Path], args) -> dict[str, Path]:
    """Run MelBand RoFormer (vocals, then denoise) on every raw file; return {stem: clean_wav_path}."""
    sep = common.RoFormerSeparator(vocal_model=args.separator_model,
                                   denoise_model=None if args.no_denoise else args.denoise_model, logger=log,
                                   overlap=args.sep_overlap)
    log.info(f"Vocal isolation: {len(raw_files)} files | vocal model '{args.separator_model}'"
             + ("" if args.no_denoise else f" -> denoise model '{args.denoise_model}'"))
    t0 = time.time()
    n_stage = 1 if args.no_denoise else 2

    def progress(stage, i, n, name):
        if i % 10 == 0 or i == n:
            el = time.time() - t0
            log.info(f"  [{stage} {i}/{n}] {name}  elapsed {el / 60:.1f} min")

    audios = sep.process_files({f.stem: f for f in raw_files}, progress=progress)
    sep.release()
    log.info(f"RoFormer processing ({n_stage} stage(s)) finished in {(time.time() - t0) / 60:.1f} min")

    out: dict[str, Path] = {}
    failures = [f.stem for f in raw_files if f.stem not in audios]
    for stem, audio in audios.items():
        dst = CLEAN_DIR / f"{stem}.wav"
        try:
            vocals = _post_process(audio, common.SEP_SR, normalize=not args.no_normalize, trim_db=args.trim_db)
            assert_audio_ok(vocals, dst.name)
            save_wav(dst, vocals, TARGET_SR)
            out[stem] = dst
        except Exception as e:  # noqa: BLE001
            log.error(f"  post-processing FAILED for {stem}: {type(e).__name__}: {e}")
            failures.append(stem)
    if failures:
        log.warning(f"{len(failures)} file(s) failed separation and are excluded: {failures}")
    log.info(f"{len(out)} clean vocal files written to {CLEAN_DIR}")
    return out


# --------------------------------------------------------------------------- #
# 2. Deduplication via normalised log-mel fingerprints
# --------------------------------------------------------------------------- #
FP_SR = 16_000
FP_MELS = 64
FP_FRAMES = 128


def fingerprint(path: Path) -> tuple[np.ndarray, float]:
    """
    Time-normalised log-mel "image" of the clip (FP_MELS x FP_FRAMES), each
    band z-scored and the whole vector unit-normalised so that the dot product
    between two fingerprints equals their Pearson correlation
    (normalised mel-spectrogram cross-correlation at zero lag).
    """
    import librosa

    y, _ = load_audio(path, sr=FP_SR)
    duration = len(y) / FP_SR
    mel = librosa.feature.melspectrogram(y=y, sr=FP_SR, n_fft=1024, hop_length=256, n_mels=FP_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)  # [mels, t]
    # resample the time axis to a fixed number of frames (average pooling)
    idx = np.linspace(0, logmel.shape[1], FP_FRAMES + 1).astype(int)
    pooled = np.stack([logmel[:, a:b].mean(axis=1) if b > a else logmel[:, min(a, logmel.shape[1] - 1)]
                       for a, b in zip(idx[:-1], idx[1:])], axis=1)
    pooled = (pooled - pooled.mean(axis=1, keepdims=True)) / (pooled.std(axis=1, keepdims=True) + 1e-6)
    v = pooled.flatten().astype(np.float32)
    v = (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-8)
    return v, duration


XCORR_SR = 8_000


_XCORR_CACHE: dict[Path, np.ndarray] = {}


def _xcorr_signal(p: Path) -> np.ndarray:
    if p not in _XCORR_CACHE:
        y, _ = load_audio(p, sr=XCORR_SR)
        _XCORR_CACHE[p] = ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)
    return _XCORR_CACHE[p]


def waveform_xcorr(a: Path, b: Path) -> float:
    """
    Peak normalised cross-correlation between two waveforms (any lag).  Two
    encodes of the same recording score ~0.9-1.0; unrelated clips of the same
    speaker score < 0.1, so this is a sharp second opinion on the fingerprints.
    """
    from scipy.signal import fftconvolve

    ya, yb = _xcorr_signal(a), _xcorr_signal(b)
    c = fftconvolve(ya, yb[::-1], mode="full") / min(len(ya), len(yb))
    return float(c.max())


def deduplicate(clean: dict[str, Path], threshold: float, dur_tol: float,
                xcorr_threshold: float = 0.5) -> tuple[list[str], list[dict]]:
    """
    Return (kept_stems, duplicate_records).

    Stage 1: normalised mel-spectrogram cross-correlation > `threshold`  -> duplicate.
    Stage 2: every remaining pair that passes the duration gate is re-checked
             with waveform cross-correlation; > `xcorr_threshold` (0.5, which
             sits in the empty gap between ~0.07 for unrelated clips and ~0.9
             for re-encodes of one recording) also marks a duplicate.
    """
    stems = sorted(clean, key=lambda s: (len(s), s))  # numeric-ish order: keep the earliest file
    log.info(f"Deduplication: fingerprinting {len(stems)} clips ...")
    fps, durs = [], []
    for s in stems:
        v, d = fingerprint(clean[s])
        fps.append(v)
        durs.append(d)
    F = np.stack(fps)  # [n, D]
    D = np.asarray(durs)
    sim = F @ F.T  # Pearson correlation matrix
    np.fill_diagonal(sim, -1.0)

    # duration gate: two clips cannot be the same recording if lengths differ a lot
    dur_ok = np.abs(D[:, None] - D[None, :]) / np.maximum(D[:, None], D[None, :]) <= dur_tol
    sim_gated = np.where(dur_ok, sim, -1.0)

    tri = np.triu_indices(len(stems), 1)
    off = sim[tri]
    log.info(f"  pairwise mel-fingerprint similarity: max={off.max():.3f}  p99={np.percentile(off, 99):.3f}  "
             f"median={np.median(off):.3f}  (threshold {threshold})")

    # stage 1: fingerprint threshold; stage 2: waveform confirmation of the borderline band
    is_dup = sim_gated > threshold
    method = np.where(is_dup, 1, 0)  # 1 = mel fingerprint, 2 = waveform xcorr
    xcorr_val = np.full_like(sim, np.nan)
    band = [(a, b) for a, b in zip(*tri) if dur_ok[a, b] and sim_gated[a, b] <= threshold]
    if band:
        log.info(f"  re-checking {len(band)} duration-compatible pair(s) with waveform cross-correlation ...")
        t0 = time.time()
        for a, b in band:
            x = waveform_xcorr(clean[stems[a]], clean[stems[b]])
            xcorr_val[a, b] = xcorr_val[b, a] = x
            if x > xcorr_threshold:
                is_dup[a, b] = is_dup[b, a] = True
                method[a, b] = method[b, a] = 2

    kept: list[str] = []
    kept_idx: list[int] = []
    dups: list[dict] = []
    for i, s in enumerate(stems):
        match = [j for j in kept_idx if is_dup[i, j]]
        if match:
            j = max(match, key=lambda j: sim_gated[i, j])
            how = "mel-fingerprint" if method[i, j] == 1 else "waveform-xcorr"
            rec = {"duplicate": s, "kept": stems[j], "mel_similarity": round(float(sim_gated[i, j]), 4),
                   "waveform_xcorr": None if np.isnan(xcorr_val[i, j]) else round(float(xcorr_val[i, j]), 4),
                   "method": how, "duration_dup": round(float(D[i]), 2), "duration_kept": round(float(D[j]), 2)}
            dups.append(rec)
            log.info(f"  DUPLICATE {s} ~ {rec['kept']}  mel-sim={rec['mel_similarity']:.4f}"
                     + (f"  xcorr={rec['waveform_xcorr']:.3f}" if rec["waveform_xcorr"] is not None else "")
                     + f"  [{how}]  ({rec['duration_dup']}s vs {rec['duration_kept']}s)")
            continue
        kept.append(s)
        kept_idx.append(i)

    if band:
        xs = np.array([xcorr_val[a, b] for a, b in band])
        log.info(f"  waveform xcorr over {len(band)} pairs in {time.time() - t0:.0f}s: "
                 f"non-duplicates max={xs[xs <= xcorr_threshold].max() if (xs <= xcorr_threshold).any() else 0:.3f}, "
                 f"duplicates min={xs[xs > xcorr_threshold].min() if (xs > xcorr_threshold).any() else 0:.3f} "
                 f"(threshold {xcorr_threshold})")
    _XCORR_CACHE.clear()
    n_fp = sum(d["method"] == "mel-fingerprint" for d in dups)
    log.info(f"Deduplication: {len(dups)} duplicate(s) removed ({n_fp} by fingerprint, {len(dups) - n_fp} confirmed "
             f"by waveform xcorr), {len(kept)} unique clips remain")
    return kept, dups


# --------------------------------------------------------------------------- #
# 3. Split
# --------------------------------------------------------------------------- #
def stratified_split(rows: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    """
    80/10/10 split stratified by duration: sort by duration, walk in blocks of
    ten and hand one clip each to test and valid, the rest to train, so every
    split covers the same range of clip lengths.
    """
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["duration"])
    n = len(rows)
    n_test = max(1, round(n * ratios[2]))
    n_valid = max(1, round(n * ratios[1]))
    splits = {s: [] for s in SPLIT_NAMES}
    block = 10
    for start in range(0, n, block):
        chunk = rows[start : start + block]
        rng.shuffle(chunk)
        for r in chunk:
            if len(splits["test"]) < n_test and len(splits["test"]) * block <= start:
                splits["test"].append(r)
            elif len(splits["valid"]) < n_valid and len(splits["valid"]) * block <= start:
                splits["valid"].append(r)
            else:
                splits["train"].append(r)
    # top-up in case rounding left test/valid short
    while len(splits["test"]) < n_test and splits["train"]:
        splits["test"].append(splits["train"].pop(rng.randrange(len(splits["train"]))))
    while len(splits["valid"]) < n_valid and splits["train"]:
        splits["valid"].append(splits["train"].pop(rng.randrange(len(splits["train"]))))
    for s in splits.values():
        s.sort(key=lambda r: (len(Path(r["audio_path"]).stem), Path(r["audio_path"]).stem))
    return splits


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Isolate vocals, deduplicate and split the raw dataset.")
    p.add_argument("--raw_dir", type=Path, default=None, help="folder of raw audio (default: raw_data/ or data/raw_data/)")
    p.add_argument("--transcriptions", type=Path, default=None, help="CSV with (filename,text) columns")
    p.add_argument("--separator_model", default=common.SEP_VOCAL_MODEL,
                   help=f"audio-separator vocal model (RoFormer .ckpt); alternatives: {', '.join(common.SEP_ALT_MODELS)}")
    p.add_argument("--denoise_model", default=common.SEP_DENOISE_MODEL, help="second-pass MelBand RoFormer denoiser")
    p.add_argument("--no_denoise", action="store_true", help="skip the denoise pass (vocals stage only)")
    p.add_argument("--sep_overlap", type=int, default=8, help="RoFormer window overlap (2-8; lower = faster)")
    p.add_argument("--trim_db", type=float, default=40.0, help="silence trim threshold in dB (0 = off)")
    p.add_argument("--no_normalize", action="store_true", help="do not RMS-normalise cleaned vocals")
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--dup_duration_tol", type=float, default=0.15, help="max relative duration diff for duplicates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_separation", action="store_true",
                   help="keep data/clean_audios/ and only rebuild dedup + splits")
    return p.parse_args()


def main():
    args = parse_args()
    common.ensure_layout()
    common.ensure_ffmpeg(log)
    common.gpu_report(log, min_free_gb=0 if args.skip_separation else 3.0)

    raw_dir = args.raw_dir or common.raw_data_dir()
    csv_path = args.transcriptions or common.transcriptions_csv()
    raw_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in common.AUDIO_EXTS],
                       key=lambda f: (len(f.stem), f.stem))
    assert raw_files, f"No audio files found in {raw_dir}"
    log.info(f"raw_data: {raw_dir} ({len(raw_files)} files) | transcriptions: {csv_path}")

    # ---- transcripts ------------------------------------------------------ #
    transcripts_raw = common.load_transcriptions(csv_path)
    transcripts: dict[str, str] = {}
    corrupt: list[str] = []
    repaired: list[str] = []
    for stem, txt in transcripts_raw.items():
        fixed, is_corrupt = common.repair_text(txt)
        if is_corrupt:
            corrupt.append(stem)
        else:
            if fixed != txt.strip():
                repaired.append(stem)
            transcripts[stem] = fixed
    if repaired:
        log.info(f"Repaired encoding damage (apostrophes/quotes only) in {len(repaired)} transcript(s): {repaired}")
    if corrupt:
        log.warning(f"{len(corrupt)} transcript(s) contain unrecoverable text and are EXCLUDED: {corrupt}")
    raw_stems = {f.stem for f in raw_files}
    missing_audio = sorted(set(transcripts) - raw_stems)
    missing_text = sorted(raw_stems - set(transcripts_raw))
    if missing_audio:
        log.warning(f"{len(missing_audio)} transcript(s) without audio: {missing_audio[:20]}")
    if missing_text:
        log.warning(f"{len(missing_text)} audio file(s) without transcript (excluded): {missing_text[:20]}")

    # ---- 1. separation ---------------------------------------------------- #
    if args.skip_separation:
        clean = {p.stem: p for p in CLEAN_DIR.glob("*.wav")}
        assert clean, f"--skip_separation given but {CLEAN_DIR} is empty"
        log.info(f"Skipping separation, found {len(clean)} clean files")
    else:
        reset_dataset_dirs()
        clean = separate_vocals(raw_files, args)
    assert clean, "No clean audio produced"

    # sanity-check every clean file we intend to use
    usable = {}
    for stem, p in sorted(clean.items()):
        if stem not in transcripts:
            continue
        y, sr = load_audio(p)
        try:
            assert sr == TARGET_SR, f"{p.name}: sample rate {sr} != {TARGET_SR}"
            assert_audio_ok(y, p.name)
            usable[stem] = {"path": p, "duration": len(y) / sr}
        except AssertionError as e:
            log.error(f"  excluded: {e}")
    log.info(f"{len(usable)} clean clips with valid transcripts")

    # ---- 2. dedup ---------------------------------------------------------- #
    kept, dups = deduplicate({s: usable[s]["path"] for s in usable}, args.dup_threshold, args.dup_duration_tol)

    text_counts: dict[str, list[str]] = {}
    for s in kept:
        text_counts.setdefault(transcripts[s].lower(), []).append(s)
    same_text = {k: v for k, v in text_counts.items() if len(v) > 1}
    if same_text:
        log.info(f"{len(same_text)} transcript(s) are shared by several distinct recordings (kept): "
                 + "; ".join(",".join(v) for v in list(same_text.values())[:10]))

    # ---- 3. split ----------------------------------------------------------- #
    rows = [{"stem": s, "audio_path": usable[s]["path"], "text": transcripts[s], "duration": usable[s]["duration"]}
            for s in kept]
    splits = stratified_split(rows, args.seed)

    for split in SPLIT_NAMES:
        d = SPLITS_DIR / split
        for old in d.glob("*.wav"):
            old.unlink()
        out_rows = []
        for r in splits[split]:
            dst = d / f"{r['stem']}.wav"
            shutil.copy2(r["audio_path"], dst)
            out_rows.append({"audio_path": common.rel(dst), "text": r["text"], "duration": r["duration"]})
        p = common.write_split_csv(split, out_rows)
        tot = sum(r["duration"] for r in out_rows) / 60
        log.info(f"  {split:5s}: {len(out_rows):3d} clips, {tot:5.1f} min -> {p}")

    report = {
        "raw_files": len(raw_files),
        "transcripts": len(transcripts_raw),
        "corrupt_transcripts_excluded": corrupt,
        "repaired_transcripts": repaired,
        "audio_without_transcript": missing_text,
        "clean_files": len(clean),
        "duplicates_removed": dups,
        "duplicate_transcript_conflicts": [
            {"kept": d["kept"], "kept_text": transcripts.get(d["kept"]), "duplicate": d["duplicate"],
             "duplicate_text": transcripts.get(d["duplicate"])}
            for d in dups
            if transcripts.get(d["kept"], "").lower() != transcripts.get(d["duplicate"], "").lower()
        ],
        "unique_clips": len(kept),
        "shared_transcripts": same_text,
        "split_sizes": {s: len(splits[s]) for s in SPLIT_NAMES},
        "split_minutes": {s: round(sum(r["duration"] for r in splits[s]) / 60, 2) for s in SPLIT_NAMES},
        "seed": args.seed,
        "dup_threshold": args.dup_threshold,
        "separator": {"vocal_model": args.separator_model,
                      "denoise_model": None if args.no_denoise else args.denoise_model},
    }
    (SPLITS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"Dataset report -> {SPLITS_DIR / 'dataset_report.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
