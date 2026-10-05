"""
Shared utilities for the aktts F5-TTS voice-cloning pipeline.

Every pipeline script imports this module for:
  * canonical project paths (created on demand with exist_ok=True)
  * consistent logging (console + per-script log file under checkpoints/logs/)
  * ffmpeg discovery (system PATH -> winget install dir -> bundled imageio-ffmpeg)
  * GPU / VRAM reporting and headroom checks
  * transcription CSV loading with encoding-damage repair (pronunciation
    spellings are NEVER normalised - only byte-level mojibake is fixed)
  * audio loading / saving helpers with silence & corruption assertions
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean_audios"
SPLITS_DIR = DATA_DIR / "splits"
INFER_SAMPLE_DIR = DATA_DIR / "inference_audio" / "sample"
GEN_TEXT_DIR = DATA_DIR / "generated_audio"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"  # centralised: .log files, training metrics (csv/xlsx), convergence plots
OUTPUT_DIR = PROJECT_ROOT / "output" / "generated_audio"  # ONLY the final generated_voice.wav / .mp3 live here
EVAL_GEN_DIR = PROJECT_ROOT / "output" / "evaluation"  # test-set syntheses, prompts, baseline (never in OUTPUT_DIR)

SPLIT_NAMES = ("train", "valid", "test")
TARGET_SR = 24_000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}

# The task spec places raw audio / transcriptions at the project root; the
# workspace currently holds them under data/.  Both locations are accepted.
_RAW_DATA_CANDIDATES = (PROJECT_ROOT / "raw_data", DATA_DIR / "raw_data")
_TRANSCRIPT_CANDIDATES = (PROJECT_ROOT / "transcriptions.csv", DATA_DIR / "transcriptions.csv")


def ensure_layout() -> None:
    """Create every directory in the enforced project layout (exist_ok=True)."""
    dirs = [
        DATA_DIR,
        CLEAN_DIR,
        SPLITS_DIR,
        *(SPLITS_DIR / s for s in SPLIT_NAMES),
        INFER_SAMPLE_DIR,
        GEN_TEXT_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        OUTPUT_DIR,
    ]
    if not any(c.exists() for c in _RAW_DATA_CANDIDATES):
        dirs.append(_RAW_DATA_CANDIDATES[0])
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _first_existing(candidates, what: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"{what} not found. Looked in: " + ", ".join(str(c) for c in candidates))


def raw_data_dir() -> Path:
    return _first_existing(_RAW_DATA_CANDIDATES, "raw_data directory")


def transcriptions_csv() -> Path:
    return _first_existing(_TRANSCRIPT_CANDIDATES, "transcriptions.csv")


def rel(p: Path | str) -> str:
    """Project-relative POSIX path (what we store in metadata CSVs)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def abs_path(p: str | Path) -> Path:
    """Resolve a metadata path (relative to project root) to an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    # library warnings (torch / librosa / pydub ...) and uncaught exceptions also land in the .log
    logging.captureWarnings(True)
    wlog = logging.getLogger("py.warnings")
    if not wlog.handlers:
        wlog.addHandler(fh)
        wlog.setLevel(logging.WARNING)

    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
    return logger


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
def ensure_ffmpeg(logger: logging.Logger | None = None) -> str:
    """
    Locate an ffmpeg binary and make it visible to pydub / ffmpeg-python /
    whisper / torchaudio by prepending its directory to PATH.

    Search order: PATH -> winget "Links" dir -> bundled imageio-ffmpeg binary.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:  # winget install (Gyan.FFmpeg) not yet on this shell's PATH
        winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
        found = sorted(winget.glob("Links/ffmpeg.exe")) + sorted(winget.glob("Packages/*FFmpeg*/**/bin/ffmpeg.exe"))
        exe = str(found[-1]) if found else None
    if exe is None:  # bundled static binary; tools such as whisper call `ffmpeg` by name, so expose that name
        try:
            import imageio_ffmpeg

            src = Path(imageio_ffmpeg.get_ffmpeg_exe())
            dst = Path(sys.prefix) / "Scripts" / "ffmpeg.exe"
            if not dst.exists():
                shutil.copy2(src, dst)
            exe = str(dst)
        except Exception:  # noqa: BLE001
            exe = None
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found. Install it (e.g. `winget install Gyan.FFmpeg` or "
            "`pip install imageio-ffmpeg`) or add it to PATH."
        )
    exe_dir = str(Path(exe).parent)
    if exe_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        from pydub import AudioSegment

        AudioSegment.converter = exe
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            AudioSegment.ffprobe = ffprobe
    except Exception:  # noqa: BLE001
        pass
    if logger:
        logger.info(f"ffmpeg: {exe}")
    return exe


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def cleanup_cache(logger: logging.Logger, require_gb: int = 0, dry_run: bool = False,
                  prune_eval: bool = False, aggressive: bool = False) -> float:
    """
    Disk hygiene before a training / evaluation / bulk-inference run.  Returns GB reclaimed.

    Idempotent and conservative: removes only regenerable junk and lists everything before
    deleting it.  Never touches data/raw_data/, data/clean_audios/, data/splits/, the retained
    checkpoint, or the logs/run<N>_*/ experiment archives.

      require_gb > 0   raise RuntimeError if less than this is free afterwards
      dry_run          list candidates, delete nothing
      prune_eval       also offer generated audio whose report is already archived
      aggressive       also offer re-downloadable framework caches (torch hub, pip)

    Ported from the former scripts/cleanup_cache.sh; same candidate rules and same output.
    """
    free_gb = lambda: shutil.disk_usage(PROJECT_ROOT).free / 1024**3  # noqa: E731
    before = free_gb()
    cand: list[tuple[int, Path]] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        path = Path(path)
        if not path.exists() or path in seen:
            return
        seen.add(path)
        cand.append((_dir_size(path), path))

    # 1. interrupted atomic saves and partial downloads
    for root in (CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT / "output"):
        if root.is_dir():
            for pat in ("*.tmp", "*.incomplete"):
                for f in root.rglob(pat):
                    add(f)
    hf = Path.home() / ".cache" / "huggingface"
    if hf.is_dir():
        for f in hf.rglob("*.incomplete"):
            add(f)

    # 2. python bytecode / test caches inside the repo (never under the venv)
    for pat in ("__pycache__", ".pytest_cache"):
        for d in PROJECT_ROOT.rglob(pat):
            if "aktts" not in d.parts:
                add(d)

    # 3. incomplete evaluation conditions: a condition dir holding fewer wavs than the test
    #    split has utterances is a smoke-test leftover, not a usable sweep.
    test_csv = SPLITS_DIR / "test.csv"
    if EVAL_GEN_DIR.is_dir() and test_csv.exists():
        n_test = max(len(test_csv.read_text(encoding="utf-8").splitlines()) - 1, 0)
        for d in sorted(EVAL_GEN_DIR.iterdir()):
            if not d.is_dir() or d.name in ("prompts", "prompts_fixed", "copysynth"):
                continue
            n = sum(1 for _ in d.rglob("*.wav"))
            if 0 < n < n_test:
                add(d)

    # 4. generated audio of conditions whose report is already archived (audio with no archived
    #    report is still the only copy of that result and is never offered here)
    if prune_eval and EVAL_GEN_DIR.is_dir():
        reports = [r for r in LOG_DIR.glob("run*/evaluation_*/evaluation_report.md")]
        archived = "\n".join(r.read_text(encoding="utf-8", errors="ignore") for r in reports)
        for d in sorted(EVAL_GEN_DIR.iterdir()):
            if d.is_dir() and d.name not in ("prompts", "prompts_fixed") and d.name in archived:
                add(d)

    # 5. re-downloadable framework caches
    if aggressive:
        add(Path.home() / ".cache" / "torch" / "hub")
        add(Path.home() / ".cache" / "pip")

    logger.info(f"cleanup | === disk before: {before:.0f} GB free ===")
    reclaimed = 0
    if not cand:
        logger.info("cleanup | nothing to reclaim (no tmp/partial files, no stale caches)")
    else:
        logger.info("cleanup | --- candidates ---")
        for size, path in sorted(cand, reverse=True):
            logger.info(f"cleanup | {size / 1048576:10.1f} MB  {path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path}")
        total = sum(size for size, _ in cand)
        logger.info(f"cleanup | --- total: {total / 1024**3:.2f} GB ---")
        if dry_run:
            logger.info("cleanup | (dry run: nothing deleted)")
        else:
            for _, path in cand:
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
            reclaimed = total

    after = free_gb()
    logger.info(f"cleanup | === disk after: {after:.0f} GB free (reclaimed {reclaimed / 1024**3:.2f} GB) ===")
    logger.info("cleanup | kept: data/ (raw, clean_audios, splits), checkpoints/*.pt, logs/run*/ archives,")
    logger.info("cleanup |       unarchived evaluation audio, ~/.cache/huggingface model blobs")
    if require_gb and after < require_gb:
        raise RuntimeError(f"only {after:.1f} GB free, below the {require_gb} GB required for this run. "
                           "Re-run with prune_eval/aggressive, or free space manually.")
    return reclaimed / 1024**3


def run_cleanup(logger: logging.Logger, skip: bool = False, require_gb: int = 0) -> None:
    """Standing pre-run hook for every training / evaluation / inference entry point."""
    if skip:
        logger.info("--skip_cleanup: not running the pre-run disk sweep")
        return
    cleanup_cache(logger, require_gb=require_gb)


def gpu_report(logger: logging.Logger, require_cuda: bool = False, min_free_gb: float = 0.0) -> str:
    """Log CUDA availability, GPU name, total & free VRAM; return device string."""
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA is NOT available - falling back to CPU (this will be very slow)."
        if require_cuda:
            raise RuntimeError(msg)
        logger.warning(msg)
        return "cpu"

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    logger.info(
        f"GPU: {props.name} | total VRAM {total_b / 1024**3:.2f} GiB | "
        f"free {free_b / 1024**3:.2f} GiB | torch {torch.__version__} | CUDA {torch.version.cuda}"
    )
    if min_free_gb and free_b / 1024**3 < min_free_gb:
        raise RuntimeError(
            f"Only {free_b / 1024**3:.2f} GiB VRAM free but {min_free_gb} GiB required. "
            "Close other GPU processes and retry."
        )
    return "cuda"


# --------------------------------------------------------------------------- #
# Transcriptions
# --------------------------------------------------------------------------- #
_REPLACEMENT = "�"
_MOJIBAKE_APOSTROPHE = re.compile(r"(?<=[A-Za-z])�+(?=[A-Za-z])")  # It<?><?><?>s -> It's
_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def repair_text(text: str) -> tuple[str, bool]:
    """
    Repair byte-level encoding damage WITHOUT touching spelling.

    * runs of U+FFFD between letters were a curly apostrophe (It's) -> "'"
    * curly quotes are mapped to straight quotes (F5-TTS does the same at
      inference time); no other characters are changed.

    Returns (text, is_corrupt) where is_corrupt is True when unrecoverable
    replacement characters remain (e.g. whole non-Latin sentences were lost).
    """
    text = text.replace("﻿", "").strip()
    text = _MOJIBAKE_APOSTROPHE.sub("'", text)
    text = text.translate(_QUOTE_MAP)
    text = re.sub(r"\s+", " ", text)
    return text, (_REPLACEMENT in text)


def load_transcriptions(path: Path | None = None) -> dict[str, str]:
    """
    Return {file_stem: text}.  Accepts headers (filename,text) or
    (Files,Transcriptions) - the first column is the file, the second the text.
    Filenames are matched by stem so `1.wav` in the CSV matches `1.mp3` on disk.
    """
    path = path or transcriptions_csv()
    # Decode as UTF-8; invalid bytes (encoding damage in the source file) become
    # U+FFFD so that repair_text() can either fix or flag them.  We deliberately
    # do NOT fall back to cp1252, which would silently turn damage into 'ý'.
    content = path.read_bytes().decode("utf-8-sig", errors="replace")

    rows = list(csv.reader(content.splitlines()))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"{path}: expected at least 2 columns (filename, text); got {header}")
    out: dict[str, str] = {}
    for r in rows[1:]:
        if len(r) < 2 or not r[0].strip():
            continue
        stem = Path(r[0].strip()).stem
        if stem in out:
            raise ValueError(f"{path}: duplicate filename entry '{r[0]}'")
        out[stem] = r[1]
    return out


def read_split_csv(split: str) -> list[dict]:
    """Read data/splits/<split>.csv (audio_path|text|duration)."""
    p = SPLITS_DIR / f"{split}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/build_clean_dataset.py first")
    rows = []
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            rows.append({"audio_path": r["audio_path"], "text": r["text"], "duration": float(r["duration"])})
    assert rows, f"{p} contains no rows"
    return rows


def write_split_csv(split: str, rows: list[dict]) -> Path:
    p = SPLITS_DIR / f"{split}.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["audio_path", "text", "duration"])
        for r in rows:
            w.writerow([r["audio_path"], r["text"], f"{r['duration']:.3f}"])
    return p


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def load_audio(path: Path | str, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32.  Returns (samples[channels, n] or [n] if mono, sr)."""
    import librosa
    import soundfile as sf

    path = str(path)
    try:
        data, file_sr = sf.read(path, dtype="float32", always_2d=True)  # [n, ch]
        data = data.T
    except Exception:  # noqa: BLE001  (formats libsndfile can't read -> ffmpeg via librosa/audioread)
        data, file_sr = librosa.load(path, sr=None, mono=False)
        data = np.atleast_2d(data).astype(np.float32)
    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    if sr is not None and file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
        file_sr = sr
    return (data[0] if mono else data), file_sr


def save_wav(path: Path | str, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Save 16-bit PCM WAV, clipping to [-1, 1]."""
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def assert_audio_ok(audio: np.ndarray, name: str, min_rms: float = 1e-4, min_sec: float = 0.3, sr: int = TARGET_SR):
    """Raise if audio is empty, NaN/Inf, silent or implausibly short."""
    assert audio.ndim == 1, f"{name}: expected mono 1-D audio, got shape {audio.shape}"
    assert audio.size >= int(min_sec * sr), f"{name}: too short ({audio.size / sr:.2f}s < {min_sec}s)"
    assert np.isfinite(audio).all(), f"{name}: contains NaN/Inf samples"
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > min_rms, f"{name}: audio is (near) silent, RMS={rms:.2e}"


def rms_normalize(audio: np.ndarray, target_rms: float = 0.1, peak: float = 0.99) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
    audio = audio * (target_rms / rms)
    mx = float(np.abs(audio).max() + 1e-9)
    if mx > peak:
        audio = audio * (peak / mx)
    return audio.astype(np.float32)


def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{n_bytes} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# Reference-prompt preparation (forced alignment)
# --------------------------------------------------------------------------- #
REF_MAX_SEC = 12.0  # F5-TTS clips reference audio to 12 s internally


class ForcedAligner:
    """
    Word-level forced alignment with torchaudio's MMS_FA (CTC) model.  Used to
    cut a <= 12 s reference prompt from a longer clip *together with the exact
    ground-truth words it contains*, so the prompt text never has to be
    re-transcribed by an ASR model that would normalise the speaker's
    idiosyncratic pronunciations.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        import torchaudio

        self.torch = torch
        self.bundle = torchaudio.pipelines.MMS_FA
        self.device = device
        self.model = self.bundle.get_model(with_star=True).to(device).eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.sr = self.bundle.sample_rate  # 16 kHz

    @staticmethod
    def _norm_word(w: str) -> str:
        w = re.sub(r"[^a-z']", "", w.lower().replace("’", "'"))
        return w if w else "*"  # digits / symbols -> <star> (matches anything)

    def align(self, audio: np.ndarray, sr: int, text: str) -> list[dict]:
        """Return [{word, start, end}] in seconds for every whitespace-delimited word of `text`."""
        import librosa

        words = text.split()
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr, res_type="soxr_hq")
        wav = self.torch.from_numpy(audio.astype(np.float32))[None].to(self.device)
        with self.torch.inference_mode():
            emission, _ = self.model(wav)
        tokens = self.tokenizer([self._norm_word(w) for w in words])
        spans = self.aligner(emission[0], tokens)
        ratio = wav.shape[1] / emission.shape[1] / self.sr  # seconds per emission frame
        return [{"word": w, "start": s[0].start * ratio, "end": s[-1].end * ratio} for w, s in zip(words, spans)]


def cut_reference_prompt(audio: np.ndarray, sr: int, text: str, aligner: "ForcedAligner | None",
                         max_sec: float = REF_MAX_SEC, min_sec: float = 3.0) -> tuple[np.ndarray, str, bool]:
    """
    Return (prompt_audio, prompt_text, exact).  If the clip already fits in
    `max_sec` it is returned whole.  Otherwise the longest word-prefix that ends
    before `max_sec` is cut (preferring a sentence boundary), and `exact` is
    True.  If alignment is unavailable/fails, the first `max_sec` seconds are
    returned with an empty text and exact=False (caller falls back to ASR).
    """
    dur = len(audio) / sr
    if dur <= max_sec:
        return audio, text, True
    if aligner is None:
        return audio[: int(max_sec * sr)], "", False
    try:
        words = aligner.align(audio, sr, text)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("common").warning(f"forced alignment failed ({e}); falling back to ASR reference text")
        return audio[: int(max_sec * sr)], "", False

    fit = [i for i, w in enumerate(words) if w["end"] + 0.15 <= max_sec]
    if not fit:
        return audio[: int(max_sec * sr)], "", False
    last = fit[-1]
    sentence_ends = [i for i in fit if words[i]["word"].rstrip('"\')').endswith((".", "!", "?", ",", ";", ":"))
                     and words[i]["end"] >= min_sec]
    if sentence_ends:
        last = sentence_ends[-1]
    end = words[last]["end"] + 0.15
    if last + 1 < len(words):
        end = min(end, words[last + 1]["start"])
    prompt = audio[: int(end * sr)]
    prompt_text = " ".join(w["word"] for w in words[: last + 1])
    return prompt, prompt_text, True


# --------------------------------------------------------------------------- #
# Vocal isolation / denoising with (Mel-Band) RoFormer via audio-separator
# --------------------------------------------------------------------------- #
SEP_VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"  # Kimberley Jensen MelBand RoFormer (Kijai's ComfyUI node), vocal SDR 12.6
SEP_DENOISE_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"  # MelBand RoFormer denoiser, 2nd pass
SEP_ALT_MODELS = ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "mel_band_roformer_kim_ft_unwa.ckpt")
SEP_SR = 44_100


class RoFormerSeparator:
    """
    Thin wrapper around `audio_separator.Separator` that keeps only the wanted
    stem, runs a whole file list through one checkpoint at a time (each model
    loads once), and returns/writes mono float32 audio.

        stage 1  vocal model    -> "vocals" stem (music / SFX removed)
        stage 2  denoise model  -> "dry" stem   (residual noise removed), optional
    """

    def __init__(self, vocal_model: str = SEP_VOCAL_MODEL, denoise_model: str | None = SEP_DENOISE_MODEL,
                 work_dir: Path | None = None, model_dir: Path | None = None, logger=None, overlap: int = 8):
        from audio_separator.separator import Separator

        self.log = logger or logging.getLogger("common")
        self.vocal_model, self.denoise_model = vocal_model, denoise_model
        self.work_dir = Path(work_dir or (DATA_DIR / "_separator_tmp"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir or (Path.home() / ".cache" / "audio-separator-models"))
        self.sep = Separator(log_level=logging.WARNING, model_file_dir=str(self.model_dir),
                             output_dir=str(self.work_dir), output_format="WAV", sample_rate=SEP_SR,
                             # overlap = number of overlapping RoFormer windows per chunk (quality vs. time; 8 = UVR default)
                             mdxc_params={"segment_size": 256, "overlap": overlap, "batch_size": 1, "pitch_shift": 0})
        self._loaded: str | None = None

    def _use(self, model_name: str) -> str:
        """Load `model_name` (once) and return its primary stem name."""
        if self._loaded != model_name:
            import time

            t0 = time.time()
            self.sep.load_model(model_filename=model_name)
            self._loaded = model_name
            self.log.info(f"separator model '{model_name}' ready ({time.time() - t0:.0f}s, "
                          f"stems: {self.sep.model_instance.primary_stem_name}/{self.sep.model_instance.secondary_stem_name})")
        stem = self.sep.model_instance.primary_stem_name
        # the loaded instance holds its own copy of this setting: write only the wanted stem
        self.sep.model_instance.output_single_stem = stem
        return stem

    def _run(self, model_name: str, src: Path, out_name: str) -> Path:
        stem = self._use(model_name)
        outs = self.sep.separate(str(src), custom_output_names={stem: out_name})
        assert len(outs) == 1, f"{model_name}: expected exactly the '{stem}' stem for {src.name}, got {outs}"
        out = self.work_dir / outs[0]
        assert out.exists(), f"separator output missing: {out}"
        return out

    def process_files(self, files: dict[str, Path], progress=None) -> dict[str, np.ndarray]:
        """
        {name: source_path} -> {name: mono float32 audio @ SEP_SR}.  Pass 1
        (vocals) over every file, then pass 2 (denoise) over every stage-1
        result, so each checkpoint is loaded once.  Failures are logged and skipped.
        """
        stage1: dict[str, Path] = {}
        for i, (name, src) in enumerate(files.items(), 1):
            try:
                stage1[name] = self._run(self.vocal_model, src, f"{name}-vocals")
            except Exception as e:  # noqa: BLE001
                self.log.error(f"  vocal separation FAILED for {src.name}: {type(e).__name__}: {e}")
            if progress:
                progress("vocals", i, len(files), name)
        final = dict(stage1)
        if self.denoise_model:
            for i, (name, voc) in enumerate(stage1.items(), 1):
                try:
                    final[name] = self._run(self.denoise_model, voc, f"{name}-dry")
                except Exception as e:  # noqa: BLE001
                    self.log.error(f"  denoise FAILED for {name} (keeping stage-1 vocals): {type(e).__name__}: {e}")
                if progress:
                    progress("denoise", i, len(stage1), name)
        out: dict[str, np.ndarray] = {}
        for name, p in final.items():
            audio, _ = load_audio(p, sr=SEP_SR, mono=True)
            out[name] = audio.astype(np.float32)
        self.cleanup()
        return out

    def process_file(self, src: Path) -> np.ndarray:
        return self.process_files({Path(src).stem: Path(src)})[Path(src).stem]

    def cleanup(self):
        for f in self.work_dir.glob("*.wav"):
            try:
                f.unlink()
            except OSError:
                pass

    def release(self):
        import gc

        import torch

        self.cleanup()
        self.sep.model_instance = None
        self._loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
