#!/usr/bin/env python
"""
Script 4 - inference.py

Zero-shot style voice cloning with the fine-tuned F5-TTS checkpoint:

    reference audio : data/inference_audio/sample/sample.wav
    reference text  : data/inference_audio/sample/sample_transcription.txt
    target text     : data/generated_audio/generation_transcription.txt
    weights         : checkpoints/best_model.pt   (fallback: last_model.pt)

    -> output/generated_audio/generated_voice.wav  (24 kHz, 16-bit PCM)
    -> output/generated_audio/generated_voice.mp3  (320 kbps)

Workflow:
  1. load sample.wav (+ transcript); if missing, fall back to a test clip cut to
     <= 12 s with its exact transcript via forced alignment
  2. denoise the reference with MelBand RoFormer (vocals stem -> denoise stem),
     the same chain used to build the training data  (--no_denoise to skip)
  3. synthesise the target text with the fine-tuned F5-TTS model

Strict output isolation: output/generated_audio/ receives ONLY generated_voice.wav
and generated_voice.mp3.  The reference prompt, the denoised reference, the
prompt+generation mel and any baseline comparison are never written there
(intermediates live under data/inference_audio/sample/, evaluation output under
output/evaluation/); the reference-conditioned frames are cropped from the
generated mel before vocoding and leading/trailing silence is trimmed.

Usage:
    python scripts/inference.py
    python scripts/inference.py --gen_text "Custom sentence to synthesise." --nfe_step 48 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CHECKPOINT_DIR, EVAL_GEN_DIR, GEN_TEXT_DIR, INFER_SAMPLE_DIR, OUTPUT_DIR, TARGET_SR  # noqa: E402

log = common.get_logger("inference")

SAMPLE_WAV = INFER_SAMPLE_DIR / "sample.wav"
SAMPLE_TXT = INFER_SAMPLE_DIR / "sample_transcription.txt"
GEN_TXT = GEN_TEXT_DIR / "generation_transcription.txt"
OUT_WAV = OUTPUT_DIR / "generated_voice.wav"
OUT_MP3 = OUTPUT_DIR / "generated_voice.mp3"

TEMPLATE_GEN_TEXT = (
    "Hello everyone, welcome back to my channel. Today I want to talk about something very important. "
    "It's not just a university, it is a brand. The people of this university, they are all very smart. "
    "So if you are working hard every day, one day you will be successful. Thank you very much, see you next time."
)

# F5-TTS configuration shared with finetune_tts.py
F5_ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False,
               conv_layers=4, pe_attn_head=1)


# --------------------------------------------------------------------------- #
def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        assert p.exists(), f"checkpoint not found: {p}"
        return p
    for name in ("best_model.pt", "last_model.pt", "f5tts_finetuned_alpha1.0.safetensors"):
        p = CHECKPOINT_DIR / name
        if p.exists():
            if name != "best_model.pt":
                log.warning(f"checkpoints/best_model.pt not found - using {name}")
            return p
    raise FileNotFoundError("No fine-tuned checkpoint in checkpoints/. Run scripts/finetune_tts.py first "
                            "(or pass --ckpt).")


def prepare_reference(aligner_device: str) -> tuple[Path, str]:
    """
    Make sure sample.wav + sample_transcription.txt exist and describe a
    <= 12 s prompt whose text matches the audio exactly.
    """
    if not SAMPLE_WAV.exists():
        vetted_wav, vetted_txt = EVAL_GEN_DIR / "prompts_fixed" / "enrol.wav", EVAL_GEN_DIR / "prompts_fixed" / "enrol.txt"
        if vetted_wav.exists() and vetted_txt.exists():
            # Reuse evaluation.py's build_fixed_prompt() selection: best DNSMOS of 21 valid-split
            # candidates, forced-aligned to an exact <= 12 s cut.  Far better than picking blind -
            # the naive fallback below once silently selected a clip this project's own dedup step
            # had flagged for an ambiguous/mis-heard transcript (duplicate_transcript_conflicts in
            # dataset_report.json), which imprinted that ambiguity into every clone.
            audio, sr = common.load_audio(vetted_wav, sr=TARGET_SR)
            common.save_wav(SAMPLE_WAV, audio, sr)
            SAMPLE_TXT.write_text(vetted_txt.read_text(encoding="utf-8"), encoding="utf-8")
            log.info(f"sample.wav missing -> reused the DNSMOS-vetted evaluation prompt {vetted_wav}")
        else:
            rows = common.read_split_csv("test")
            # exclude clips this project's own dedup flagged as having an ambiguous/disputed transcript
            flagged = set()
            report_p = common.PROJECT_ROOT / "data" / "splits" / "dataset_report.json"
            if report_p.exists():
                import json
                flagged = {c["kept"] for c in json.loads(report_p.read_text(encoding="utf-8")).get("duplicate_transcript_conflicts", [])}
            candidates = [r for r in rows if Path(r["audio_path"]).stem not in flagged] or rows
            # prefer a clip that already fits the 12 s prompt window, otherwise the shortest
            candidates = sorted(candidates, key=lambda r: (r["duration"] > common.REF_MAX_SEC, -r["duration"]))
            src = candidates[0]
            log.info(f"sample.wav missing -> using fallback test clip {src['audio_path']} ({src['duration']:.1f}s)")
            audio, sr = common.load_audio(common.abs_path(src["audio_path"]), sr=TARGET_SR)
            aligner = common.ForcedAligner(aligner_device) if src["duration"] > common.REF_MAX_SEC else None
            audio, text, exact = common.cut_reference_prompt(audio, sr, src["text"], aligner)
            common.save_wav(SAMPLE_WAV, audio, sr)
            SAMPLE_TXT.write_text(text, encoding="utf-8")
            log.info(f"wrote {SAMPLE_WAV} ({len(audio) / sr:.1f}s) and {SAMPLE_TXT} (exact text: {exact})")

    ref_text = SAMPLE_TXT.read_text(encoding="utf-8").strip() if SAMPLE_TXT.exists() else ""
    if not ref_text:
        log.warning(f"{SAMPLE_TXT} missing/empty - F5-TTS will transcribe the reference with Whisper "
                    "(idiosyncratic spellings may be normalised)")
        SAMPLE_TXT.write_text("", encoding="utf-8")

    audio, sr = common.load_audio(SAMPLE_WAV, sr=TARGET_SR)
    common.assert_audio_ok(audio, SAMPLE_WAV.name, min_sec=1.0)
    if len(audio) / sr > common.REF_MAX_SEC + 0.5 and ref_text:
        # F5-TTS would truncate the audio but keep the full text -> mismatch.  Cut both consistently.
        log.warning(f"reference is {len(audio) / sr:.1f}s (> {common.REF_MAX_SEC:.0f}s) - cutting a prompt with "
                    "forced alignment so audio and text stay in sync")
        cut_audio, cut_text, exact = common.cut_reference_prompt(audio, sr, ref_text, common.ForcedAligner(aligner_device))
        clipped = INFER_SAMPLE_DIR / "sample_clipped.wav"
        common.save_wav(clipped, cut_audio, sr)
        log.info(f"  -> {clipped} ({len(cut_audio) / sr:.1f}s), exact text: {exact}")
        return clipped, cut_text
    return SAMPLE_WAV, ref_text


def prepare_gen_text(cli_text: str | None) -> str:
    if cli_text:
        return cli_text.strip()
    if not GEN_TXT.exists() or not GEN_TXT.read_text(encoding="utf-8").strip():
        GEN_TXT.parent.mkdir(parents=True, exist_ok=True)
        GEN_TXT.write_text(TEMPLATE_GEN_TEXT, encoding="utf-8")
        log.info(f"generation text missing -> wrote template prompt to {GEN_TXT}")
    return GEN_TXT.read_text(encoding="utf-8").strip()


PRETRAINED_CKPT = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"  # same as finetune_tts.MODEL_CONFIGS


def _ema_state(ckpt_path: Path | str) -> dict:
    """
    EMA weights of a checkpoint as {param_name: fp32 tensor} (CFM naming, no 'ema_model.' prefix).

    Accepts both formats this project produces: the training checkpoint written by
    finetune_tts.py (a torch .pt holding `ema_model_state_dict`), and the EMA-only
    .safetensors published at huggingface.co/Rezuwan/AktarKhan_Weights - so a fresh clone can
    point straight at a downloaded release file without converting it first.
    """
    import torch

    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return {k: (v.float() if v.is_floating_point() else v) for k, v in load_file(str(ckpt_path)).items()}
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    sd = ck["ema_model_state_dict"]
    out = {}
    for k, v in sd.items():
        if k in ("initted", "step"):
            continue
        k = k.replace("ema_model.", "")
        if k in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
            continue  # non-trainable STFT buffers, dropped by f5_tts.load_checkpoint too
        out[k] = v.float() if torch.is_tensor(v) and v.is_floating_point() else v
    return out


def interpolate_weights(model, ckpt: Path, alpha: float, pretrained: str | None = None):
    """
    theta(alpha) = (1 - alpha) * theta_pretrained + alpha * theta_finetuned over the EMA weights.
    alpha=1 is the fine-tuned checkpoint as-is, alpha=0 the pretrained F5TTS_Base.  Blended in fp32
    on CPU, then cast to the model's dtype; every floating tensor must exist in both checkpoints.
    """
    import torch
    from cached_path import cached_path

    pre_path = pretrained or str(cached_path(PRETRAINED_CKPT))
    pre, ft = _ema_state(pre_path), _ema_state(ckpt)
    cur = model.state_dict()
    float_keys = [k for k, v in cur.items() if torch.is_tensor(v) and v.is_floating_point()]
    missing = [k for k in float_keys if k not in pre or k not in ft]
    assert not missing, f"cannot interpolate, {len(missing)} weight(s) missing from a checkpoint: {missing[:5]}"
    n_changed, sq = 0, 0.0
    for k in float_keys:
        assert pre[k].shape == ft[k].shape == cur[k].shape, f"{k}: shape mismatch {pre[k].shape}/{ft[k].shape}"
        blended = (1.0 - alpha) * pre[k] + alpha * ft[k]
        sq += float(((ft[k] - pre[k]) ** 2).sum())
        cur[k] = blended.to(dtype=cur[k].dtype, device=cur[k].device)
        n_changed += 1
    model.load_state_dict(cur)
    log.info(f"interpolated {n_changed} tensors at alpha={alpha:.2f} (||theta_ft - theta_pre||_2 = {sq ** 0.5:.2f}) "
             f"from {Path(pre_path).name} + {Path(ckpt).name}")
    return model


def load_tts(ckpt: Path, device: str, alpha: float | None = None, pretrained: str | None = None):
    from importlib.resources import files

    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    log.info(f"loading fine-tuned weights {ckpt} ({ckpt.stat().st_size / 1024**3:.2f} GB, EMA weights)")
    model = load_model(DiT, F5_ARCH, str(ckpt), mel_spec_type="vocos", vocab_file=vocab_file,
                       ode_method="euler", use_ema=True, device=device)
    if alpha is not None and alpha != 1.0:
        model = interpolate_weights(model, ckpt, alpha, pretrained)
    vocoder = load_vocoder("vocos", is_local=False, device=device)
    return model, vocoder


def synthesize(model, vocoder, ref_wav: Path, ref_text: str, gen_text: str, args, device: str) -> np.ndarray:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    from f5_tts.model.utils import seed_everything

    seed_everything(args.seed)
    ref_wav_proc, ref_text_proc = preprocess_ref_audio_text(str(ref_wav), ref_text, show_info=log.info)
    wav, sr, _ = infer_process(
        ref_wav_proc, ref_text_proc, gen_text, model, vocoder, mel_spec_type="vocos", show_info=log.info,
        nfe_step=args.nfe_step, cfg_strength=args.cfg_strength, sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed, cross_fade_duration=args.cross_fade, device=device,
    )
    assert wav is not None and sr == TARGET_SR, "synthesis returned nothing"
    wav = np.asarray(wav, dtype=np.float32)
    common.assert_audio_ok(wav, "generated audio", min_sec=0.5)
    return wav


def denoise_reference(ref_wav: Path, args) -> Path:
    """Run the MelBand RoFormer chain on the reference; returns the cleaned file (kept next to the sample)."""
    from scipy.signal import butter, sosfiltfilt

    sep = common.RoFormerSeparator(vocal_model=args.separator_model,
                                   denoise_model=None if args.no_denoise_pass else args.denoise_model, logger=log)
    t0 = time.time()
    audio = sep.process_file(ref_wav)
    sep.release()
    sos = butter(4, 60.0, btype="highpass", fs=common.SEP_SR, output="sos")
    audio = sosfiltfilt(sos, audio).astype(np.float32)
    import librosa

    audio = librosa.resample(audio, orig_sr=common.SEP_SR, target_sr=TARGET_SR, res_type="soxr_hq")
    audio = common.rms_normalize(audio, target_rms=0.1)
    common.assert_audio_ok(audio, "denoised reference", min_sec=1.0)
    out = INFER_SAMPLE_DIR / f"{ref_wav.stem}_denoised.wav"
    common.save_wav(out, audio, TARGET_SR)
    log.info(f"reference denoised with '{args.separator_model}'"
             + ("" if args.no_denoise_pass else f" + '{args.denoise_model}'")
             + f" in {time.time() - t0:.1f}s -> {out}")
    return out


def crop_generated(wav: np.ndarray, top_db: float = 45.0) -> np.ndarray:
    """
    Keep only the synthesised speech.  F5-TTS already drops the
    reference-conditioned frames from the generated mel (infer_process:
    `generated[:, ref_audio_len:]`), so the waveform never contains the prompt;
    here the leading/trailing silence around the new speech is trimmed too.
    """
    import librosa

    _, (st, en) = librosa.effects.trim(wav, top_db=top_db, frame_length=2048, hop_length=512)
    pad = int(0.05 * TARGET_SR)
    cropped = wav[max(0, st - pad) : min(len(wav), en + pad)]
    if len(cropped) < int(0.5 * TARGET_SR):  # never let an over-eager trim destroy the output
        return wav
    return cropped


def enforce_output_isolation() -> None:
    """output/generated_audio/ may contain only the two deliverables."""
    legacy = OUTPUT_DIR / "eval_test_set"
    if legacy.exists():  # outputs of an older evaluation layout -> move under output/evaluation/
        import shutil

        common.EVAL_GEN_DIR.mkdir(parents=True, exist_ok=True)
        for child in legacy.iterdir():
            dst = common.EVAL_GEN_DIR / child.name
            if dst.exists():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
            else:
                shutil.move(str(child), str(dst))
        legacy.rmdir()
        log.info(f"moved legacy evaluation outputs {legacy} -> {common.EVAL_GEN_DIR}")
    for item in OUTPUT_DIR.iterdir():
        if item in (OUT_WAV, OUT_MP3):
            continue
        if item.is_file():
            item.unlink()
            log.info(f"removed non-deliverable file from output dir: {item.name}")
        else:
            log.warning(f"unexpected sub-folder in {OUTPUT_DIR}: {item.name} (left untouched)")


def export(wav: np.ndarray, remove_silence: bool) -> None:
    from pydub import AudioSegment

    common.save_wav(OUT_WAV, wav, TARGET_SR)
    if remove_silence:
        from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

        remove_silence_for_generated_wav(str(OUT_WAV))
    seg = AudioSegment.from_wav(str(OUT_WAV))
    # MP3 at 24 kHz is MPEG-2 Layer III, which caps at 160 kbps; 320 kbps needs an
    # MPEG-1 sample rate, so the MP3 copy is upsampled to 44.1 kHz (WAV stays 24 kHz).
    seg.set_frame_rate(44100).export(str(OUT_MP3), format="mp3", bitrate="320k")
    for p in (OUT_WAV, OUT_MP3):
        assert p.exists() and p.stat().st_size > 1000, f"{p} was not written correctly"
    enforce_output_isolation()


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Synthesise speech in the cloned voice with the fine-tuned F5-TTS model.")
    p.add_argument("--ckpt", default=None, help="checkpoint (default: checkpoints/best_model.pt)")
    p.add_argument("--alpha", type=float, default=None,
                   help="weight interpolation: (1-alpha)*pretrained + alpha*fine-tuned over the EMA weights "
                        "(1.0 = fine-tuned checkpoint as-is, 0.0 = zero-shot F5TTS_Base). See evaluation.py --alphas")
    p.add_argument("--pretrain", default=None, help="local pretrained checkpoint for --alpha (default: HF hub cache)")
    p.add_argument("--gen_text", default=None, help="text to synthesise (default: generation_transcription.txt)")
    p.add_argument("--nfe_step", type=int, default=32, help="ODE steps (16-64; more = higher fidelity)")
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    p.add_argument("--speed", type=float, default=1.0, help="<1 slower / >1 faster; 1.0 keeps the reference cadence")
    p.add_argument("--cross_fade", type=float, default=0.15, help="cross-fade between text chunks (s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--remove_silence", action="store_true", help="strip long silences from the output")
    p.add_argument("--keep_edges", action="store_true", help="do not trim leading/trailing silence of the output")
    p.add_argument("--no_denoise", action="store_true", help="feed the raw reference to the model (skip RoFormer)")
    p.add_argument("--separator_model", default=common.SEP_VOCAL_MODEL, help="RoFormer vocal model for the reference")
    p.add_argument("--denoise_model", default=common.SEP_DENOISE_MODEL, help="second-pass MelBand RoFormer denoiser")
    p.add_argument("--no_denoise_pass", action="store_true", help="vocal-isolation pass only (skip the denoiser)")
    p.add_argument("--skip_cleanup", action="store_true", help="skip the pre-run disk sweep (common.cleanup_cache) at startup")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    common.ensure_layout()
    common.run_cleanup(log, skip=args.skip_cleanup, require_gb=2)
    common.ensure_ffmpeg(log)
    device = args.device or common.gpu_report(log, min_free_gb=3.0)

    ckpt = resolve_checkpoint(args.ckpt)
    ref_wav, ref_text = prepare_reference(device)  # 1. load / prepare sample.wav + transcript
    if not args.no_denoise:
        ref_wav = denoise_reference(ref_wav, args)  # 2. MelBand RoFormer cleaning of the reference
    gen_text = prepare_gen_text(args.gen_text)
    log.info(f"reference: {ref_wav} | ref text: {ref_text[:100]!r}")
    log.info(f"target text ({len(gen_text)} chars): {gen_text[:200]!r}")

    model, vocoder = load_tts(ckpt, device, alpha=args.alpha, pretrained=args.pretrain)  # 3. fine-tuned F5-TTS
    wav = synthesize(model, vocoder, ref_wav, ref_text, gen_text, args, device)
    raw_len = len(wav)
    if not args.keep_edges:
        wav = crop_generated(wav)
    log.info(f"generated speech only (prompt frames cropped by F5-TTS): {raw_len / TARGET_SR:.2f}s"
             + ("" if args.keep_edges else f" -> {len(wav) / TARGET_SR:.2f}s after edge-silence trim"))
    export(wav, args.remove_silence)

    print("\n=== Generated audio ===")
    for p in (OUT_WAV, OUT_MP3):
        print(f"  {p.resolve()}  ({common.human_size(p.stat().st_size)})")
    print(f"  duration: {len(wav) / TARGET_SR:.2f} s | seed {args.seed} | nfe {args.nfe_step} | "
          f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
