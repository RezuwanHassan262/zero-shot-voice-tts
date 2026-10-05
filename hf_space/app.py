"""
Gradio Space for a fine-tuned F5-TTS voice clone.

DISCLAIMER: This is an unofficial, non-commercial hobby project made purely for
educational and experimental purposes (learning how TTS fine-tuning works). It is
not affiliated with, endorsed by, or connected to the person whose voice it imitates,
and is made without any intention to harm anyone's reputation. All generated audio is
synthetic. The project will be taken down promptly on request.

Two checkpoints are bundled (see config.json for the full trade-off writeup):
  alpha=1.0  - default. WER matches the real speaker's own WER; SIM-o at the
               natural same-speaker ceiling.
  alpha=1.2  - more idiosyncratic-pronunciation retention, at a real WER cost.
Both are EMA-only weights, theta(alpha) = (1-alpha)*pretrained + alpha*finetuned,
extracted from a fine-tune of SWivid/F5-TTS F5TTS_Base.

The weight files themselves are not bundled in this Space's repo - they're hosted
separately at https://huggingface.co/Rezuwan/AktarKhan_Weights and downloaded
(and cached) on first use.

Output is delivered as MP3 (320 kbps, converted via ffmpeg after synthesis) rather
than the raw generated WAV, with an "AI-generated" note embedded in the MP3 metadata.
A playback-speed control (default 1.1x) is exposed and applied at synthesis time via
F5-TTS's own `speed` parameter - not as a post-hoc resample, so prosody/duration stay
consistent with the model's own timing.
"""

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import spaces
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-tuned checkpoints live in a separate model repo, not in this Space's own repo.
# config.json's weights[<key>]["file"] gives the filename inside that repo, e.g.
# "f5tts_finetuned_alpha1.0.safetensors" / "f5tts_finetuned_alpha1.2.safetensors".
WEIGHTS_REPO_ID = "Rezuwan/AktarKhan_Weights"
SPACE_ID = "Rezuwan/Aktar_Khan_TTS"
DEFAULT_SPEED = 1.1

# Embedded in every generated MP3's metadata so the file identifies itself as synthetic.
AI_METADATA_COMMENT = (
    "AI-generated synthetic speech. Unofficial, non-commercial, educational/experimental "
    f"project ({SPACE_ID}). Not the real person's voice and not endorsed by them."
)

DISCLAIMER_MD = f"""
> ⚠️ **Disclaimer — please read before using**
>
> - This is an **unofficial, non-commercial hobby project**, made **for educational, experimental and fun purposes only** — to learn how text-to-speech fine-tuning works.
> - It is **not affiliated with, endorsed by, or connected to** the person whose voice it imitates. **Every output is AI-generated** and does not represent anything that person has actually said, believes, or approves of.
> - It was made **without any intention to harm anyone's reputation**.
> - **Do not** use this to impersonate anyone, deceive or defraud people, spread misinformation, harass, defame, or create content presented as real. You are solely responsible for what you generate and how you use it.
> - **Takedown:** if you are the voice owner (or represent them) and want this removed, please open a post in the [Community tab](https://huggingface.co/spaces/{SPACE_ID}/discussions) and it will be taken down promptly.
>
> *এটি একটি অনানুষ্ঠানিক, অবাণিজ্যিক শিক্ষামূলক ও পরীক্ষামূলক প্রকল্প। সংশ্লিষ্ট ব্যক্তির সাথে এর কোনো সম্পর্ক বা অনুমোদন নেই, এবং কারো সম্মানহানির কোনো উদ্দেশ্য নেই। সব অডিও কৃত্রিমভাবে (AI দিয়ে) তৈরি।*
"""

ACK_LABEL = (
    "I understand this is an unofficial AI voice for educational/experimental use only, "
    "and I will not use the output to impersonate, deceive, defame or harm anyone."
)

vocab_char_map, vocab_size = get_tokenizer(str(ROOT / CFG["vocab_file"]), CFG["tokenizer"])

# ZeroGPU only attaches a GPU inside a function decorated with @spaces.GPU, so nothing
# that touches CUDA (vocoder, model .to(DEVICE)) can run at import time - it all has to
# be lazy-loaded the first time it's actually needed, inside such a function.
_vocoder = None
_models: dict[str, CFM] = {}
_weight_paths: dict[str, str] = {}

OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="aktts_out_"))


def get_vocoder():
    global _vocoder
    if _vocoder is None:
        _vocoder = load_vocoder(vocoder_name=CFG["mel_spec_kwargs"]["mel_spec_type"], is_local=False, device=DEVICE)
    return _vocoder


def resolve_weight_path(weight_key: str) -> str:
    """Download (and cache locally, keyed by revision hash) a checkpoint from
    WEIGHTS_REPO_ID the first time it's requested."""
    if weight_key not in _weight_paths:
        filename = CFG["weights"][weight_key]["file"]
        try:
            _weight_paths[weight_key] = hf_hub_download(repo_id=WEIGHTS_REPO_ID, filename=filename)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(
                f"Could not download '{filename}' from {WEIGHTS_REPO_ID}: {e}"
            ) from e
    return _weight_paths[weight_key]


def get_model(weight_key: str) -> CFM:
    if weight_key not in _models:
        model = CFM(
            transformer=DiT(**CFG["architecture"], text_num_embeds=vocab_size),
            mel_spec_kwargs=CFG["mel_spec_kwargs"],
            vocab_char_map=vocab_char_map,
        ).to(DEVICE)
        sd = load_file(resolve_weight_path(weight_key))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing, f"missing keys loading {weight_key}: {missing[:5]}"
        model.eval()
        _models[weight_key] = model
    return _models[weight_key]


REF_WAV = ROOT / "reference" / "voice_reference.wav"
REF_TXT = (ROOT / "reference" / "voice_reference.txt").read_text(encoding="utf-8").strip()
_ref_cache: dict = {}


def get_reference(custom_wav, custom_text):
    if custom_wav is not None:
        return preprocess_ref_audio_text(custom_wav, custom_text or "", show_info=lambda *_: None)
    if "default" not in _ref_cache:
        _ref_cache["default"] = preprocess_ref_audio_text(str(REF_WAV), REF_TXT, show_info=lambda *_: None)
    return _ref_cache["default"]


def _ffmpeg_bin() -> str:
    """Resolve an ffmpeg executable: system PATH first, falling back to the
    static binary bundled by imageio-ffmpeg, so this doesn't depend on the
    host image having ffmpeg installed via apt."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"ffmpeg is not available for MP3 export: {e}") from e


def wav_array_to_mp3(wav: np.ndarray, sr: int) -> str:
    """Write a generated waveform to a WAV temp file, convert to 320 kbps MP3
    via ffmpeg (tagging it as AI-generated in the ID3 metadata), and return the
    MP3 path. Each call gets a unique filename so concurrent requests on a shared
    Space instance don't collide."""
    stem = OUTPUT_DIR / f"ai_generated_{uuid.uuid4().hex}"
    wav_path = stem.with_suffix(".wav")
    mp3_path = stem.with_suffix(".mp3")

    sf.write(str(wav_path), wav, sr, subtype="PCM_16")

    cmd = [
        _ffmpeg_bin(), "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-ar", "44100", "-b:a", "320k",
        "-id3v2_version", "3",
        "-metadata", "title=AI-generated speech (synthetic)",
        "-metadata", f"comment={AI_METADATA_COMMENT}",
        "-metadata", f"publisher=https://huggingface.co/spaces/{SPACE_ID}",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    wav_path.unlink(missing_ok=True)
    if result.returncode != 0 or not mp3_path.exists():
        raise gr.Error(f"MP3 conversion failed: {result.stderr.strip()[:500]}")
    return str(mp3_path)


@spaces.GPU()
def synthesize(acknowledged, gen_text, weight_key, speed, nfe_step, cfg_strength, sway, seed, custom_wav, custom_text):
    if not acknowledged:
        raise gr.Error("Please read the disclaimer and tick the acknowledgement box before generating.")
    if not gen_text or not gen_text.strip():
        raise gr.Error("Enter some text to synthesise.")
    model = get_model(weight_key)
    vocoder = get_vocoder()
    ref_wav, ref_text = get_reference(custom_wav, custom_text)
    if seed is not None and seed >= 0:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    wav, sr, _ = infer_process(
        ref_wav, ref_text, gen_text.strip(), model, vocoder,
        mel_spec_type=CFG["mel_spec_kwargs"]["mel_spec_type"],
        nfe_step=int(nfe_step), cfg_strength=float(cfg_strength), sway_sampling_coef=float(sway),
        speed=float(speed), cross_fade_duration=0.15, show_info=lambda *_: None, progress=None, device=DEVICE,
    )
    mp3_path = wav_array_to_mp3(np.asarray(wav, dtype=np.float32), sr)
    return mp3_path


WEIGHT_CHOICES = [(f"{k}  —  {v['description']}", k) for k, v in CFG["weights"].items()]

with gr.Blocks(title="Unofficial F5-TTS Voice Clone (Educational / Experimental)") as demo:
    gr.Markdown(
        "# Unofficial fine-tuned F5-TTS voice clone\n"
        "*An educational / experimental project — not affiliated with or endorsed by the voice owner.*"
    )
    gr.Markdown(DISCLAIMER_MD)
    gr.Markdown(
        "Type text and generate **AI-generated** speech in the cloned voice. Two checkpoints are available, "
        "reflecting a validated retention-vs-intelligibility trade-off (see the model card / config.json).\n\n"
        f"Weights are downloaded on first use from "
        f"[{WEIGHTS_REPO_ID}](https://huggingface.co/{WEIGHTS_REPO_ID}). "
        "Output is delivered as MP3 and is tagged as AI-generated in its metadata."
    )
    with gr.Row():
        with gr.Column():
            acknowledged = gr.Checkbox(label=ACK_LABEL, value=False)
            gen_text = gr.Textbox(label="Text to synthesise", lines=4,
                                   placeholder="Type the sentence you want spoken in the cloned voice...")
            weight_key = gr.Radio(choices=WEIGHT_CHOICES, value="alpha_1.0", label="Model variant")
            speed = gr.Slider(0.5, 2.0, value=DEFAULT_SPEED, step=0.05,
                               label="Playback speed",
                               info="1.00 = model's natural pace. Default 1.10 (10% faster).")
            with gr.Accordion("Advanced: decoding parameters", open=False):
                nfe_step = gr.Slider(16, 64, value=CFG["recommended_decoding"]["nfe_step"], step=1, label="NFE steps")
                cfg_strength = gr.Slider(0.5, 4.0, value=CFG["recommended_decoding"]["cfg_strength"], step=0.1, label="CFG strength")
                sway = gr.Slider(-1.0, 1.0, value=CFG["recommended_decoding"]["sway_sampling_coef"], step=0.1, label="Sway sampling coefficient")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            with gr.Accordion("Advanced: use your own reference voice instead", open=False):
                gr.Markdown(
                    "Only upload **your own voice**, or a voice you have **explicit permission** to use."
                )
                custom_wav = gr.Audio(label="Reference audio (5-12s, clean)", type="filepath")
                custom_text = gr.Textbox(label="Exact transcript of the reference audio")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            out_audio = gr.Audio(label="Generated speech (MP3) — AI-generated, not a real recording", type="filepath")

    gr.Markdown(
        "---\n"
        "<small>For educational and experimental purposes only. No commercial use. "
        "Not affiliated with or endorsed by the voice owner. No intention to harm anyone's reputation. "
        f"Takedown requests: [Community tab](https://huggingface.co/spaces/{SPACE_ID}/discussions).</small>"
    )

    btn.click(
        synthesize,
        inputs=[acknowledged, gen_text, weight_key, speed, nfe_step, cfg_strength, sway, seed, custom_wav, custom_text],
        outputs=out_audio,
    )

if __name__ == "__main__":
    demo.queue().launch()