from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from dataset_dns import mag_pha_istft, mag_pha_stft
from env import AttrDict
from models.model import DANSE_Unimodal


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance speech and/or estimate noise with the DANSE model."
    )
    parser.add_argument(
        "--input_path",
        "--input",
        dest="input_path",
        required=True,
        help="Path to one audio file or a directory of audio files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="cp/danse_unimodal",
        help="Directory containing the DANSE checkpoint named g_best.",
    )
    parser.add_argument(
        "--output_dir",
        "--output",
        dest="output_dir",
        default="output",
        help="Directory for generated audio (default: output).",
    )
    parser.add_argument(
        "--signal",
        choices=("both", "speech", "noise"),
        default="both",
        help="Output type (default: both).",
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.json"),
        help="Model config. Defaults to config.json beside this script.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_config(config_path: Path) -> AttrDict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        return AttrDict(json.load(config_file))


def load_model(config: AttrDict, checkpoint_dir: Path, device: torch.device) -> DANSE_Unimodal:
    checkpoint_path = checkpoint_dir / "g_best"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DANSE checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("generator", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    model = DANSE_Unimodal(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def find_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No supported audio files found in: {input_path}")
    return sorted(files)


def output_relative_path(audio_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(f"{audio_path.stem}.wav")
    return audio_path.relative_to(input_path).with_suffix(".wav")


def trim_or_pad(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    audio = audio.squeeze(0)
    if audio.numel() >= target_length:
        return audio[:target_length]
    return functional.pad(audio, (0, target_length - audio.numel()))


@torch.inference_mode()
def enhance(model: DANSE_Unimodal, waveform: torch.Tensor, config: AttrDict, signal: str) -> Dict[str, torch.Tensor]:
    norm_factor = torch.sqrt(
        waveform.numel() / torch.clamp(torch.sum(waveform.square()), min=1e-12)
    )
    normalized_waveform = (waveform * norm_factor).unsqueeze(0)
    noisy_amp, noisy_phase, _ = mag_pha_stft(
        normalized_waveform,
        config.n_fft,
        config.hop_size,
        config.win_size,
        config.compress_factor,
    )
    speech_amp, speech_phase, _, noise_complex = model(noisy_amp, noisy_phase)

    outputs: Dict[str, torch.Tensor] = {}
    if signal in ("both", "speech"):
        speech = mag_pha_istft(
            speech_amp,
            speech_phase,
            config.n_fft,
            config.hop_size,
            config.win_size,
            config.compress_factor,
        )
        outputs["speech"] = trim_or_pad(speech / norm_factor, waveform.numel())

    if signal in ("both", "noise"):
        noise_magnitude = torch.abs(torch.complex(noise_complex[..., 0], noise_complex[..., 1]))
        noise_phase = torch.angle(torch.complex(noise_complex[..., 0], noise_complex[..., 1]))
        noise = mag_pha_istft(
            noise_magnitude,
            noise_phase,
            config.n_fft,
            config.hop_size,
            config.win_size,
            config.compress_factor,
        )
        outputs["noise"] = trim_or_pad(noise / norm_factor, waveform.numel())

    return outputs


def load_audio(audio_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = librosa.load(audio_path, sr=None, mono=True)
    if source_rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=source_rate, target_sr=sample_rate)
        print(f"Resampled {audio_path.name}: {source_rate} Hz -> {sample_rate} Hz")
    if waveform.size == 0:
        raise ValueError(f"Input audio is empty: {audio_path}")
    return torch.from_numpy(np.asarray(waveform, dtype=np.float32))


def write_outputs(outputs: Dict[str, torch.Tensor], output_dir: Path, relative_path: Path, sample_rate: int) -> None:
    for output_type, waveform in outputs.items():
        output_path = output_dir / output_type / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = waveform.detach().cpu().numpy().astype(np.float32)
        sf.write(output_path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
        print(f"Saved {output_type}: {output_path}")


def run_inference(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    output_dir = resolve_path(args.output_dir)
    config = load_config(resolve_path(args.config))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = load_model(config, checkpoint_dir, device)
    audio_files = find_audio_files(input_path)
    print(f"Processing {len(audio_files)} file(s) on {device}.")

    for audio_path in tqdm(audio_files, unit="file"):
        waveform = load_audio(audio_path, config.sampling_rate)
        try:
            outputs = enhance(model, waveform.to(device), config, args.signal)
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            print(f"CUDA out of memory for {audio_path.name}; retrying on CPU.")
            model.to("cpu")
            outputs = enhance(model, waveform, config, args.signal)
            model.to(device)
            torch.cuda.empty_cache()
        write_outputs(outputs, output_dir, output_relative_path(audio_path, input_path), config.sampling_rate)


if __name__ == "__main__":
    run_inference(parse_args())
