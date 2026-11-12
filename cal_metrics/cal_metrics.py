from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
from pesq import pesq
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_ROOT = SCRIPT_DIR.parent.parent / "cal_metrics"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
METRIC_FIELDS = (
    "PESQ_WB",
    "PESQ_NB",
    "CSIG",
    "CBAK",
    "COVL",
    "SSNR",
    "SI-SDR",
    "STOI",
    "DNSMOS_SIG",
    "DNSMOS_BAK",
    "DNSMOS_OVRL",
    "DNSMOS_P808",
    "NISQA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DANSE enhanced audio against clean references.")
    parser.add_argument("--clean_dir", required=True, help="Directory containing clean reference audio.")
    parser.add_argument("--enhanced_dir", required=True, help="Directory containing enhanced audio.")
    parser.add_argument("--metrics_root", default=str(DEFAULT_METRICS_ROOT), help="Directory containing DNSMOS and NISQA resources.")
    parser.add_argument("--sample_rate", default=16000, type=int, help="Evaluation sample rate (default: 16000).")
    parser.add_argument("--output_csv", default="metrics.csv", help="Per-file metrics CSV path (default: metrics.csv).")
    return parser.parse_args()


def find_audio_files(directory: Path) -> dict[Path, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    files = {
        path.relative_to(directory).with_suffix(".wav"): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    }
    if not files:
        raise FileNotFoundError(f"No supported audio files found in: {directory}")
    return files


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    return np.asarray(audio, dtype=np.float64)


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    reference_energy = max(float(np.dot(reference, reference)), 1e-12)
    target = np.dot(estimate, reference) * reference / reference_energy
    noise = estimate - target
    return 10 * math.log10(max(float(np.dot(target, target)), 1e-12) / max(float(np.dot(noise, noise)), 1e-12))


def require_finite(metrics: dict[str, float], audio_path: Path) -> None:
    invalid = {name: value for name, value in metrics.items() if not np.isfinite(value)}
    if invalid:
        raise ValueError(f"Non-finite metrics for {audio_path}: {invalid}")


def import_metric_backends(metrics_root: Path):
    stage2_root = metrics_root.parent.parent / "Stage2-AudioClassification"
    for directory in (metrics_root, stage2_root):
        directory_string = str(directory)
        if directory_string not in sys.path:
            sys.path.insert(0, directory_string)

    from compute_metrics import compute_metrics
    from dnsmos import calculate_dnsmos_yxc

    return compute_metrics, calculate_dnsmos_yxc


def run_nisqa(nisqa_root: Path, enhanced_dir: Path) -> dict[str, float]:
    predictor = nisqa_root / "run_predict.py"
    weights = nisqa_root / "weights" / "nisqa.tar"
    if not predictor.is_file() or not weights.is_file():
        raise FileNotFoundError(f"NISQA predictor or weights are missing under: {nisqa_root}")

    with tempfile.TemporaryDirectory(prefix="danse_nisqa_") as output_dir:
        subprocess.run(
            [
                sys.executable,
                str(predictor),
                "--mode",
                "predict_dir",
                "--pretrained_model",
                str(weights),
                "--data_dir",
                str(enhanced_dir),
                "--output_dir",
                output_dir,
                "--bs",
                "1",
            ],
            check=True,
        )
        result_path = Path(output_dir) / "NISQA_results.csv"
        if not result_path.is_file():
            raise FileNotFoundError(f"NISQA did not produce: {result_path}")
        import pandas as pd

        results = pd.read_csv(result_path)
    return {Path(row.deg).with_suffix(".wav").name: float(row.mos_pred) for row in results.itertuples()}


def main(args: argparse.Namespace) -> None:
    clean_dir = Path(args.clean_dir).expanduser().resolve()
    enhanced_dir = Path(args.enhanced_dir).expanduser().resolve()
    metrics_root = Path(args.metrics_root).expanduser().resolve()
    compute_metrics, calculate_dnsmos_yxc = import_metric_backends(metrics_root)

    clean_files = find_audio_files(clean_dir)
    enhanced_files = find_audio_files(enhanced_dir)
    common_paths = sorted(clean_files.keys() & enhanced_files.keys())
    if not common_paths:
        raise ValueError("No matching clean/enhanced audio pairs found by relative path.")
    missing_enhanced = sorted(clean_files.keys() - enhanced_files.keys())
    if missing_enhanced:
        print(f"Skipping {len(missing_enhanced)} clean file(s) without an enhanced counterpart.")

    nisqa_scores = run_nisqa(metrics_root / "NISQA-master", enhanced_dir)
    dnsmos_root = metrics_root / "DNSMOS"
    rows = []
    for relative_path in tqdm(common_paths, desc="Evaluating", unit="file"):
        clean_path = clean_files[relative_path]
        enhanced_path = enhanced_files[relative_path]
        clean = load_audio(clean_path, args.sample_rate)
        enhanced = load_audio(enhanced_path, args.sample_rate)
        length = min(clean.size, enhanced.size)
        if length < args.sample_rate // 4:
            raise ValueError(f"Audio pair is shorter than 0.25 seconds: {relative_path}")
        clean = clean[:length]
        enhanced = enhanced[:length]

        composite_input_clean = np.clip(clean, -1.0, 1.0) * 32767.0
        composite_input_enhanced = np.clip(enhanced, -1.0, 1.0) * 32767.0
        _, csig, cbak, covl, ssnr, stoi = compute_metrics(
            composite_input_clean, composite_input_enhanced, args.sample_rate, 0
        )
        dnsmos = calculate_dnsmos_yxc(str(enhanced_path), str(dnsmos_root), args.sample_rate)
        nisqa_key = enhanced_path.with_suffix(".wav").name
        if nisqa_key not in nisqa_scores:
            raise KeyError(f"NISQA result not found for: {enhanced_path}")

        metrics = {
            "PESQ_WB": pesq(args.sample_rate, clean, enhanced, "wb"),
            "PESQ_NB": pesq(args.sample_rate, clean, enhanced, "nb"),
            "CSIG": csig,
            "CBAK": cbak,
            "COVL": covl,
            "SSNR": ssnr,
            "SI-SDR": si_sdr(enhanced, clean),
            "STOI": stoi,
            "DNSMOS_SIG": dnsmos["SIG"],
            "DNSMOS_BAK": dnsmos["BAK"],
            "DNSMOS_OVRL": dnsmos["OVRL"],
            "DNSMOS_P808": dnsmos["P808_MOS"],
            "NISQA": nisqa_scores[nisqa_key],
        }
        require_finite(metrics, enhanced_path)
        rows.append({"wav_name": str(relative_path), **metrics})

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("wav_name", *METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    averages = {name: float(np.mean([row[name] for row in rows])) for name in METRIC_FIELDS}
    print(f"Evaluated {len(rows)} pair(s).")
    for name in METRIC_FIELDS:
        print(f"{name}: {averages[name]:.3f}")
    print(f"Saved metrics: {output_csv}")


if __name__ == "__main__":
    main(parse_args())
