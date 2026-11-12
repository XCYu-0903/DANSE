"""Create DANSE manifests from paired clean and noisy audio directories."""

import argparse
from pathlib import Path


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a DANSE training or validation manifest.")
    parser.add_argument("--clean_dir", required=True, help="Directory containing clean reference audio.")
    parser.add_argument("--noisy_dir", required=True, help="Directory containing paired noisy audio.")
    parser.add_argument("--output_file", required=True, help="Output manifest path.")
    return parser.parse_args()


def find_audio_files(directory):
    return {
        utterance_id(path): path
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    }


def utterance_id(path):
    return "_".join(path.stem.split("_")[-2:])


def main():
    args = parse_args()
    clean_files = find_audio_files(args.clean_dir)
    noisy_files = find_audio_files(args.noisy_dir)
    if not clean_files:
        raise FileNotFoundError(f"No supported audio files found in: {args.clean_dir}")

    missing_noisy = sorted(set(clean_files) - set(noisy_files))
    if missing_noisy:
        preview = ", ".join(missing_noisy[:5])
        raise ValueError(f"Missing noisy pairs for {len(missing_noisy)} clean files: {preview}")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    paired_ids = sorted(set(clean_files) & set(noisy_files))
    with output_file.open("w", encoding="utf-8") as file:
        for utterance_id in paired_ids:
            file.write(f"{utterance_id}|{clean_files[utterance_id]}\n")

    print(f"Wrote {len(paired_ids)} paired utterances to {output_file}")


if __name__ == "__main__":
    main()
