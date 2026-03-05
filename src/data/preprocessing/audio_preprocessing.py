"""
Audio preprocessing utilities.

Converts raw .wav files into fixed-length, fixed-sample-rate arrays saved as .npy.
The output directory mirrors the input directory structure.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
from tqdm import tqdm

DEFAULT_RAW_DATA_ROOT = "/mnt/disk1/backup_user/hoang.pm/UFFIA_data"
DEFAULT_FIXED_DATA_ROOT = f"{DEFAULT_RAW_DATA_ROOT}/fixed"
DEFAULT_AUDIO_INPUT_DIR = f"{DEFAULT_RAW_DATA_ROOT}/audio_dataset"
DEFAULT_AUDIO_OUTPUT_DIR = f"{DEFAULT_FIXED_DATA_ROOT}/processed_audio"


def _pad_or_trim(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim waveform to a fixed length."""
    if len(waveform) < target_length:
        return np.pad(waveform, (0, target_length - len(waveform)), mode="constant")
    if len(waveform) > target_length:
        return waveform[:target_length]
    return waveform


def preprocess_audio_dataset(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 32000,
    duration: float = 2.0,
    extensions: Tuple[str, ...] = (".wav",)
):
    """
    Preprocess entire audio dataset by resampling and fixing duration.

    Args:
        input_dir: Input directory containing audio files
        output_dir: Output directory for preprocessed .npy files
        sample_rate: Target sample rate
        duration: Target duration in seconds
        extensions: Audio file extensions to process
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_length = int(sample_rate * duration)

    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.rglob(f"*{ext}"))

    print(f"\nPreprocessing {len(audio_files)} audio files...")
    print(f"Output directory: {output_dir}\n")

    success_count = 0
    failed_files = []

    for audio_file in tqdm(audio_files, desc="Processing audio"):
        rel_path = audio_file.relative_to(input_path)
        output_file = output_path / rel_path.with_suffix(".npy")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            success_count += 1
            continue

        try:
            waveform, _ = librosa.load(str(audio_file), sr=sample_rate, mono=True)
            waveform = _pad_or_trim(waveform, target_length).astype(np.float32)
            np.save(output_file, waveform)
            success_count += 1
        except Exception:
            failed_files.append(str(audio_file))

    print(f"\n{'='*80}")
    print("Audio Preprocessing Summary")
    print(f"{'='*80}")
    print(f"  Total audio files: {len(audio_files)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_AUDIO_INPUT_DIR,
        help="Input audio directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_AUDIO_OUTPUT_DIR,
        help="Output directory for .npy files"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=32000,
        help="Target sample rate"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Target duration in seconds"
    )

    args = parser.parse_args()

    preprocess_audio_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration=args.duration
    )
