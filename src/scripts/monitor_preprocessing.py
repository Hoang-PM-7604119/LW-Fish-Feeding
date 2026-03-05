#!/usr/bin/env python3
"""
Monitor preprocessing progress for video/audio outputs.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_RAW_ROOT = Path("/mnt/disk1/backup_user/hoang.pm/UFFIA_data")
DEFAULT_FIXED_ROOT = DEFAULT_RAW_ROOT / "fixed"

DEFAULT_RAW_VIDEO_DIR = DEFAULT_RAW_ROOT / "video_dataset"
DEFAULT_RAW_AUDIO_DIR = DEFAULT_RAW_ROOT / "audio_dataset"

DEFAULT_VIDEO_OUTPUTS = {
    "video_uniform": DEFAULT_FIXED_ROOT / "processed_video_uniform",
    "video_random": DEFAULT_FIXED_ROOT / "processed_video_random",
    "video_consecutive": DEFAULT_FIXED_ROOT / "processed_video_consecutive",
}
DEFAULT_AUDIO_OUTPUT = DEFAULT_FIXED_ROOT / "processed_audio"


def count_files(root: Path, extensions: Tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    count = 0
    for ext in extensions:
        count += sum(1 for _ in root.rglob(f"*{ext}"))
    return count


def get_progress(
    raw_video_dir: Path,
    raw_audio_dir: Path,
    video_outputs: Dict[str, Path],
    audio_output: Path,
) -> Dict[str, Dict[str, int]]:
    raw_video_count = count_files(raw_video_dir, (".mp4", ".avi", ".mov"))
    raw_audio_count = count_files(raw_audio_dir, (".wav",))

    progress = {
        "raw": {
            "videos": raw_video_count,
            "audios": raw_audio_count,
        }
    }

    for key, out_dir in video_outputs.items():
        progress[key] = {
            "processed": count_files(out_dir, (".pkl",)),
        }

    progress["audio"] = {
        "processed": count_files(audio_output, (".npy",)),
    }

    return progress


def print_report(progress: Dict[str, Dict[str, int]]) -> None:
    raw_videos = progress["raw"]["videos"]
    raw_audios = progress["raw"]["audios"]

    print("\nPreprocessing Progress")
    print("=" * 60)
    print(f"Raw videos: {raw_videos}")
    print(f"Raw audios: {raw_audios}")
    print("-" * 60)

    for key in ("video_uniform", "video_random", "video_consecutive"):
        processed = progress[key]["processed"]
        pct = (processed / raw_videos * 100) if raw_videos else 0.0
        print(f"{key:18s}: {processed:7d} ({pct:5.1f}%)")

    audio_processed = progress["audio"]["processed"]
    audio_pct = (audio_processed / raw_audios * 100) if raw_audios else 0.0
    print(f"{'audio':18s}: {audio_processed:7d} ({audio_pct:5.1f}%)")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor preprocessing progress.")
    parser.add_argument("--raw_video_dir", type=Path, default=DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument("--raw_audio_dir", type=Path, default=DEFAULT_RAW_AUDIO_DIR)
    parser.add_argument(
        "--video_uniform_dir", type=Path, default=DEFAULT_VIDEO_OUTPUTS["video_uniform"]
    )
    parser.add_argument(
        "--video_random_dir", type=Path, default=DEFAULT_VIDEO_OUTPUTS["video_random"]
    )
    parser.add_argument(
        "--video_consecutive_dir",
        type=Path,
        default=DEFAULT_VIDEO_OUTPUTS["video_consecutive"],
    )
    parser.add_argument("--audio_output_dir", type=Path, default=DEFAULT_AUDIO_OUTPUT)
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Refresh interval in seconds (0 = single report)",
    )

    args = parser.parse_args()

    video_outputs = {
        "video_uniform": args.video_uniform_dir,
        "video_random": args.video_random_dir,
        "video_consecutive": args.video_consecutive_dir,
    }

    if args.watch > 0:
        while True:
            progress = get_progress(
                args.raw_video_dir,
                args.raw_audio_dir,
                video_outputs,
                args.audio_output_dir,
            )
            print_report(progress)
            time.sleep(args.watch)
    else:
        progress = get_progress(
            args.raw_video_dir,
            args.raw_audio_dir,
            video_outputs,
            args.audio_output_dir,
        )
        print_report(progress)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
