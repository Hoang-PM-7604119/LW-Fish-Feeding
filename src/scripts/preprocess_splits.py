#!/usr/bin/env python3
"""
Preprocess video and audio files referenced in splits.json.

Instead of scanning an entire directory, this script reads the fixed splits
file and preprocesses only the clips that are actually used in training and
evaluation.  Preprocessed files are saved to output directories while
maintaining the same relative path structure as the raw dataset:

  raw:   video_dir/2022_6_23/AM_70/strong/23_video_37.mp4
  out:   out_video_dir/2022_6_23/AM_70/strong/23_video_37.pkl

  raw:   audio_dir/2022_6_23/AM_70/strong/23_audio_37.wav
  out:   out_audio_dir/2022_6_23/AM_70/strong/23_audio_37.npy

Usage
-----
python scripts/preprocess_splits.py \\
    --split_file   src/data/splits/splits.json \\
    --video_dir    /home/icclab/UFFIA_dataset/video_dataset \\
    --audio_dir    /home/icclab/UFFIA_dataset/audio_dataset \\
    --out_video    /home/icclab/UFFIA_dataset/preprocessed_video \\
    --out_audio    /home/icclab/UFFIA_dataset/preprocessed_audio \\
    --num_frames   16 \\
    --img_size     224 224 \\
    --sample_rate  32000 \\
    --duration     2.0 \\
    --num_workers  4
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional

import numpy as np
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _sample_video_uniform(
    video_path: str,
    num_frames: int,
    img_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Uniformly sample ``num_frames`` frames from *video_path*.

    Returns float32 array [T, H, W, C] in [0, 1], or None on failure.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frames.append(frame.astype(np.float32) / 255.0)
        else:
            if frames:
                frames.append(frames[-1])   # repeat last frame
            else:
                cap.release()
                return None
    cap.release()
    return np.stack(frames)   # [T, H, W, C]


def _preprocess_one_video(
    src: str,
    dst: str,
    num_frames: int,
    img_size: Tuple[int, int],
) -> Tuple[bool, str]:
    """Preprocess a single video file.  Returns (success, src)."""
    if os.path.exists(dst):
        return True, src          # already done

    frames = _sample_video_uniform(src, num_frames, img_size)
    if frames is None:
        return False, src

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    payload = {
        "video_form": frames,
        "metadata": {"num_frames": num_frames, "img_size": img_size},
    }
    with open(dst, "wb") as f:
        pickle.dump(payload, f)
    return True, src


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _preprocess_one_audio(
    src: str,
    dst: str,
    sample_rate: int,
    target_length: int,
) -> Tuple[bool, str]:
    """Preprocess a single audio file.  Returns (success, src)."""
    if os.path.exists(dst):
        return True, src

    try:
        waveform, _ = librosa.load(src, sr=sample_rate, mono=True)
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, waveform.astype(np.float32))
        return True, src
    except Exception:
        return False, src


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_unique_pairs(split_file: str):
    """Return lists of unique (video_rel, audio_rel) paths from splits.json."""
    with open(split_file) as f:
        data = json.load(f)

    video_rels, audio_rels = set(), set()
    for split_items in data["splits"].values():
        for item in split_items:
            video_rels.add(item["video_file"])
            audio_rels.add(item["audio_file"])
    return sorted(video_rels), sorted(audio_rels)


def preprocess_videos(
    video_rels,
    raw_video_dir: str,
    out_video_dir: str,
    num_frames: int,
    img_size: Tuple[int, int],
    num_workers: int,
):
    raw_video_dir = Path(raw_video_dir)
    out_video_dir = Path(out_video_dir)

    tasks = []
    for rel in video_rels:
        src = str(raw_video_dir / rel)
        # Replace extension: .mp4 → .pkl
        dst = str(out_video_dir / Path(rel).with_suffix(".pkl"))
        tasks.append((src, dst, num_frames, img_size))

    print(f"\nPreprocessing {len(tasks)} video clips → {out_video_dir}")
    success, failed = 0, []

    if num_workers <= 1:
        for src, dst, nf, sz in tqdm(tasks, desc="Video"):
            ok, path = _preprocess_one_video(src, dst, nf, sz)
            if ok:
                success += 1
            else:
                failed.append(path)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_preprocess_one_video, *t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Video"):
                ok, path = fut.result()
                if ok:
                    success += 1
                else:
                    failed.append(path)

    print(f"  Done: {success}/{len(tasks)} succeeded, {len(failed)} failed")
    if failed:
        for f in failed[:5]:
            print(f"    FAILED: {f}")


def preprocess_audios(
    audio_rels,
    raw_audio_dir: str,
    out_audio_dir: str,
    sample_rate: int,
    duration: float,
    num_workers: int,
):
    raw_audio_dir = Path(raw_audio_dir)
    out_audio_dir = Path(out_audio_dir)
    target_length = int(sample_rate * duration)

    tasks = []
    for rel in audio_rels:
        src = str(raw_audio_dir / rel)
        dst = str(out_audio_dir / Path(rel).with_suffix(".npy"))
        tasks.append((src, dst, sample_rate, target_length))

    print(f"\nPreprocessing {len(tasks)} audio clips → {out_audio_dir}")
    success, failed = 0, []

    if num_workers <= 1:
        for src, dst, sr, tl in tqdm(tasks, desc="Audio"):
            ok, path = _preprocess_one_audio(src, dst, sr, tl)
            if ok:
                success += 1
            else:
                failed.append(path)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_preprocess_one_audio, *t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Audio"):
                ok, path = fut.result()
                if ok:
                    success += 1
                else:
                    failed.append(path)

    print(f"  Done: {success}/{len(tasks)} succeeded, {len(failed)} failed")
    if failed:
        for f in failed[:5]:
            print(f"    FAILED: {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video/audio files referenced in splits.json"
    )
    parser.add_argument(
        "--split_file",
        default="/home/icclab/LW-Fish-Feeding/src/data/splits/splits.json",
        help="Path to splits.json"
    )
    parser.add_argument(
        "--video_dir",
        default="/home/icclab/UFFIA_dataset/video_dataset",
        help="Raw video directory"
    )
    parser.add_argument(
        "--audio_dir",
        default="/home/icclab/UFFIA_dataset/audio_dataset",
        help="Raw audio directory"
    )
    parser.add_argument(
        "--out_video",
        default="/home/icclab/UFFIA_dataset/preprocessed_video",
        help="Output directory for preprocessed video (.pkl)"
    )
    parser.add_argument(
        "--out_audio",
        default="/home/icclab/UFFIA_dataset/preprocessed_audio",
        help="Output directory for preprocessed audio (.npy)"
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        metavar=("H", "W"))
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel workers (0 or 1 = single-process)")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip video preprocessing")
    parser.add_argument("--skip_audio", action="store_true",
                        help="Skip audio preprocessing")
    args = parser.parse_args()

    print("=" * 70)
    print("Preprocessing files referenced in splits.json")
    print("=" * 70)
    print(f"  Split file : {args.split_file}")
    print(f"  Video in   : {args.video_dir}")
    print(f"  Video out  : {args.out_video}")
    print(f"  Audio in   : {args.audio_dir}")
    print(f"  Audio out  : {args.out_audio}")
    print(f"  Frames     : {args.num_frames}  Size: {args.img_size}")
    print(f"  Sample rate: {args.sample_rate}  Duration: {args.duration}s")
    print(f"  Workers    : {args.num_workers}")

    video_rels, audio_rels = collect_unique_pairs(args.split_file)
    print(f"\n  Unique video files: {len(video_rels)}")
    print(f"  Unique audio files: {len(audio_rels)}")

    if not args.skip_video:
        preprocess_videos(
            video_rels,
            args.video_dir,
            args.out_video,
            num_frames=args.num_frames,
            img_size=tuple(args.img_size),
            num_workers=args.num_workers,
        )

    if not args.skip_audio:
        preprocess_audios(
            audio_rels,
            args.audio_dir,
            args.out_audio,
            sample_rate=args.sample_rate,
            duration=args.duration,
            num_workers=args.num_workers,
        )

    print("\n" + "=" * 70)
    print("Done!  Update your config:")
    print(f"  data.video_dir: {args.out_video}")
    print(f"  data.audio_dir: {args.out_audio}")
    print("=" * 70)


if __name__ == "__main__":
    main()
