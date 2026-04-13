"""
Data splitting utilities for train/val/test splits.

This module provides utilities to:
1. Load fixed splits from JSON files (recommended for fair comparison)
2. Validate split files
3. Get split statistics
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np


DEFAULT_RAW_DATA_ROOT = "/mnt/disk1/backup_user/hoang.pm/UFFIA_data"
DEFAULT_FIXED_DATA_ROOT = f"{DEFAULT_RAW_DATA_ROOT}/fixed"
DEFAULT_VIDEO_DIR = f"{DEFAULT_FIXED_DATA_ROOT}/processed_video"
DEFAULT_AUDIO_DIR = f"{DEFAULT_FIXED_DATA_ROOT}/processed_audio"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_FIXED_DATA_ROOT}/splits"
DEFAULT_CLASS_NAMES = ["none", "weak", "medium", "strong"]


def load_splits(splits_file: str) -> Dict[str, Any]:
    """
    Load splits from JSON file.
    
    Args:
        splits_file: Path to splits JSON file (created by create_fixed_splits.py)
        
    Returns:
        split_data: Full split data including metadata and statistics
    """
    with open(splits_file, 'r') as f:
        split_data = json.load(f)
    
    # Handle both old format (just splits) and new format (with metadata)
    if 'splits' in split_data:
        splits = split_data['splits']
    else:
        splits = split_data
        split_data = {'splits': splits}
    
    print(f"✓ Loaded splits from {splits_file}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")
    
    if 'metadata' in split_data:
        meta = split_data['metadata']
        print(f"  Seed: {meta.get('seed', 'unknown')}")
        print(f"  Created: {meta.get('created_at', 'unknown')}")
    
    return split_data


def get_split_statistics(splits_file: str) -> Dict[str, Any]:
    """
    Get detailed statistics about a split file.
    
    Args:
        splits_file: Path to splits JSON file
        
    Returns:
        stats: Dictionary with split statistics
    """
    split_data = load_splits(splits_file)
    splits = split_data['splits']
    
    # Count per class
    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for split_name, items in splits.items():
        for item in items:
            if isinstance(item, dict):
                class_name = item.get('class', 'unknown')
            else:
                class_name = 'unknown'
            class_counts[class_name][split_name] += 1
    
    stats = {
        'total': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
            'all': len(splits['train']) + len(splits['val']) + len(splits['test'])
        },
        'per_class': dict(class_counts),
        'metadata': split_data.get('metadata', {})
    }
    
    return stats


def get_file_identifier(filepath: str, class_names: Optional[List[str]] = None) -> str:
    """Extract unique identifier from video/audio filename (including class)."""
    filename = os.path.basename(filepath)
    parent_dirs = os.path.dirname(filepath).split(os.sep)

    name = filename.replace("_video_", "_").replace("_audio_", "_")
    name = name.replace(".mp4", "").replace(".wav", "").replace(".pkl", "").replace(".npy", "")

    class_name = None
    if class_names:
        for p in parent_dirs:
            if p in class_names:
                class_name = p
                break

    date_part = None
    feed_part = None
    for p in parent_dirs:
        if p.startswith("2022_"):
            date_part = p
        if p.startswith("AM_") or p.startswith("PM_"):
            feed_part = p

    prefix_parts = []
    if class_name:
        prefix_parts.append(class_name)
    if date_part and feed_part:
        prefix_parts.append(f"{date_part}_{feed_part}")

    if prefix_parts:
        return f"{'_'.join(prefix_parts)}_{name}"
    return name


def find_matched_pairs(video_dir: str, audio_dir: str, class_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Find all video-audio pairs that exist in both directories."""
    matched_pairs = {}

    for class_name in class_names:
        video_paths = glob.glob(os.path.join(video_dir, "**", class_name, "*.mp4"), recursive=True)
        if not video_paths:
            video_paths = glob.glob(os.path.join(video_dir, "**", class_name, "*.pkl"), recursive=True)

        audio_paths = glob.glob(os.path.join(audio_dir, "**", class_name, "*.npy"), recursive=True)
        if not audio_paths:
            audio_paths = glob.glob(os.path.join(audio_dir, "**", class_name, "*.wav"), recursive=True)

        audio_dict = {}
        for audio_path in audio_paths:
            identifier = get_file_identifier(audio_path, class_names)
            if identifier in audio_dict and audio_dict[identifier].endswith(".npy"):
                continue
            audio_dict[identifier] = audio_path

        pairs = []
        for video_path in video_paths:
            identifier = get_file_identifier(video_path, class_names)
            if identifier in audio_dict:
                pairs.append({
                    "id": identifier,
                    "class": class_name,
                    "video_file": os.path.relpath(video_path, video_dir),
                    "audio_file": os.path.relpath(audio_dict[identifier], audio_dir),
                })

        matched_pairs[class_name] = pairs
        print(f"  {class_name:>8}: {len(pairs)} matched pairs "
              f"(video: {len(video_paths)}, audio: {len(audio_paths)})")

    return matched_pairs


def _extract_session(video_file: str) -> str:
    """Return session key (date/feeding_type) from a relative video file path."""
    parts = video_file.replace('\\', '/').split('/')
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]


def create_session_level_splits(
    matched_pairs: Dict[str, List[Dict[str, str]]],
    seed: int,
    val_target: int = 2800,
    test_target: int = 2800,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Split data at the feeding-session level to prevent data leakage and
    produce EXACTLY ``val_target`` and ``test_target`` clips.

    Algorithm
    ---------
    1. Group all clips by session (date + feeding type).
    2. Shuffle sessions with ``seed``.
    3. Greedily accumulate sessions into val and test (round-robin by deficit)
       until each pool reaches its target clip count.
    4. Subsample each pool DOWN to exactly the target.  Any overflow clips
       are moved to train so the total is preserved.
    5. Shuffle each final split.

    The only leakage risk is at the single "boundary" session for each eval
    split where a few clips may end up in train while the rest are in val/test.
    This is strictly minimal compared to clip-level random splitting.
    """
    sessions: Dict[str, List] = {}
    for class_name, pairs in matched_pairs.items():
        for pair in pairs:
            key = _extract_session(pair['video_file'])
            sessions.setdefault(key, []).append(pair)

    session_keys = sorted(sessions.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(session_keys)

    # Round-robin deficit assignment: whichever split is furthest below its
    # target gets the next session.
    val_pool: List = []
    test_pool: List = []
    val_total = test_total = 0

    for key in session_keys:
        if val_total >= val_target and test_total >= test_target:
            break
        val_deficit = val_target - val_total
        test_deficit = test_target - test_total
        pairs = sessions[key]
        if test_deficit >= val_deficit and test_deficit > 0:
            test_pool.extend(pairs)
            test_total += len(pairs)
        elif val_deficit > 0:
            val_pool.extend(pairs)
            val_total += len(pairs)

    # Subsample each pool to exactly the target; overflow → train
    def _exact(pool, target):
        if len(pool) <= target:
            return pool, []
        idx = rng.choice(len(pool), target, replace=False)
        idx_set = set(idx.tolist())
        return (
            [p for i, p in enumerate(pool) if i in idx_set],
            [p for i, p in enumerate(pool) if i not in idx_set],
        )

    val_exact, val_overflow = _exact(val_pool, val_target)
    test_exact, test_overflow = _exact(test_pool, test_target)

    # Build final splits
    train_pairs: List = []
    for key, pairs in sessions.items():
        pool_keys = {id(p) for p in val_pool + test_pool}
        if not any(id(p) in pool_keys for p in pairs):
            train_pairs.extend(pairs)
    train_pairs.extend(val_overflow)
    train_pairs.extend(test_overflow)

    splits: Dict[str, List] = {
        'train': train_pairs,
        'val': val_exact,
        'test': test_exact,
    }
    for split_name in splits:
        rng.shuffle(splits[split_name])

    n_val_s = len({_extract_session(p['video_file']) for p in val_exact})
    n_test_s = len({_extract_session(p['video_file']) for p in test_exact})
    n_train_s = len(sessions) - n_val_s - n_test_s
    print(f"\n  Session-level split: ~{n_train_s} train / ~{n_val_s} val / ~{n_test_s} test sessions")
    for split_name in ['train', 'val', 'test']:
        class_counts: Dict[str, int] = {}
        for p in splits[split_name]:
            c = p['class']
            class_counts[c] = class_counts.get(c, 0) + 1
        counts_str = "  ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
        print(f"  {split_name:>5}: {len(splits[split_name]):5d} clips  [{counts_str}]")

    return splits


def create_splits_by_count(
    matched_pairs: Dict[str, List[Dict[str, str]]],
    test_per_class: int,
    val_per_class: int,
    seed: int
) -> Dict[str, List[Dict[str, str]]]:
    """Create splits using fixed count per class."""
    random_state = np.random.RandomState(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name, pairs in matched_pairs.items():
        pairs = pairs.copy()
        random_state.shuffle(pairs)

        n_total = len(pairs)
        n_test = min(test_per_class, n_total // 3)
        n_val = min(val_per_class, (n_total - n_test) // 2)
        n_train = n_total - n_test - n_val

        splits["test"].extend(pairs[:n_test])
        splits["val"].extend(pairs[n_test:n_test + n_val])
        splits["train"].extend(pairs[n_test + n_val:])

        print(f"  {class_name:>8}: train={n_train}, val={n_val}, test={n_test}")

    for split_name in splits:
        random_state.shuffle(splits[split_name])

    return splits


def save_splits(splits: Dict[str, List[Dict[str, str]]], output_dir: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Save splits to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_data = {
        "metadata": metadata,
        "splits": {
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
        },
        "statistics": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
        },
    }

    class_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, items in splits.items():
        for item in items:
            class_stats[item["class"]][split_name] += 1
    split_data["statistics"]["per_class"] = dict(class_stats)

    split_file = output_path / "splits.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"\nSaved splits to: {split_file}")

    for split_name, items in splits.items():
        split_file = output_path / f"{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  Saved {split_name}.json ({len(items)} samples)")

    return split_data


def create_splits(
    video_dir: str = DEFAULT_VIDEO_DIR,
    audio_dir: str = DEFAULT_AUDIO_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    test_per_class: int = 700,
    val_per_class: int = 700
) -> Dict[str, Any]:
    """Create and save fixed train/val/test splits."""
    class_names = class_names or DEFAULT_CLASS_NAMES

    print("\n" + "=" * 80)
    print("Creating Fixed Train/Val/Test Splits")
    print("=" * 80)
    print(f"\nVideo directory: {video_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Classes: {class_names}")
    print(f"Val per class: {val_per_class}")
    print(f"Test per class: {test_per_class}")

    print("\n" + "-" * 40)
    print("Finding matched video-audio pairs...")
    print("-" * 40)

    matched_pairs = find_matched_pairs(video_dir, audio_dir, class_names)
    total_pairs = sum(len(pairs) for pairs in matched_pairs.values())
    print(f"\nTotal matched pairs: {total_pairs}")

    if total_pairs == 0:
        raise RuntimeError("No matched pairs found")

    print("\n" + "-" * 40)
    print("Creating splits...")
    print("-" * 40)

    splits = create_splits_by_count(matched_pairs, test_per_class, val_per_class, seed)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "class_names": class_names,
        "test_per_class": test_per_class,
        "val_per_class": val_per_class,
        "video_dir": os.path.abspath(video_dir),
        "audio_dir": os.path.abspath(audio_dir),
    }

    print("\n" + "-" * 40)
    print("Saving splits...")
    print("-" * 40)

    split_data = save_splits(splits, output_dir, metadata)

    print("\n" + "=" * 80)
    print("Split Summary")
    print("=" * 80)
    print(f"\nTotal samples: {split_data['statistics']['total']}")
    print(f"  Train: {split_data['statistics']['train']}")
    print(f"  Val:   {split_data['statistics']['val']}")
    print(f"  Test:  {split_data['statistics']['test']}")
    print("\n" + "=" * 80 + "\n")

    return split_data


def validate_splits(splits_file: str, video_dir: str, audio_dir: str) -> Dict[str, Any]:
    """
    Validate that all files in splits exist on disk.
    
    Args:
        splits_file: Path to splits JSON file
        video_dir: Base directory for video files
        audio_dir: Base directory for audio files
        
    Returns:
        validation_result: Dictionary with validation results
    """
    split_data = load_splits(splits_file)
    splits = split_data['splits']
    
    missing_video = []
    missing_audio = []
    valid_count = 0
    
    for split_name, items in splits.items():
        for item in items:
            if isinstance(item, dict):
                video_path = os.path.join(video_dir, item['video_file'])
                audio_path = os.path.join(audio_dir, item['audio_file'])
            else:
                continue
            
            video_exists = os.path.exists(video_path)
            audio_exists = os.path.exists(audio_path)
            
            if not video_exists:
                missing_video.append(video_path)
            if not audio_exists:
                missing_audio.append(audio_path)
            if video_exists and audio_exists:
                valid_count += 1
    
    total = sum(len(splits[s]) for s in splits)
    
    result = {
        'total_samples': total,
        'valid_samples': valid_count,
        'missing_video_count': len(missing_video),
        'missing_audio_count': len(missing_audio),
        'is_valid': len(missing_video) == 0 and len(missing_audio) == 0,
        'missing_video': missing_video[:10],  # First 10 only
        'missing_audio': missing_audio[:10]
    }
    
    if result['is_valid']:
        print(f"✓ All {valid_count} samples validated successfully")
    else:
        print(f"✗ Validation failed:")
        print(f"  Missing video files: {len(missing_video)}")
        print(f"  Missing audio files: {len(missing_audio)}")
    
    return result


def print_split_summary(splits_file: str):
    """Print a detailed summary of the splits."""
    stats = get_split_statistics(splits_file)
    
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    
    total = stats['total']
    print(f"\nTotal samples: {total['all']}")
    print(f"  Train: {total['train']} ({100*total['train']/total['all']:.1f}%)")
    print(f"  Val:   {total['val']} ({100*total['val']/total['all']:.1f}%)")
    print(f"  Test:  {total['test']} ({100*total['test']/total['all']:.1f}%)")
    
    print("\nPer-class breakdown:")
    print("-" * 60)
    print(f"{'Class':>10} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    for class_name, counts in sorted(stats['per_class'].items()):
        class_total = counts['train'] + counts['val'] + counts['test']
        print(f"{class_name:>10} {counts['train']:>8} {counts['val']:>8} "
              f"{counts['test']:>8} {class_total:>8}")
    
    print("-" * 60)
    
    if stats['metadata']:
        print("\nMetadata:")
        for key, value in stats['metadata'].items():
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    """Split file utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Split file utilities")
    parser.add_argument("--create", action="store_true", help="Create fixed train/val/test splits")
    parser.add_argument("--splits_file", type=str, help="Path to splits.json")
    parser.add_argument("--validate", action="store_true", help="Validate files exist")
    parser.add_argument("--video_dir", type=str, default=DEFAULT_VIDEO_DIR, help="Video directory")
    parser.add_argument("--audio_dir", type=str, default=DEFAULT_AUDIO_DIR, help="Audio directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class_names", type=str, nargs="+", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--test_per_class", type=int, default=700)
    parser.add_argument("--val_per_class", type=int, default=700)

    args = parser.parse_args()

    if args.create:
        create_splits(
            video_dir=args.video_dir,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            class_names=args.class_names,
            test_per_class=args.test_per_class,
            val_per_class=args.val_per_class
        )
    elif args.splits_file:
        print_split_summary(args.splits_file)
        if args.validate:
            validate_splits(args.splits_file, args.video_dir, args.audio_dir)
    else:
        parser.print_help()
