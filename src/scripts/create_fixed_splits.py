#!/usr/bin/env python
"""
Create Fixed Train/Val/Test Splits for Fair Experiment Comparison

This script creates and saves fixed data splits to ensure all experiments
use exactly the same train/val/test data, enabling fair comparison.

The splits are saved as JSON files containing file identifiers,
making them portable across different machines/setups.

Usage:
    python scripts/create_fixed_splits.py

Or with custom parameters:
    python scripts/create_fixed_splits.py \
        --video_dir /mnt/e/U-FFIA_data/data/video_dataset \
        --audio_dir /mnt/e/U-FFIA_data/data/audio_dataset \
        --output_dir /mnt/e/U-FFIA_data/data/splits \
        --seed 42 \
        --val_per_class 700 \
        --test_per_class 700
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default paths
DEFAULT_VIDEO_DIR = '/mnt/e/U-FFIA_data/data/video_dataset'
DEFAULT_AUDIO_DIR = '/mnt/e/U-FFIA_data/data/audio_dataset'
DEFAULT_OUTPUT_DIR = '/mnt/e/U-FFIA_data/data/splits'
DEFAULT_CLASS_NAMES = ['none', 'weak', 'medium', 'strong']


def get_file_identifier(filepath, include_feed=False):
    """Extract unique identifier from video/audio filename.
    include_feed: if False (default), match by date+name only so video in PM_100
    and audio in PM_70 for the same clip still pair.
    """
    filename = os.path.basename(filepath)
    parent_dirs = os.path.dirname(filepath).split(os.sep)
    
    name = filename.replace('_video_', '_').replace('_audio_', '_')
    name = name.replace('.mp4', '').replace('.wav', '').replace('.pkl', '').replace('.npy', '')
    
    date_part = None
    feed_part = None
    for p in parent_dirs:
        if p.startswith('2022_'):
            date_part = p
        if p.startswith('AM_') or p.startswith('PM_'):
            feed_part = p
    
    if date_part and (include_feed and feed_part):
        return f"{date_part}_{feed_part}_{name}"
    if date_part:
        return f"{date_part}_{name}"
    return name


def find_matched_pairs(video_dir, audio_dir, class_names):
    """Find all video-audio pairs that exist in both directories."""
    matched_pairs = {}
    video_dir = os.path.abspath(video_dir)
    audio_dir = os.path.abspath(audio_dir)

    if not os.path.isdir(video_dir):
        print(f"  WARNING: video_dir does not exist or is not a directory: {video_dir}")
    if not os.path.isdir(audio_dir):
        print(f"  WARNING: audio_dir does not exist or is not a directory: {audio_dir}")

    for class_name in class_names:
        # Nested: video_dir/.../class_name/*.mp4 or *.pkl
        video_paths = glob.glob(
            os.path.join(video_dir, '**', class_name, '*.mp4'),
            recursive=True
        )
        if not video_paths:
            video_paths = glob.glob(
                os.path.join(video_dir, '**', class_name, '*.pkl'),
                recursive=True
            )
        # Flat: video_dir/class_name/*.mp4 or *.pkl (e.g. processed_video/none/*.pkl)
        if not video_paths:
            video_paths = glob.glob(os.path.join(video_dir, class_name, '*.mp4'))
        if not video_paths:
            video_paths = glob.glob(os.path.join(video_dir, class_name, '*.pkl'))

        audio_paths = glob.glob(
            os.path.join(audio_dir, '**', class_name, '*.wav'),
            recursive=True
        )
        if not audio_paths:
            audio_paths = glob.glob(
                os.path.join(audio_dir, '**', class_name, '*.npy'),
                recursive=True
            )
        if not audio_paths:
            audio_paths = glob.glob(os.path.join(audio_dir, class_name, '*.wav'))
        if not audio_paths:
            audio_paths = glob.glob(os.path.join(audio_dir, class_name, '*.npy'))

        # identifier -> list of audio paths (same date+name can appear in different feed folders)
        audio_by_id = {}
        for audio_path in audio_paths:
            identifier = get_file_identifier(audio_path)
            audio_by_id.setdefault(identifier, []).append(audio_path)
        
        pairs = []
        for video_path in video_paths:
            identifier = get_file_identifier(video_path)
            if identifier in audio_by_id and audio_by_id[identifier]:
                audio_path = audio_by_id[identifier].pop(0)
                pairs.append({
                    'id': identifier,
                    'class': class_name,
                    'video_file': os.path.relpath(video_path, video_dir),
                    'audio_file': os.path.relpath(audio_path, audio_dir)
                })
        
        matched_pairs[class_name] = pairs
        print(f"  {class_name:>8}: {len(pairs)} matched pairs "
              f"(video: {len(video_paths)}, audio: {len(audio_paths)})")
    
    return matched_pairs


def create_splits_by_count(matched_pairs, test_per_class, val_per_class, seed, ensure_disjoint_audio=True):
    """Create splits. If ensure_disjoint_audio=True (default), each audio path is in only one split (no leakage)."""
    random_state = np.random.RandomState(seed)
    splits = {'train': [], 'val': [], 'test': []}

    if ensure_disjoint_audio:
        audio_to_split = {}
        for class_name, pairs in matched_pairs.items():
            audio_by_class = list({p['audio_file'] for p in pairs})
            random_state.shuffle(audio_by_class)
            n = len(audio_by_class)
            n_test = min(test_per_class, max(0, n // 3))
            n_val = min(val_per_class, max(0, (n - n_test) // 2))
            n_train = n - n_test - n_val
            for i, audio in enumerate(audio_by_class):
                if i < n_test:
                    audio_to_split[audio] = 'test'
                elif i < n_test + n_val:
                    audio_to_split[audio] = 'val'
                else:
                    audio_to_split[audio] = 'train'
            print(f"  {class_name:>8}: train={n_train}, val={n_val}, test={n_test} (by unique audio)")

        for class_name, pairs in matched_pairs.items():
            for p in pairs:
                split_name = audio_to_split.get(p['audio_file'], 'train')
                splits[split_name].append(p)

        for split_name in splits:
            random_state.shuffle(splits[split_name])
    else:
        for class_name, pairs in matched_pairs.items():
            pairs = pairs.copy()
            random_state.shuffle(pairs)
            n_total = len(pairs)
            n_test = min(test_per_class, n_total // 3)
            n_val = min(val_per_class, (n_total - n_test) // 2)
            n_train = n_total - n_test - n_val
            splits['test'].extend(pairs[:n_test])
            splits['val'].extend(pairs[n_test:n_test + n_val])
            splits['train'].extend(pairs[n_test + n_val:])
            print(f"  {class_name:>8}: train={n_train}, val={n_val}, test={n_test}")
        for split_name in splits:
            random_state.shuffle(splits[split_name])

    return splits


def save_splits(splits, output_dir, metadata):
    """Save splits to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    split_data = {
        'metadata': metadata,
        'splits': {
            'train': splits['train'],
            'val': splits['val'],
            'test': splits['test']
        },
        'statistics': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
            'total': len(splits['train']) + len(splits['val']) + len(splits['test'])
        }
    }
    
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    for split_name, items in splits.items():
        for item in items:
            class_stats[item['class']][split_name] += 1
    split_data['statistics']['per_class'] = dict(class_stats)
    
    split_file = output_path / 'splits.json'
    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\nSaved splits to: {split_file}")
    
    for split_name, items in splits.items():
        split_file = output_path / f'{split_name}.json'
        with open(split_file, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"  Saved {split_name}.json ({len(items)} samples)")
    
    return split_data


def main():
    parser = argparse.ArgumentParser(description='Create fixed train/val/test splits')
    
    parser.add_argument('--video_dir', type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class_names', type=str, nargs='+', default=DEFAULT_CLASS_NAMES)
    parser.add_argument('--test_per_class', type=int, default=700)
    parser.add_argument('--val_per_class', type=int, default=700)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Creating Fixed Train/Val/Test Splits")
    print("=" * 80)
    print(f"\nVideo directory: {args.video_dir}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Classes: {args.class_names}")
    print(f"Val per class: {args.val_per_class}")
    print(f"Test per class: {args.test_per_class}")
    
    print("\n" + "-" * 40)
    print("Finding matched video-audio pairs...")
    print("-" * 40)
    
    matched_pairs = find_matched_pairs(args.video_dir, args.audio_dir, args.class_names)
    
    total_pairs = sum(len(pairs) for pairs in matched_pairs.values())
    print(f"\nTotal matched pairs: {total_pairs}")

    if total_pairs == 0:
        print("\nERROR: No matched pairs found!")
        print("\nDiagnostics:")
        for label, base in [("video_dir", args.video_dir), ("audio_dir", args.audio_dir)]:
            ab = os.path.abspath(base)
            if os.path.isdir(ab):
                try:
                    top = os.listdir(ab)[:15]
                    print(f"  {label} (first 15 items): {top}")
                except OSError as e:
                    print(f"  {label}: {e}")
            else:
                print(f"  {label}: not a directory or missing: {ab}")
                # If video_dir is missing, show parent so user can see actual folder names
                if label == "video_dir":
                    parent = os.path.dirname(ab)
                    if os.path.isdir(parent):
                        try:
                            siblings = os.listdir(parent)[:20]
                            print(f"  -> Parent directory exists. Contents: {siblings}")
                        except OSError:
                            pass
        print("\nExpected layout: video_dir/.../none|weak|medium|strong/*.mp4 or *.pkl, same for audio with .wav or .npy")
        print("If video_dir path is wrong, check the parent contents above and pass the correct --video_dir.")
        sys.exit(1)
    
    print("\n" + "-" * 40)
    print("Creating splits...")
    print("-" * 40)
    
    splits = create_splits_by_count(
        matched_pairs,
        args.test_per_class,
        args.val_per_class,
        args.seed
    )
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'seed': args.seed,
        'class_names': args.class_names,
        'test_per_class': args.test_per_class,
        'val_per_class': args.val_per_class,
        'video_dir': os.path.abspath(args.video_dir),
        'audio_dir': os.path.abspath(args.audio_dir),
    }
    
    print("\n" + "-" * 40)
    print("Saving splits...")
    print("-" * 40)
    
    split_data = save_splits(splits, args.output_dir, metadata)
    
    print("\n" + "=" * 80)
    print("Split Summary")
    print("=" * 80)
    print(f"\nTotal samples: {split_data['statistics']['total']}")
    print(f"  Train: {split_data['statistics']['train']}")
    print(f"  Val:   {split_data['statistics']['val']}")
    print(f"  Test:  {split_data['statistics']['test']}")
    
    print("\nPer-class breakdown:")
    for class_name, stats in split_data['statistics']['per_class'].items():
        print(f"  {class_name:>8}: train={stats['train']:4d}, "
              f"val={stats['val']:4d}, test={stats['test']:4d}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
