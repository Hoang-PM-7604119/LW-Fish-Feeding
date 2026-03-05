#!/usr/bin/env python3
"""
Check potential data leakage caused by duplicate file names across classes
and split overlap.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_ROOT = Path("/mnt/disk1/backup_user/hoang.pm/UFFIA_data")
DEFAULT_RAW_VIDEO_DIR = DEFAULT_ROOT / "video_dataset"
DEFAULT_RAW_AUDIO_DIR = DEFAULT_ROOT / "audio_dataset"
DEFAULT_SPLIT_FILE = DEFAULT_ROOT / "fixed" / "splits" / "splits.json"
DEFAULT_CLASS_NAMES = ["none", "weak", "medium", "strong"]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".pkl")
AUDIO_EXTS = (".wav", ".npy")


def extract_id(path: Path) -> str:
    name = path.name
    name = name.replace("_video_", "_").replace("_audio_", "_")
    for ext in VIDEO_EXTS + AUDIO_EXTS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return files


def collect_ids_by_class(root: Path, exts: Tuple[str, ...], class_names: List[str]) -> Dict[str, List[Path]]:
    ids_by_class: Dict[str, List[Path]] = defaultdict(list)
    for cls in class_names:
        class_root = root / cls
        for path in iter_files(class_root, exts):
            ids_by_class[cls].append(path)
    return ids_by_class


def find_cross_class_duplicates(
    ids_by_class: Dict[str, List[Path]],
) -> Dict[str, Set[str]]:
    id_to_classes: Dict[str, Set[str]] = defaultdict(set)
    for cls, paths in ids_by_class.items():
        for path in paths:
            id_to_classes[extract_id(path)].add(cls)
    return {id_: classes for id_, classes in id_to_classes.items() if len(classes) > 1}


def find_within_class_duplicates(
    ids_by_class: Dict[str, List[Path]],
) -> Dict[str, Dict[str, int]]:
    dupes: Dict[str, Dict[str, int]] = {}
    for cls, paths in ids_by_class.items():
        counts: Dict[str, int] = defaultdict(int)
        for path in paths:
            counts[extract_id(path)] += 1
        cls_dupes = {id_: c for id_, c in counts.items() if c > 1}
        if cls_dupes:
            dupes[cls] = cls_dupes
    return dupes


def check_split_overlap(split_file: Path) -> Tuple[bool, Dict[str, Set[str]]]:
    if not split_file.exists():
        return False, {}
    data = json.loads(split_file.read_text())
    splits = data["splits"] if "splits" in data else data

    id_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            if not isinstance(item, dict):
                continue
            cls = item.get("class", "unknown")
            vid = extract_id(Path(item.get("video_file", "")))
            aid = extract_id(Path(item.get("audio_file", "")))
            key = f"{cls}:{vid}:{aid}"
            id_to_splits[key].add(split_name)

    overlaps = {k: v for k, v in id_to_splits.items() if len(v) > 1}
    return True, overlaps


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data leakage risks.")
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument("--audio_dir", type=Path, default=DEFAULT_RAW_AUDIO_DIR)
    parser.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--max_examples", type=int, default=10)
    args = parser.parse_args()

    print(f"Video dir: {args.video_dir}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Split file: {args.split_file}")

    video_ids = collect_ids_by_class(args.video_dir, VIDEO_EXTS, args.class_names)
    audio_ids = collect_ids_by_class(args.audio_dir, AUDIO_EXTS, args.class_names)

    print_section("Cross-class duplicate IDs (video)")
    video_cross = find_cross_class_duplicates(video_ids)
    if not video_cross:
        print("OK: No cross-class duplicate video IDs.")
    else:
        print(f"Found {len(video_cross)} duplicate video IDs across classes.")
        for i, (id_, classes) in enumerate(video_cross.items()):
            if i >= args.max_examples:
                print("... (truncated)")
                break
            print(f"  {id_} -> {sorted(classes)}")

    print_section("Cross-class duplicate IDs (audio)")
    audio_cross = find_cross_class_duplicates(audio_ids)
    if not audio_cross:
        print("OK: No cross-class duplicate audio IDs.")
    else:
        print(f"Found {len(audio_cross)} duplicate audio IDs across classes.")
        for i, (id_, classes) in enumerate(audio_cross.items()):
            if i >= args.max_examples:
                print("... (truncated)")
                break
            print(f"  {id_} -> {sorted(classes)}")

    print_section("Within-class duplicate IDs (video)")
    video_within = find_within_class_duplicates(video_ids)
    if not video_within:
        print("OK: No within-class duplicate video IDs.")
    else:
        for cls, dupes in video_within.items():
            print(f"{cls}: {len(dupes)} duplicate IDs")
            for i, (id_, count) in enumerate(dupes.items()):
                if i >= args.max_examples:
                    print("  ... (truncated)")
                    break
                print(f"  {id_} -> {count} files")

    print_section("Within-class duplicate IDs (audio)")
    audio_within = find_within_class_duplicates(audio_ids)
    if not audio_within:
        print("OK: No within-class duplicate audio IDs.")
    else:
        for cls, dupes in audio_within.items():
            print(f"{cls}: {len(dupes)} duplicate IDs")
            for i, (id_, count) in enumerate(dupes.items()):
                if i >= args.max_examples:
                    print("  ... (truncated)")
                    break
                print(f"  {id_} -> {count} files")

    has_splits, overlaps = check_split_overlap(args.split_file)
    print_section("Split overlap check")
    if not has_splits:
        print("Split file not found; skipped.")
    elif not overlaps:
        print("OK: No train/val/test overlap detected.")
    else:
        print(f"Found {len(overlaps)} items in multiple splits.")
        for i, (key, splits) in enumerate(overlaps.items()):
            if i >= args.max_examples:
                print("... (truncated)")
                break
            print(f"  {key} -> {sorted(splits)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
