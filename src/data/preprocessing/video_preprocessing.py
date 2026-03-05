"""
Video preprocessing utilities.

Supports multiple frame sampling methods:
- Uniform: Evenly spaced frames across the video
- Random: Random frame selection
- Consecutive: Sequential frames from a position
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

DEFAULT_RAW_DATA_ROOT = "/mnt/disk1/backup_user/hoang.pm/UFFIA_data"
DEFAULT_FIXED_DATA_ROOT = f"{DEFAULT_RAW_DATA_ROOT}/fixed"
DEFAULT_VIDEO_INPUT_DIR = f"{DEFAULT_RAW_DATA_ROOT}/video_dataset"
DEFAULT_VIDEO_OUTPUT_DIR = f"{DEFAULT_FIXED_DATA_ROOT}/processed_video"


def uniform_sampling(
    video_path: str,
    num_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Sample frames uniformly (evenly spaced) from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        img_size: Target image size (H, W)
        
    Returns:
        frames: Array of shape [num_frames, H, W, C] or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        cap.release()
        return None
    
    # Calculate frame indices
    if total_frames < num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                cap.release()
                return None
    
    cap.release()
    return np.array(frames)


def random_sampling(
    video_path: str,
    num_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    seed: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Sample frames randomly from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        img_size: Target image size (H, W)
        seed: Random seed for reproducibility
        
    Returns:
        frames: Array of shape [num_frames, H, W, C] or None if failed
    """
    if seed is not None:
        np.random.seed(seed)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        cap.release()
        return None
    
    # Randomly sample frame indices
    if total_frames < num_frames:
        frame_indices = np.random.choice(total_frames, num_frames, replace=True)
    else:
        frame_indices = np.random.choice(total_frames, num_frames, replace=False)
    
    frame_indices = np.sort(frame_indices)  # Sort for efficient reading
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                cap.release()
                return None
    
    cap.release()
    return np.array(frames)


def consecutive_sampling(
    video_path: str,
    num_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    start_position: str = 'center'
) -> Optional[np.ndarray]:
    """
    Sample consecutive frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        img_size: Target image size (H, W)
        start_position: Where to start ('start', 'center', 'end')
        
    Returns:
        frames: Array of shape [num_frames, H, W, C] or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        cap.release()
        return None
    
    # Calculate start frame
    if start_position == 'start':
        start_frame = 0
    elif start_position == 'center':
        start_frame = max(0, (total_frames - num_frames) // 2)
    elif start_position == 'end':
        start_frame = max(0, total_frames - num_frames)
    else:
        start_frame = 0
    
    # Sample consecutive frames
    frame_indices = np.arange(start_frame, min(start_frame + num_frames, total_frames))
    
    # Pad if needed
    if len(frame_indices) < num_frames:
        padding = num_frames - len(frame_indices)
        frame_indices = np.pad(frame_indices, (0, padding), mode='edge')
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                cap.release()
                return None
    
    cap.release()
    return np.array(frames)


def preprocess_video_dataset(
    input_dir: str,
    output_dir: str,
    sampling_method: str = 'uniform',
    num_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
):
    """
    Preprocess entire video dataset.
    
    Args:
        input_dir: Input directory containing videos
        output_dir: Output directory for preprocessed data
        sampling_method: Sampling method ('uniform', 'random', 'consecutive')
        num_frames: Number of frames to sample
        img_size: Target image size (H, W)
        extensions: Video file extensions to process
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select sampling function
    if sampling_method == 'uniform':
        sample_fn = uniform_sampling
    elif sampling_method == 'random':
        sample_fn = random_sampling
    elif sampling_method == 'consecutive':
        sample_fn = consecutive_sampling
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Find all video files
    video_files = []
    for ext in extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"\nPreprocessing {len(video_files)} videos with {sampling_method} sampling...")
    print(f"Output directory: {output_dir}\n")
    
    success_count = 0
    failed_files = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        # Get relative path to maintain directory structure
        rel_path = video_file.relative_to(input_path)
        output_file = output_path / rel_path.with_suffix('.pkl')
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if output_file.exists():
            success_count += 1
            continue
        
        # Process video
        frames = sample_fn(str(video_file), num_frames=num_frames, img_size=img_size)
        
        if frames is not None:
            # Save as pickle (dict for backward-compatible loading)
            payload = {
                "video_form": frames,
                "metadata": {
                    "sampling_method": sampling_method,
                    "num_frames": num_frames,
                    "img_size": img_size
                }
            }
            with open(output_file, 'wb') as f:
                pickle.dump(payload, f)
            success_count += 1
        else:
            failed_files.append(str(video_file))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Preprocessing Summary")
    print(f"{'='*80}")
    print(f"  Total videos:    {len(video_files)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed:          {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    """Test video preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument(
        '--input_dir',
        type=str,
        default=DEFAULT_VIDEO_INPUT_DIR,
        help='Input video directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_VIDEO_OUTPUT_DIR,
        help='Output directory'
    )
    parser.add_argument('--sampling_method', type=str, default='uniform',
                       choices=['uniform', 'random', 'consecutive'],
                       help='Frame sampling method')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (H W)')
    
    args = parser.parse_args()
    
    preprocess_video_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sampling_method=args.sampling_method,
        num_frames=args.num_frames,
        img_size=tuple(args.img_size)
    )
