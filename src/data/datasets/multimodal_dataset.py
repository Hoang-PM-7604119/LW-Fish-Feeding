"""
Multimodal Dataset Loader
Combines video data (mp4 or pkl) and audio data (wav)

Data structure:
    video_dir/date/feeding_level/class_name/*.mp4 (or *.pkl)
    audio_dir/date/feeding_level/class_name/*.wav

Classes: none, weak, medium, strong
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import librosa
from scipy.signal import resample
import warnings
warnings.filterwarnings("ignore")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_pickle(fname):
    """Load object from pickle file"""
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res


def load_video_frames(path, num_frames=16, target_size=(224, 224)):
    """
    Load video frames from mp4 file
    
    Args:
        path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target frame size (H, W)
    
    Returns:
        frames: Tensor of shape [T, C, H, W]
    """
    if not HAS_CV2:
        raise ImportError("cv2 required for loading mp4 videos. Install with: pip install opencv-python")
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, target_size[0], target_size[1])
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
            frames.append(frame)
        else:
            frames.append(torch.zeros(3, target_size[0], target_size[1]))
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(torch.zeros(3, target_size[0], target_size[1]))
    
    return torch.stack(frames[:num_frames])  # [T, C, H, W]


def load_audio(path, target_sr=32000, duration=2.0):
    """
    Load audio file and resample to target sample rate
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 32000)
        duration: Target duration in seconds (default: 2.0)
    
    Returns:
        waveform: numpy array of shape (n_samples,)
    """
    try:
        target_length = int(target_sr * duration)

        if path.endswith(".npy"):
            y = np.load(path).astype(np.float32)
        else:
            y, sr = librosa.load(path, sr=None)
            if sr != target_sr:
                n_samples = int(len(y) * target_sr / sr)
                y = resample(y, n_samples)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant', constant_values=0)
        elif len(y) > target_length:
            y = y[:target_length]

        return y.astype(np.float32)

    except Exception as e:
        print(f"Error loading audio {path}: {str(e)}")
        return np.zeros(int(target_sr * duration), dtype=np.float32)


def get_file_identifier(filepath, class_names=None):
    """Extract unique identifier from filename including class/date/feed."""
    filename = os.path.basename(filepath)
    parent_dirs = os.path.dirname(filepath).split(os.sep)
    
    name = filename.replace('_video_', '_').replace('_audio_', '_')
    name = name.replace('.mp4', '').replace('.wav', '').replace('.pkl', '').replace('.npy', '')
    
    class_name = None
    if class_names:
        for p in parent_dirs:
            if p in class_names:
                class_name = p
                break

    date_part = None
    feed_part = None
    for p in parent_dirs:
        if p.startswith('2022_'):
            date_part = p
        if p.startswith('AM_') or p.startswith('PM_'):
            feed_part = p
    
    prefix_parts = []
    if class_name:
        prefix_parts.append(class_name)
    if date_part and feed_part:
        prefix_parts.append(f"{date_part}_{feed_part}")

    if prefix_parts:
        return f"{'_'.join(prefix_parts)}_{name}"
    return name


def find_all_files(base_dir, class_name, extension):
    """Find all files of given extension for a class."""
    if isinstance(extension, (list, tuple)):
        extensions = extension
    else:
        extensions = [extension]

    patterns = []
    for ext in extensions:
        patterns.extend([
            os.path.join(base_dir, '**', class_name, f'*.{ext}'),
            os.path.join(base_dir, class_name, f'*.{ext}'),
        ])
    
    all_paths = []
    for pattern in patterns:
        all_paths.extend(glob.glob(pattern, recursive=True))
    
    return list(set(all_paths))


def load_fixed_splits(split_file, video_dir, audio_dir):
    """Load fixed splits from JSON file."""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    label_map = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for split_name in ['train', 'val', 'test']:
        items = split_data['splits'][split_name]
        for item in items:
            video_path = os.path.join(video_dir, item['video_file'])
            audio_path = os.path.join(audio_dir, item['audio_file'])
            label = label_map[item['class']]
            splits[split_name].append([(video_path, audio_path), label])
    
    return splits


def data_generator(processed_video_path, audio_dataset_path, seed, test_sample_per_class=700,
                   split_file=None):
    """
    Generate train/test/val splits for multimodal data
    
    Class to label mapping:
        none: 0, weak: 1, medium: 2, strong: 3
    """
    if split_file is not None and os.path.exists(split_file):
        print(f"  Loading fixed splits from: {split_file}")
        splits = load_fixed_splits(split_file, processed_video_path, audio_dataset_path)
        
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val:   {len(splits['val'])} samples")
        print(f"  Test:  {len(splits['test'])} samples")
        
        return splits['train'], splits['test'], splits['val']
    
    print(f"  Using dynamic splitting (seed={seed}, test_per_class={test_sample_per_class})")
    
    random_state = np.random.RandomState(seed)
    
    classes = ['none', 'weak', 'medium', 'strong']
    class_pairs = {}
    
    for class_name in classes:
        video_paths = find_all_files(processed_video_path, class_name, 'mp4')
        if not video_paths:
            video_paths = find_all_files(processed_video_path, class_name, 'pkl')
        audio_paths = find_all_files(audio_dataset_path, class_name, ['npy', 'wav'])
        
        audio_dict = {}
        for audio_path in audio_paths:
            identifier = get_file_identifier(audio_path, classes)
            if identifier in audio_dict:
                if audio_dict[identifier].endswith(".npy"):
                    continue
                if audio_path.endswith(".npy"):
                    audio_dict[identifier] = audio_path
            else:
                audio_dict[identifier] = audio_path
        
        pairs = []
        for video_path in video_paths:
            identifier = get_file_identifier(video_path, classes)
            if identifier in audio_dict:
                pairs.append((video_path, audio_dict[identifier]))
        
        class_pairs[class_name] = pairs
        print(f"  {class_name:>8}: {len(pairs)} pairs")
    
    for class_name in classes:
        random_state.shuffle(class_pairs[class_name])
    
    train_dict = []
    test_dict = []
    val_dict = []
    
    label_map = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}
    
    for class_name in classes:
        pairs = class_pairs[class_name]
        label = label_map[class_name]
        
        test_pairs = pairs[:test_sample_per_class]
        val_pairs = pairs[test_sample_per_class:2*test_sample_per_class]
        train_pairs = pairs[2*test_sample_per_class:]
        
        for pair in train_pairs:
            train_dict.append([pair, label])
        for pair in test_pairs:
            test_dict.append([pair, label])
        for pair in val_pairs:
            val_dict.append([pair, label])
    
    random_state.shuffle(train_dict)
    random_state.shuffle(test_dict)
    random_state.shuffle(val_dict)
    
    return train_dict, test_dict, val_dict


class MultimodalDataset(Dataset):
    """Multimodal Dataset for audio-visual classification"""
    
    def __init__(self, processed_video_path, audio_dataset_path, seed, split='train',
                 test_sample_per_class=700, audio_sr=32000, audio_duration=2.0,
                 num_frames=16, frame_size=(224, 224), split_file=None, use_audio_only=False):
        """
        Args:
            processed_video_path: Path to video directory
            audio_dataset_path: Path to audio directory
            seed: Random seed
            split: 'train', 'test', or 'val'
            test_sample_per_class: Samples per class for test/val
            audio_sr: Audio sample rate
            audio_duration: Audio duration in seconds
            num_frames: Number of video frames to extract
            frame_size: Video frame size (H, W)
            split_file: Path to fixed splits JSON file
            use_audio_only: If True, do not load video from disk (return zeros); use when training audio-only.
        """
        self.processed_video_path = processed_video_path
        self.audio_dataset_path = audio_dataset_path
        self.audio_sr = audio_sr
        self.audio_duration = audio_duration
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.use_audio_only = use_audio_only
        
        print(f"\nCreating {split.upper()} dataset")
        print(f"Video path: {processed_video_path}")
        print(f"Audio path: {audio_dataset_path}")
        
        train_dict, test_dict, val_dict = data_generator(
            processed_video_path, audio_dataset_path, seed, test_sample_per_class,
            split_file=split_file
        )
        
        if split == 'train':
            self.data_dict = train_dict
        elif split == 'test':
            self.data_dict = test_dict
        elif split == 'val':
            self.data_dict = val_dict
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"{split.upper()} dataset: {len(self.data_dict)} samples\n")
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        (video_path, audio_path), label = self.data_dict[index]
        
        if self.use_audio_only:
            # Do not load video from disk; return dummy tensor so same splits work for audio-only
            video_frames = torch.zeros(
                self.num_frames, 3, self.frame_size[0], self.frame_size[1],
                dtype=torch.float32
            )
        elif video_path.endswith('.pkl'):
            video_data = load_pickle(video_path)
            if isinstance(video_data, dict) and "video_form" in video_data:
                video_frames = video_data["video_form"]
            else:
                video_frames = video_data

            if isinstance(video_frames, np.ndarray):
                if video_frames.ndim == 4 and video_frames.shape[-1] == 3:
                    video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2)
                else:
                    video_frames = torch.from_numpy(video_frames)
        else:
            video_frames = load_video_frames(video_path, self.num_frames, self.frame_size)
        
        audio_waveform = load_audio(audio_path, self.audio_sr, self.audio_duration)
        audio_tensor = torch.from_numpy(audio_waveform)
        
        return {
            'video': video_frames,
            'audio': audio_tensor,
            'label': label,
            'video_path': video_path,
            'audio_path': audio_path
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    videos = torch.stack([data['video'] for data in batch])
    audios = torch.stack([data['audio'] for data in batch])
    labels = torch.LongTensor([data['label'] for data in batch])
    
    return {
        'video': videos,
        'audio': audios,
        'label': labels,
        'video_paths': [data['video_path'] for data in batch],
        'audio_paths': [data['audio_path'] for data in batch]
    }


def get_multimodal_dataloader(processed_video_path, audio_dataset_path, split,
                              batch_size, seed, test_sample_per_class=700,
                              audio_sr=32000, audio_duration=2.0,
                              shuffle=True, drop_last=False, num_workers=4,
                              split_file=None, num_frames=16, frame_size=(224, 224),
                              use_audio_only=False):
    """Create DataLoader for multimodal dataset"""
    dataset = MultimodalDataset(
        processed_video_path=processed_video_path,
        audio_dataset_path=audio_dataset_path,
        split=split,
        seed=seed,
        test_sample_per_class=test_sample_per_class,
        audio_sr=audio_sr,
        audio_duration=audio_duration,
        split_file=split_file,
        num_frames=num_frames,
        frame_size=frame_size,
        use_audio_only=use_audio_only
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
