"""
Data processing module.

Includes:
- Preprocessing: Video and audio preprocessing
- Splits: Dataset splitting utilities
- Datasets: PyTorch dataset classes
"""

from .preprocessing import video_preprocessing, audio_preprocessing
from .splits import split_utils
from .datasets import multimodal_dataset
