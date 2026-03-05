"""
Encoder architectures for video and audio modalities.
"""

from .video_encoders import (
    S3DEncoder,
    X3DEncoder,
    MoViNetEncoder,
    I3DEncoder,
    VideoMAEEncoder,
    get_video_encoder
)

from .audio_encoders import (
    ResNet18Encoder,
    ResNet50Encoder,
    MobileNetV2Encoder,
    EfficientNetEncoder,
    PANNCNN10Encoder,
    PANNCNN14Encoder,
    get_audio_encoder
)

__all__ = [
    # Video encoders
    'S3DEncoder',
    'X3DEncoder',
    'MoViNetEncoder',
    'I3DEncoder',
    'VideoMAEEncoder',
    'get_video_encoder',
    # Audio encoders
    'ResNet18Encoder',
    'ResNet50Encoder',
    'MobileNetV2Encoder',
    'EfficientNetEncoder',
    'PANNCNN10Encoder',
    'PANNCNN14Encoder',
    'get_audio_encoder',
]
