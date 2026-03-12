"""
Model architectures module.

Includes:
- Video encoders (S3D, X3D, MoViNet, I3D, Video MAE)
- Audio encoders (ResNet, MobileNet, EfficientNet, PANN)
- Fusion methods (Concat, Cross-Attention, Gated, Joint, MBT)
"""

from .encoders import video_encoders, audio_encoders
from .fusion import fusion_methods
