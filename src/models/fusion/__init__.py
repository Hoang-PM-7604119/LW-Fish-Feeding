"""
Fusion methods for multimodal learning.
"""

from .fusion_methods import (
    ConcatFusion,
    CrossAttentionFusion,
    GatedFusion,
    JointCrossAttentionFusion,
    MBTFusion,
    get_fusion_method
)

__all__ = [
    'ConcatFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'JointCrossAttentionFusion',
    'MBTFusion',
    'get_fusion_method',
]
