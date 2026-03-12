"""
Video encoder architectures.

Supports:
- S3D: Spatiotemporal 3D CNN
- X3D: Efficient 3D CNN
- MoViNet: Mobile Video Networks
- I3D: Inflated 3D ConvNet
- Video MAE: Masked Autoencoder for Video
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s3d_original import S3D as S3DBackbone


class S3DEncoder(nn.Module):
    """
    S3D (Spatiotemporal 3D) video encoder.
    
    Uses S3D_16frames backbone for 16-frame video clips.
    Pretrained on Kinetics-400.
    
    Args:
        output_dim: Output embedding dimension (default: 1024)
        input_channels: Number of input channels (default: 3 for RGB)
        classes_num: Number of classes for pretrained model (default: 4)
        pretrained_path: Path to pretrained weights
    """
    
    def __init__(
        self,
        output_dim: int = 1024,
        input_channels: int = 3,  # kept for API compatibility
        classes_num: int = 400,   # S3D pretrained on Kinetics-400
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        # Use vendored S3D implementation (from kylemin/S3D) so the
        # project runs without extra git clones or pip installs.
        self.backbone = S3DBackbone(num_class=classes_num)
        
        self.output_dim = output_dim
        self.proj = (
            nn.Linear(1024, output_dim) if output_dim != 1024 else nn.Identity()
        )
        
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Video tensor [B, T, C, H, W] or [B, C, T, H, W]
            
        Returns:
            features: [B, 1, output_dim]
        """
        # S3D expects [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)

        # Use backbone up to the final conv features (before classification head)
        with torch.set_grad_enabled(self.training):
            feats_3d = self.backbone.base(x)  # [B, 1024, T', H', W']
            # Global average pooling over time and space
            feats_3d = F.avg_pool3d(feats_3d, (feats_3d.size(2), feats_3d.size(3), feats_3d.size(4)))

        feat = feats_3d.view(feats_3d.size(0), -1)  # [B, 1024]
        feat = self.proj(feat)
        return feat.unsqueeze(1)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        else:
            state_dict = ckpt
        
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"✓ Loaded S3D pretrained weights from {checkpoint_path}")


def _load_pyth_state(path: str) -> Optional[dict]:
    """Load state dict from PyTorch Video .pyth checkpoint (uses 'model_state' key)."""
    if not path or not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("state_dict") or (ckpt if isinstance(ckpt, dict) else None)
    return state if isinstance(state, dict) else None


def _x3d_backbone_from_pytorchvideo(variant: str, pretrained_path: Optional[str] = None) -> nn.Module:
    """
    Build X3D backbone from PyTorch Video (torch.hub). .pyth checkpoints use this
    architecture, so loading is exact. Returns module that outputs [B, 2048].
    """
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        f"x3d_{variant}",
        pretrained=False,
        trust_repo=True,
    )
    state = _load_pyth_state(pretrained_path) if pretrained_path else None
    if state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  X3D load: {len(missing)} keys missing (often head); {len(unexpected)} unexpected.")
        print(f"✓ Loaded X3D-{variant.upper()} pretrained from {pretrained_path}")
    # PyTorch Video X3D: model.blocks = [stem, stage1, stage2, stage3, stage4, head]
    # Replace head's classifier so we get 2048-d features instead of num_classes
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        head = model.blocks[-1]
        if hasattr(head, "proj"):
            head.proj = nn.Identity()
    return model


def _i3d_backbone_from_pytorchvideo(pretrained_path: Optional[str] = None) -> nn.Module:
    """
    Build I3D R50 backbone from PyTorch Video (torch.hub). .pyth checkpoints
    use this architecture. Returns module that outputs [B, 2048].
    """
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "i3d_r50",
        pretrained=False,
        trust_repo=True,
    )
    state = _load_pyth_state(pretrained_path) if pretrained_path else None
    if state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  I3D load: {len(missing)} keys missing; {len(unexpected)} unexpected.")
        print(f"✓ Loaded I3D R50 pretrained from {pretrained_path}")
    # Replace head so we get 2048-d features
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        head = model.blocks[-1]
        if hasattr(head, "proj"):
            head.proj = nn.Identity()
    return model


class X3DEncoder(nn.Module):
    """
    X3D (Efficient 3D CNN) video encoder.
    
    X3D uses expansion-compression design for efficiency.
    Available variants: XS, S, M, L.
    When pretrained_path points to a .pyth file, weights are loaded via PyTorch Video
    (torch.hub) so keys match; otherwise torchvision is used if available.
    
    Args:
        output_dim: Output embedding dimension (default: 2048)
        variant: Model variant ('xs', 's', 'm', 'l') (default: 's')
        pretrained: Use torchvision pretrained weights (default: False)
        pretrained_path: Path to .pyth checkpoint (PyTorch Video format)
    """
    
    def __init__(
        self,
        output_dim: int = 2048,
        variant: str = 's',
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        backbone_dim = 2048
        use_pytorchvideo = pretrained_path and os.path.exists(pretrained_path)
        if use_pytorchvideo:
            try:
                self.backbone = _x3d_backbone_from_pytorchvideo(variant, pretrained_path)
            except Exception as e:
                raise RuntimeError(
                    f"X3D from .pyth requires pytorchvideo (pip install pytorchvideo). {e}"
                ) from e
        else:
            try:
                import torchvision.models.video as video_models
                use_pretrained = pretrained
                if variant == 'xs':
                    self.backbone = video_models.x3d_xs(pretrained=use_pretrained)
                elif variant == 's':
                    self.backbone = video_models.x3d_s(pretrained=use_pretrained)
                elif variant == 'm':
                    self.backbone = video_models.x3d_m(pretrained=use_pretrained)
                elif variant == 'l':
                    self.backbone = video_models.x3d_l(pretrained=use_pretrained)
                else:
                    raise ValueError(f"Unknown X3D variant: {variant}")
                self.backbone.fc = nn.Identity()
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    "X3D requires either pretrained_path (.pyth + pytorchvideo) or "
                    "torchvision.models.video. Install pytorchvideo or torchvision>=0.12."
                ) from e
        self.output_dim = output_dim
        self.proj = (
            nn.Linear(backbone_dim, output_dim) if output_dim != backbone_dim
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Video tensor [B, C, T, H, W]
            
        Returns:
            features: [B, 1, output_dim]
        """
        if x.dim() == 5 and x.shape[1] != 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class MoViNetEncoder(nn.Module):
    """
    MoViNet (Mobile Video Networks) encoder.
    
    Efficient video model designed for mobile devices.
    Available variants: A0, A1, A2, A3, A4, A5.
    Weights can be loaded from torchvision (pretrained=True) or from disk (pretrained_path).
    
    Args:
        output_dim: Output embedding dimension (default: 2048)
        variant: Model variant ('a0', 'a1', 'a2', 'a3', 'a4', 'a5') (default: 'a0')
        pretrained: Use torchvision pretrained weights (default: False)
        pretrained_path: Path to .pt checkpoint (overrides pretrained when set)
    """
    
    def __init__(
        self,
        output_dim: int = 2048,
        variant: str = 'a0',
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        try:
            from torchvision.models.video import (
                movinet_a0, movinet_a1, movinet_a2,
                movinet_a3, movinet_a4, movinet_a5
            )
            variant_map = {
                'a0': (movinet_a0, 2048),
                'a1': (movinet_a1, 2048),
                'a2': (movinet_a2, 2048),
                'a3': (movinet_a3, 2048),
                'a4': (movinet_a4, 2048),
                'a5': (movinet_a5, 2048),
            }
            if variant not in variant_map:
                raise ValueError(f"Unknown MoViNet variant: {variant}")
            use_pretrained = pretrained and not (pretrained_path and os.path.exists(pretrained_path))
            model_fn, backbone_dim = variant_map[variant]
            self.backbone = model_fn(pretrained=use_pretrained)
            self.backbone.classifier = nn.Identity()
            self.output_dim = output_dim
            self.proj = (
                nn.Linear(backbone_dim, output_dim) if output_dim != backbone_dim
                else nn.Identity()
            )
            if pretrained_path and os.path.exists(pretrained_path):
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(state, dict):
                    model_dict = self.backbone.state_dict()
                    loaded = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
                    if loaded:
                        model_dict.update(loaded)
                        self.backbone.load_state_dict(model_dict, strict=False)
                        print(f"✓ Loaded MoViNet pretrained weights from {pretrained_path}")
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "MoViNet requires torchvision.models.video (e.g. movinet_a0). "
                "Install torchvision>=0.12 or use S3D/I3D."
            ) from e
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Video tensor [B, C, T, H, W] or [B, T, C, H, W]
            
        Returns:
            features: [B, 1, output_dim]
        """
        # MoViNet expects [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        
        feat = self.backbone(x)  # [B, backbone_dim]
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class I3DEncoder(nn.Module):
    """
    I3D (Inflated 3D ConvNet) encoder.
    
    2D CNN inflated to 3D for video understanding.
    When pretrained_path points to a .pyth file, weights are loaded via PyTorch Video
    (torch.hub) so keys match; otherwise torchvision or a lightweight placeholder.
    
    Args:
        output_dim: Output embedding dimension (default: 1024)
        pretrained: Use torchvision pretrained weights (default: False)
        pretrained_path: Path to I3D checkpoint, e.g. I3D_8x8_R50_kinetics400.pyth
    """
    
    def __init__(
        self,
        output_dim: int = 1024,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        self.output_dim = output_dim
        backbone_dim = 2048  # I3D R50 feature dim before classifier
        use_pytorchvideo = pretrained_path and os.path.exists(pretrained_path)
        if use_pytorchvideo:
            try:
                self.backbone = _i3d_backbone_from_pytorchvideo(pretrained_path)
            except Exception as e:
                raise RuntimeError(
                    f"I3D from .pyth requires pytorchvideo (pip install pytorchvideo). {e}"
                ) from e
        else:
            try:
                import torchvision.models.video as video_models
                i3d_fn = getattr(video_models, "i3d_r50_video", None)
                if i3d_fn is not None:
                    self.backbone = i3d_fn(pretrained=pretrained)
                    self.backbone.fc = nn.Identity()
                    backbone_dim = 2048
                else:
                    raise AttributeError("i3d_r50_video not in torchvision")
            except (ImportError, AttributeError):
                print("Warning: I3D using lightweight placeholder. Use pretrained_path (.pyth) + pytorchvideo for full I3D.")
                self.backbone = nn.Sequential(
                    nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.AdaptiveAvgPool3d((1, 1, 1))
                )
                backbone_dim = 64
        self.proj = (
            nn.Linear(backbone_dim, output_dim) if output_dim != backbone_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 5 and x.shape[1] != 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class VideoMAEEncoder(nn.Module):
    """
    Video MAE (Masked Autoencoder) encoder.
    
    Self-supervised pretrained video encoder.
    
    Args:
        output_dim: Output embedding dimension (default: 768)
        pretrained: Use pretrained weights (default: False)
    """
    
    def __init__(
        self,
        output_dim: int = 768,
        pretrained: bool = False
    ):
        super().__init__()
        
        # Placeholder - Video MAE implementation
        print("Warning: Video MAE encoder is a placeholder. Implement with actual Video MAE model.")
        
        self.output_dim = output_dim
        # Dummy backbone
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.proj = nn.Linear(64, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 5 and x.shape[1] != 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


def get_video_encoder(
    encoder_type: str,
    output_dim: int = 1024,
    **kwargs
) -> nn.Module:
    """
    Factory function to get video encoder by name.
    
    Args:
        encoder_type: Type of encoder ('s3d', 'x3d', 'movinet', 'i3d', 'videomae')
        output_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Video encoder module
        
    Example:
        >>> encoder = get_video_encoder('s3d', output_dim=1024, pretrained_path='s3d.pth')
        >>> encoder = get_video_encoder('movinet', output_dim=512, variant='a0', pretrained=True)
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 's3d':
        return S3DEncoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'x3d':
        return X3DEncoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'movinet':
        return MoViNetEncoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'i3d':
        return I3DEncoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'videomae':
        return VideoMAEEncoder(output_dim=output_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown video encoder type: {encoder_type}. "
            f"Available: s3d, x3d, movinet, i3d, videomae"
        )


if __name__ == '__main__':
    """Test video encoders."""
    print("="*80)
    print("Testing Video Encoders")
    print("="*80)
    
    batch_size = 2
    num_frames = 16
    channels = 3
    height = 224
    width = 224
    
    # Test input [B, T, C, H, W]
    video_input = torch.randn(batch_size, num_frames, channels, height, width)
    print(f"\nInput shape: {video_input.shape}")
    
    # Test S3D
    print("\n" + "-"*80)
    print("Testing S3D Encoder")
    print("-"*80)
    try:
        encoder = S3DEncoder(output_dim=1024)
        output = encoder(video_input)
        print(f"✓ S3D output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    except Exception as e:
        print(f"✗ S3D failed: {e}")
    
    print("\n" + "="*80)
    print("✓ Video encoder testing completed!")
    print("="*80)
