"""
Audio encoder architectures.

Supports:
- ResNet-18/50: Residual networks
- MobileNet V2: Efficient mobile architecture
- EfficientNet: Compound scaling CNN
- PANN CNN10/14: Pretrained Audio Neural Networks
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .pann_local import LogMelFrontend, UFFIALogMelFrontend, Cnn10Backbone, Cnn14Backbone


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 audio encoder.
    
    Uses 2D ResNet-18 on mel-spectrogram input.
    
    Args:
        output_dim: Output embedding dimension (default: 512)
        sample_rate: Audio sample rate (default: 32000)
        n_mels: Number of mel bands (default: 64)
        pretrained: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        output_dim: int = 512,
        sample_rate: int = 32000,
        n_mels: int = 64,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Mel-spectrogram frontend
        from torchaudio.transforms import MelSpectrogram
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels
        )
        
        # ResNet-18 backbone
        resnet = tv_models.resnet18(pretrained=pretrained)
        
        # Modify first conv for single channel mel-spectrogram
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head
        resnet.fc = nn.Identity()
        
        self.backbone = resnet
        self.output_dim = output_dim
        self.proj = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio: Audio waveform [B, n_samples]
            
        Returns:
            features: [B, 1, output_dim]
        """
        # Convert to mel-spectrogram
        mel = self.mel_transform(audio)  # [B, n_mels, time]
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, time]
        
        # Extract features
        feat = self.backbone(mel)  # [B, 512]
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 audio encoder.
    
    Uses 2D ResNet-50 on mel-spectrogram input.
    
    Args:
        output_dim: Output embedding dimension (default: 2048)
        sample_rate: Audio sample rate (default: 32000)
        n_mels: Number of mel bands (default: 64)
        pretrained: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        output_dim: int = 2048,
        sample_rate: int = 32000,
        n_mels: int = 64,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Mel-spectrogram frontend
        from torchaudio.transforms import MelSpectrogram
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels
        )
        
        # ResNet-50 backbone
        resnet = tv_models.resnet50(pretrained=pretrained)
        
        # Modify first conv for single channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head
        resnet.fc = nn.Identity()
        
        self.backbone = resnet
        self.output_dim = output_dim
        self.proj = nn.Linear(2048, output_dim) if output_dim != 2048 else nn.Identity()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mel = self.mel_transform(audio)
        mel = mel.unsqueeze(1)
        feat = self.backbone(mel)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class MobileNetV2Encoder(nn.Module):
    """
    MobileNet V2 audio encoder.
    
    Efficient mobile architecture for audio.
    
    Args:
        output_dim: Output embedding dimension (default: 1280)
        sample_rate: Audio sample rate (default: 32000)
        n_mels: Number of mel bands (default: 64)
        pretrained: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        output_dim: int = 1280,
        sample_rate: int = 32000,
        n_mels: int = 64,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Mel-spectrogram frontend
        from torchaudio.transforms import MelSpectrogram
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels
        )
        
        # MobileNet V2 backbone
        mobilenet = tv_models.mobilenet_v2(pretrained=pretrained)
        
        # Modify first conv for single channel
        mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Remove classification head
        mobilenet.classifier = nn.Identity()
        
        self.backbone = mobilenet
        self.output_dim = output_dim
        self.proj = nn.Linear(1280, output_dim) if output_dim != 1280 else nn.Identity()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mel = self.mel_transform(audio)
        mel = mel.unsqueeze(1)
        feat = self.backbone(mel)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet audio encoder.
    
    Compound scaling CNN for audio.
    
    Args:
        output_dim: Output embedding dimension (default: 1280)
        variant: EfficientNet variant ('b0', 'b1', ..., 'b7') (default: 'b0')
        sample_rate: Audio sample rate (default: 32000)
        n_mels: Number of mel bands (default: 64)
        pretrained: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        output_dim: int = 1280,
        variant: str = 'b0',
        sample_rate: int = 32000,
        n_mels: int = 64,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Mel-spectrogram frontend
        from torchaudio.transforms import MelSpectrogram
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels
        )
        
        # EfficientNet backbone
        if variant == 'b0':
            efficientnet = tv_models.efficientnet_b0(pretrained=pretrained)
            backbone_dim = 1280
        elif variant == 'b1':
            efficientnet = tv_models.efficientnet_b1(pretrained=pretrained)
            backbone_dim = 1280
        elif variant == 'b2':
            efficientnet = tv_models.efficientnet_b2(pretrained=pretrained)
            backbone_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        # Modify first conv for single channel
        efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Remove classification head
        efficientnet.classifier = nn.Identity()
        
        self.backbone = efficientnet
        self.output_dim = output_dim
        self.proj = (
            nn.Linear(backbone_dim, output_dim) if output_dim != backbone_dim
            else nn.Identity()
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mel = self.mel_transform(audio)
        mel = mel.unsqueeze(1)
        feat = self.backbone(mel)
        feat = self.proj(feat)
        return feat.unsqueeze(1)


def _build_pann_frontend(use_uffia: bool, sample_rate: int, window_size: int, hop_size: int,
                         mel_bins: int, fmin: int, fmax: int, **kwargs):
    """Build LogMelFrontend or UFFIALogMelFrontend from kwargs."""
    base = dict(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
                mel_bins=mel_bins, fmin=fmin, fmax=fmax)
    if use_uffia:
        uffia_kw = {
            "time_pad": kwargs.pop("time_pad", 2),
            "spec_augment": kwargs.pop("spec_augment", True),
            "time_drop_width": kwargs.pop("time_drop_width", 64),
            "time_stripes_num": kwargs.pop("time_stripes_num", 2),
            "freq_drop_width": kwargs.pop("freq_drop_width", 8),
            "freq_stripes_num": kwargs.pop("freq_stripes_num", 2),
        }
        return UFFIALogMelFrontend(**base, **uffia_kw)
    return LogMelFrontend(**base)


class PANNCNN10Encoder(nn.Module):
    """PANN CNN10 encoder. Waveform -> LogMel (or U-FFIA frontend) -> Cnn10 -> [B, 1, 512] or logits."""

    def __init__(
        self,
        output_dim: int = 512,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
        pretrained_path: Optional[str] = None,
        use_uffia_frontend: bool = False,
        use_uffia_style: bool = False,
        **kwargs
    ):
        super().__init__()
        self.use_uffia_style = use_uffia_style
        self.classes_num = classes_num if use_uffia_style else 527
        self.frontend = _build_pann_frontend(
            use_uffia=use_uffia_frontend,
            sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
            mel_bins=mel_bins, fmin=fmin, fmax=fmax,
            **kwargs
        )
        self.backbone = Cnn10Backbone(classes_num=self.classes_num)
        self.output_dim = output_dim
        assert output_dim == 512
        if pretrained_path:
            if os.path.exists(pretrained_path):
                self._load_pretrained(pretrained_path)
            else:
                print(f"⚠ Pretrained path not found (skipping load): {pretrained_path}")
                print(f"  Resolve with: place checkpoint there or set pretrained_path to an absolute path.")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(audio)
        if self.use_uffia_style:
            _, logits = self.backbone(x, return_logits=True)
            return logits  # [B, num_classes]
        embedding = self.backbone(x)
        return embedding.unsqueeze(1)

    def _load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        backbone_dict = self.backbone.state_dict()
        pretrained = {k: v for k, v in state_dict.items() if k in backbone_dict and v.shape == backbone_dict[k].shape}
        not_loaded = [k for k in backbone_dict if k not in pretrained]
        if pretrained:
            self.backbone.load_state_dict(pretrained, strict=False)
            print(f"✓ Loaded PANN CNN10 pretrained: {len(pretrained)} keys from {checkpoint_path}")
            if not_loaded:
                print(f"  Skipped (shape mismatch or not in checkpoint): {not_loaded}")
        else:
            print(f"⚠ No matching keys in checkpoint (wrong format?): {checkpoint_path}")


class PANNCNN14Encoder(nn.Module):
    """PANN CNN14 encoder. Waveform -> LogMel (or U-FFIA frontend) -> Cnn14 -> [B, 1, 2048] or logits."""

    def __init__(
        self,
        output_dim: int = 2048,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
        pretrained_path: Optional[str] = None,
        use_uffia_frontend: bool = False,
        use_uffia_style: bool = False,
        **kwargs
    ):
        super().__init__()
        self.use_uffia_style = use_uffia_style
        self.classes_num = classes_num if use_uffia_style else 527
        self.frontend = _build_pann_frontend(
            use_uffia=use_uffia_frontend,
            sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
            mel_bins=mel_bins, fmin=fmin, fmax=fmax,
            **kwargs
        )
        self.backbone = Cnn14Backbone(classes_num=self.classes_num)
        self.output_dim = output_dim
        assert output_dim == 2048
        if pretrained_path:
            if os.path.exists(pretrained_path):
                self._load_pretrained(pretrained_path)
            else:
                print(f"⚠ Pretrained path not found (skipping load): {pretrained_path}")
                print(f"  Resolve with: place checkpoint there or set pretrained_path to an absolute path.")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(audio)
        if self.use_uffia_style:
            _, logits = self.backbone(x, return_logits=True)
            return logits  # [B, num_classes]
        embedding = self.backbone(x)
        return embedding.unsqueeze(1)

    def _load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        backbone_dict = self.backbone.state_dict()
        pretrained = {k: v for k, v in state_dict.items() if k in backbone_dict and v.shape == backbone_dict[k].shape}
        not_loaded = [k for k in backbone_dict if k not in pretrained]
        if pretrained:
            self.backbone.load_state_dict(pretrained, strict=False)
            print(f"✓ Loaded PANN CNN14 pretrained: {len(pretrained)} keys from {checkpoint_path}")
            if not_loaded:
                print(f"  Skipped (shape mismatch or not in checkpoint): {not_loaded}")
        else:
            print(f"⚠ No matching keys in checkpoint (wrong format?): {checkpoint_path}")


def get_audio_encoder(
    encoder_type: str,
    output_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to get audio encoder by name.
    
    Args:
        encoder_type: Type of encoder ('resnet18', 'resnet50', 'mobilenet', 
                     'efficientnet', 'pann_cnn10', 'pann_cnn14')
        output_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Audio encoder module
        
    Example:
        >>> encoder = get_audio_encoder('pann_cnn10', output_dim=512, pretrained_path='cnn10.pth')
        >>> encoder = get_audio_encoder('resnet18', output_dim=512, pretrained=True)
    """
    encoder_type = encoder_type.lower()

    # Only PANN encoders should receive `pretrained_path`. For torchvision
    # encoders (resnet / mobilenet / efficientnet), drop it to avoid passing
    # unexpected kwargs into their constructors.
    pann_types = ['pann_cnn10', 'cnn10', 'pann_cnn14', 'cnn14']
    if encoder_type not in pann_types:
        kwargs.pop("pretrained_path", None)
    
    if encoder_type == 'resnet18':
        return ResNet18Encoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'resnet50':
        return ResNet50Encoder(output_dim=output_dim, **kwargs)
    elif encoder_type in ['mobilenet', 'mobilenetv2']:
        return MobileNetV2Encoder(output_dim=output_dim, **kwargs)
    elif encoder_type == 'efficientnet':
        return EfficientNetEncoder(output_dim=output_dim, **kwargs)
    elif encoder_type in ['pann_cnn10', 'cnn10']:
        return PANNCNN10Encoder(output_dim=output_dim, **kwargs)
    elif encoder_type in ['pann_cnn14', 'cnn14']:
        return PANNCNN14Encoder(output_dim=output_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown audio encoder type: {encoder_type}. "
            f"Available: resnet18, resnet50, mobilenet, efficientnet, pann_cnn10, pann_cnn14"
        )


if __name__ == '__main__':
    """Test audio encoders."""
    print("="*80)
    print("Testing Audio Encoders")
    print("="*80)
    
    batch_size = 2
    audio_samples = 64000  # 2 seconds at 32kHz
    
    audio_input = torch.randn(batch_size, audio_samples)
    print(f"\nInput shape: {audio_input.shape}")
    
    # Test ResNet18
    print("\n" + "-"*80)
    print("Testing ResNet18 Encoder")
    print("-"*80)
    try:
        encoder = ResNet18Encoder(output_dim=512)
        output = encoder(audio_input)
        print(f"✓ ResNet18 output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    except Exception as e:
        print(f"✗ ResNet18 failed: {e}")
    
    # Test MobileNetV2
    print("\n" + "-"*80)
    print("Testing MobileNetV2 Encoder")
    print("-"*80)
    try:
        encoder = MobileNetV2Encoder(output_dim=512)
        output = encoder(audio_input)
        print(f"✓ MobileNetV2 output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    except Exception as e:
        print(f"✗ MobileNetV2 failed: {e}")
    
    print("\n" + "="*80)
    print("✓ Audio encoder testing completed!")
    print("="*80)
