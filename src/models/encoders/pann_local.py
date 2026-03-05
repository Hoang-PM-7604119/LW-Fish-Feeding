"""
Local PANN (Pretrained Audio Neural Network) reimplementation.
No dependency on U-FFIA or torchlibrosa. Uses torchaudio for mel spectrogram.
Compatible with audioset_tagging_cnn checkpoint keys for Cnn10/Cnn14.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    """PANN-style ConvBlock: two 3x3 convs + BN + ReLU, then pool in forward."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weight()

    def _init_weight(self):
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        else:
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class LogMelFrontend(nn.Module):
    """Waveform -> log-mel spectrogram. PANN defaults: 32kHz, 1024 hop 320, 64 mels."""

    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=mel_bins,
            power=2.0,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.melspec(wav)
        x = torch.clamp(x, min=1e-10).log()
        return x.unsqueeze(1)


class Cnn10Backbone(nn.Module):
    """Cnn10 backbone. Expects input [B, 1, mel_bins, time]. Output [B, 512]."""

    def __init__(self, classes_num: int = 527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self._init_weight()

    def _init_weight(self):
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)
        x = x.permute(0, 3, 2, 1)
        x = self.bn0(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return embedding


class Cnn14Backbone(nn.Module):
    """Cnn14 backbone. Expects input [B, 1, mel_bins, time]. Output [B, 2048]."""

    def __init__(self, classes_num: int = 527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        self._init_weight()

    def _init_weight(self):
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)
        x = x.permute(0, 3, 2, 1)
        x = self.bn0(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return embedding
