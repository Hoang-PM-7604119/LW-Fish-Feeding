"""
Fusion methods for combining video and audio modalities.

Implements:
- Concat: Simple concatenation + MLP
- Cross-Attention: Bi-directional cross-modal attention
- Gated Fusion: Learnable gating mechanism
- Joint Cross-Attention: Multi-layer self + cross attention
- MBT: Multimodal Bottleneck Transformer
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention for intra-modality refinement."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            mask: optional attention mask
        Returns:
            output: [B, T, D]
            attn_weights: [B, H, T, T]
        """
        bsz, t, dim = x.shape
        
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        q = q.view(bsz, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, t, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, t, dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class CrossAttention(nn.Module):
    """Cross-attention for inter-modality fusion."""
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, T_q, D_q]
            key:   [B, T_k, D_k]
            value: [B, T_v, D_v]
        Returns:
            output: [B, T_q, D]
            attn_weights: [B, H, T_q, T_k]
        """
        bsz, t_q, _ = query.size()
        t_k = key.size(1)
        
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        q = q.view(bsz, t_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, t_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, t_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, t_q, self.embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.
    
    Concatenates pooled video and audio features, then applies MLP.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Output embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embed_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.video_proj = nn.Linear(video_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            video_features: [B, T_v, D_v]
            audio_features: [B, T_a, D_a]
        Returns:
            fused: [B, embed_dim]
            info: Empty dict (for consistency with other fusion methods)
        """
        v = self.video_proj(video_features.mean(dim=1))
        a = self.audio_proj(audio_features.mean(dim=1))
        fused = self.fusion(torch.cat([v, a], dim=-1))
        return fused, {}


class CrossAttentionFusion(nn.Module):
    """
    Bi-directional cross-attention fusion.
    
    Video attends to audio and audio attends to video.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_proj = nn.Linear(video_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        self.video_to_audio = CrossAttention(embed_dim, embed_dim, embed_dim, num_heads, dropout)
        self.audio_to_video = CrossAttention(embed_dim, embed_dim, embed_dim, num_heads, dropout)
        
        self.video_norm = nn.LayerNorm(embed_dim)
        self.audio_norm = nn.LayerNorm(embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            video_features: [B, T_v, D_v]
            audio_features: [B, T_a, D_a]
        Returns:
            fused: [B, embed_dim]
            attn: dict with attention weights
        """
        v = self.video_proj(video_features)
        a = self.audio_proj(audio_features)
        
        # Video attends to audio
        v_cross, attn_v2a = self.video_to_audio(query=v, key=a, value=a)
        v = self.video_norm(v + v_cross)
        
        # Audio attends to video
        a_cross, attn_a2v = self.audio_to_video(query=a, key=v, value=v)
        a = self.audio_norm(a + a_cross)
        
        v_pooled = v.mean(dim=1)
        a_pooled = a.mean(dim=1)
        
        fused = self.fusion(torch.cat([v_pooled, a_pooled], dim=-1))
        attn = {"video_to_audio": attn_v2a, "audio_to_video": attn_a2v}
        return fused, attn


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable gates.
    
    Uses sigmoid gates to control information flow from each modality.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Output embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embed_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.video_proj = nn.Linear(video_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.candidate = nn.Linear(embed_dim * 2, embed_dim)
        self.residual = nn.Linear(embed_dim * 2, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            video_features: [B, T_v, D_v]
            audio_features: [B, T_a, D_a]
        Returns:
            fused: [B, embed_dim]
            info: Empty dict
        """
        v = self.video_proj(video_features.mean(dim=1))
        a = self.audio_proj(audio_features.mean(dim=1))
        
        z = torch.cat([v, a], dim=-1)
        
        g = torch.sigmoid(self.gate(z))
        h = torch.tanh(self.candidate(z))
        z_res = self.residual(z)
        
        fused = g * h + (1.0 - g) * z_res
        fused = self.norm(self.dropout(fused))
        return fused, {}


class JointCrossAttentionFusion(nn.Module):
    """
    Joint cross-attention fusion with multiple layers.
    
    Each layer contains:
    - Self-attention for each modality
    - Bi-directional cross-attention
    - Feed-forward networks
    - Residual connections and layer normalization
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout probability
        use_residual: Use residual connections
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.video_proj = nn.Linear(video_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "video_self_attn": SelfAttention(embed_dim, num_heads, dropout),
                "audio_self_attn": SelfAttention(embed_dim, num_heads, dropout),
                "video_to_audio_cross_attn": CrossAttention(
                    embed_dim, embed_dim, embed_dim, num_heads, dropout
                ),
                "audio_to_video_cross_attn": CrossAttention(
                    embed_dim, embed_dim, embed_dim, num_heads, dropout
                ),
                "video_self_norm": nn.LayerNorm(embed_dim),
                "audio_self_norm": nn.LayerNorm(embed_dim),
                "video_cross_norm": nn.LayerNorm(embed_dim),
                "audio_cross_norm": nn.LayerNorm(embed_dim),
                "video_ffn_norm": nn.LayerNorm(embed_dim),
                "audio_ffn_norm": nn.LayerNorm(embed_dim),
                "video_ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
                "audio_ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_layers)
        ])
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, list]]:
        """
        Args:
            video_features: [B, T_v, D_v]
            audio_features: [B, T_a, D_a]
        Returns:
            fused: [B, embed_dim]
            attention_weights: dict of attention maps
        """
        video_emb = self.video_proj(video_features)
        audio_emb = self.audio_proj(audio_features)
        
        attn_weights: Dict[str, list] = {
            "video_self": [],
            "audio_self": [],
            "video_cross": [],
            "audio_cross": [],
        }
        
        for layer in self.layers:
            # Self-attention
            video_self, video_self_attn = layer["video_self_attn"](video_emb)
            if self.use_residual:
                video_self = video_emb + video_self
            video_self = layer["video_self_norm"](video_self)
            
            audio_self, audio_self_attn = layer["audio_self_attn"](audio_emb)
            if self.use_residual:
                audio_self = audio_emb + audio_self
            audio_self = layer["audio_self_norm"](audio_self)
            
            # Cross-attention
            video_cross, video_cross_attn = layer["video_to_audio_cross_attn"](
                query=video_self, key=audio_self, value=audio_self
            )
            if self.use_residual:
                video_cross = video_self + video_cross
            video_cross = layer["video_cross_norm"](video_cross)
            
            audio_cross, audio_cross_attn = layer["audio_to_video_cross_attn"](
                query=audio_self, key=video_self, value=video_self
            )
            if self.use_residual:
                audio_cross = audio_self + audio_cross
            audio_cross = layer["audio_cross_norm"](audio_cross)
            
            # FFN
            video_ffn = layer["video_ffn"](video_cross)
            if self.use_residual:
                video_ffn = video_cross + video_ffn
            video_emb = layer["video_ffn_norm"](video_ffn)
            
            audio_ffn = layer["audio_ffn"](audio_cross)
            if self.use_residual:
                audio_ffn = audio_cross + audio_ffn
            audio_emb = layer["audio_ffn_norm"](audio_ffn)
            
            attn_weights["video_self"].append(video_self_attn)
            attn_weights["audio_self"].append(audio_self_attn)
            attn_weights["video_cross"].append(video_cross_attn)
            attn_weights["audio_cross"].append(audio_cross_attn)
        
        # Temporal pooling
        video_pooled = video_emb.mean(dim=1)
        audio_pooled = audio_emb.mean(dim=1)
        
        combined = torch.cat([video_pooled, audio_pooled], dim=-1)
        fused = self.fusion_proj(combined)
        fused = self.output_norm(fused)
        fused = self.output_proj(fused)
        
        return fused, attn_weights


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        return x


class MBTFusion(nn.Module):
    """
    Multimodal Bottleneck Transformer (MBT) fusion.
    
    Uses learnable bottleneck tokens to compress multimodal information.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_bottleneck_tokens: Number of bottleneck tokens
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_bottleneck_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_proj = nn.Linear(video_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.bottleneck = nn.Parameter(torch.randn(num_bottleneck_tokens, embed_dim))
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            video_features: [B, T_v, D_v]
            audio_features: [B, T_a, D_a]
        Returns:
            fused: [B, embed_dim]
            info: dict with bottleneck tokens
        """
        bsz = video_features.size(0)
        
        v = self.video_proj(video_features)
        a = self.audio_proj(audio_features)
        
        # Expand bottleneck tokens
        b_tokens = self.bottleneck.unsqueeze(0).expand(bsz, -1, -1)
        
        # Concatenate [video, bottleneck, audio]
        x = torch.cat([v, b_tokens, a], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        # Extract and pool bottleneck tokens
        t_v = v.size(1)
        t_b = self.num_bottleneck_tokens
        
        bottleneck_slice = x[:, t_v : t_v + t_b, :]
        fused = bottleneck_slice.mean(dim=1)
        fused = self.output_norm(fused)
        
        info = {"bottleneck_tokens": bottleneck_slice}
        return fused, info


def get_fusion_method(
    fusion_type: str,
    video_dim: int,
    audio_dim: int,
    embed_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to get fusion method by name.
    
    Args:
        fusion_type: Type of fusion ('concat', 'cross_attention', 'gated', 
                    'joint_cross_attention', 'mbt')
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        embed_dim: Embedding dimension
        **kwargs: Additional arguments for specific fusion methods
        
    Returns:
        Fusion module
        
    Example:
        >>> fusion = get_fusion_method('joint_cross_attention', 1024, 512, 512, num_heads=8)
        >>> fusion = get_fusion_method('mbt', 1024, 512, 512, num_bottleneck_tokens=4)
    """
    fusion_type = fusion_type.lower()
    
    if fusion_type == 'concat':
        return ConcatFusion(video_dim, audio_dim, embed_dim, **kwargs)
    elif fusion_type in ['cross_attention', 'crossattn']:
        return CrossAttentionFusion(video_dim, audio_dim, embed_dim, **kwargs)
    elif fusion_type == 'gated':
        return GatedFusion(video_dim, audio_dim, embed_dim, **kwargs)
    elif fusion_type in ['joint_cross_attention', 'joint_crossattn', 'jca']:
        return JointCrossAttentionFusion(video_dim, audio_dim, embed_dim, **kwargs)
    elif fusion_type == 'mbt':
        return MBTFusion(video_dim, audio_dim, embed_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Available: concat, cross_attention, gated, joint_cross_attention, mbt"
        )


if __name__ == '__main__':
    """Test fusion methods."""
    print("="*80)
    print("Testing Fusion Methods")
    print("="*80)
    
    batch_size = 2
    video_features = torch.randn(batch_size, 1, 1024)
    audio_features = torch.randn(batch_size, 1, 512)
    
    print(f"\nVideo features: {video_features.shape}")
    print(f"Audio features: {audio_features.shape}")
    
    # Test all fusion methods
    fusion_types = ['concat', 'cross_attention', 'gated', 'joint_cross_attention', 'mbt']
    
    for fusion_type in fusion_types:
        print(f"\n{'-'*80}")
        print(f"Testing {fusion_type} Fusion")
        print(f"{'-'*80}")
        try:
            fusion = get_fusion_method(fusion_type, 1024, 512, 512)
            fused, info = fusion(video_features, audio_features)
            print(f"✓ {fusion_type} output shape: {fused.shape}")
            print(f"  Parameters: {sum(p.numel() for p in fusion.parameters()):,}")
        except Exception as e:
            print(f"✗ {fusion_type} failed: {e}")
    
    print(f"\n{'='*80}")
    print("✓ Fusion methods testing completed!")
    print(f"{'='*80}")
