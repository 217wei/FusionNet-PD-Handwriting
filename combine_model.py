import torch
import torch.nn as nn
from restormer_arch import TransformerBlock, OverlapPatchEmbed, CrossAttention


# ====== 單一 Transformer branch ======
class TransformerBranch(nn.Module):
    def __init__(self, in_c=1, embed_dim=64, num_blocks=3, num_heads=8, num_classes=2):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_c=in_c, embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=embed_dim, num_heads=num_heads,
                             ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias')
            for _ in range(num_blocks)
        ])
        print(num_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)    # (B, embed_dim, H, W)
        x = self.blocks(x)         # Transformer encode
        x = self.pool(x).flatten(1)  # (B, embed_dim)
        x = self.fc(x)             # (B, num_classes)
        return x


# ====== Combined Transformer model ======
class CombinedTransformer(nn.Module):
    def __init__(self, embed_dim=48, num_heads=4, num_blocks=3):
        super().__init__()
        self.modelA = TransformerBranch(in_c=1, embed_dim=embed_dim,
                                        num_blocks=num_blocks, num_heads=num_heads, num_classes=2)
        self.modelB = TransformerBranch(in_c=1, embed_dim=embed_dim,
                                        num_blocks=num_blocks, num_heads=num_heads, num_classes=2)
     
        # 加入雙向 Cross-Attention
        self.cross_attn1 = CrossAttention(embed_dim, num_heads)
        self.cross_attn2 = CrossAttention(embed_dim, num_heads)

        self.fuse_block = TransformerBlock(dim=embed_dim, num_heads=num_heads,
                                           ffn_expansion_factor=2.0, bias=False,
                                           LayerNorm_type='WithBias')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x1, x2):
        # 原圖與 FFT 分支
        f1 = self.modelA.patch_embed(x1)
        f1 = self.modelA.blocks(f1)
        f2 = self.modelB.patch_embed(x2)
        f2 = self.modelB.blocks(f2)

        # 雙向 cross attention
        out1 = self.cross_attn1(f1, f2)
        out2 = self.cross_attn2(f2, f1) 
        fused = out1 + out2

        fused = self.fuse_block(fused)
        fused = self.pool(fused).flatten(1)
        out = self.fc(fused)
        return out


# ====== make_model 用於 train2.py ======
def make_model(nonfft_weight=None, fft_weight=None):
    ## model = CombinedTransformer()
    model = CombinedTransformer(embed_dim=64, num_heads=8, num_blocks=4)
    return model


# ====== for train1 / train2 個別訓練用 ======
def make_model_single():
    model = TransformerBranch(in_c=1, embed_dim=64, num_blocks=4, num_heads=8, num_classes=2)
    return model


