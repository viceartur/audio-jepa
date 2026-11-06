import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.utils.pos_enc import get_2d_sincos_pos_enc
from model.utils.mask import get_mask_indices_2d

class PatchEmbed(nn.Module):
    def __init__(self, input_size=(128, 3000), patch_size=(16, 16), in_chans=1, embed_dim=None):
        super().__init__()
        mel_h, mel_width = input_size
        patch_height, patch_width = patch_size
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches_h = mel_h // patch_height
        self.num_patches_w = mel_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings buffer
        pos_embed = get_2d_sincos_pos_enc(self.num_patches_h, self.num_patches_w, embed_dim) # B x 1496 x D
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x): # B x 1 x 128 x 3000
        h = self.proj(x) # B x D x (128//16=8) x (3000//16=187)
        h = h.flatten(2).transpose(1, 2) # B x 1496 x D
        h = h + self.pos_embed
        return h


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_Base(nn.Module):
    def __init__(self, embed_dim=None, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.vit_encoder = ViT_Base(embed_dim=embed_dim)

    def forward(self, x, mask_indices=None):
        """
        mask_indices: Only passed for Target Encoder to extract targets.
        """
        x_embed = self.patch_embed(x) # B x L x D
        B, L, D = x_embed.shape

        # Mask input for Context Encoder
        if mask_indices is None:
            mask_indices = get_mask_indices_2d(x, mask_ratio=0.6) # B x L_masked ~ U(0.4,0.6)
            bool_mask = torch.zeros((B, L), dtype=torch.bool, device=x_embed.device)
            bool_mask.scatter_(dim=1, index=mask_indices, value=True)

            # Only process visible parts
            L_unmasked = L - mask_indices.shape[1]
            x_unmasked = x_embed[~bool_mask].view(B, L_unmasked, D) # B x L_unmasked x D
            x_out = self.vit_encoder(x_unmasked)
            
            return x_out, mask_indices, bool_mask # context    
      
        # Preserve the full input for Target Encoder
        x_out = self.vit_encoder(x_embed)

        # Get targets only
        batch_ids = torch.arange(B, device=x.device).unsqueeze(1)
        x_out = x_out[batch_ids, mask_indices, :]

        return x_out # targets
