import torch
from torch import nn
from pos_enc import get_2d_sincos_pos_enc

class Predictor(nn.Module):
    def __init__(self, encoder_dim=None, embed_dim=None, depth=6, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        pos_enc = get_2d_sincos_pos_enc(
            128//16,
            3000//16,
            embed_dim=embed_dim
        ) # (8*187=1496) x D, L x D
        self.register_buffer("_pos_enc_tensor", pos_enc, persistent=False) 

        self._mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ctx_proj = nn.Linear(encoder_dim, embed_dim)
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, encoder_dim)

        # Gaussian init
        nn.init.normal_(self._mask_token, std=0.02)

    def forward(self, x_ctx, mask_indices, bool_mask):
        """
        x_ctx: B x L_unmasked x D_enc
        mask_indices: B x L_masked
        bool_mask: B x L
        """
        B = x_ctx.shape[0]
        x_ctx = self.ctx_proj(x_ctx) # B x L_unmasked x D_embed
        
        L_unmasked = x_ctx.shape[1]
        pos_enc_expanded = self._pos_enc_tensor.unsqueeze(0).expand(B, -1, -1) # B x L x D_embed

        ctx_pos_embed = pos_enc_expanded[~bool_mask].view(B, L_unmasked, -1) # B x L_unmasked x D_embed
        target_pos_embed = self._pos_enc_tensor[mask_indices] # B x L_masked x D_embed
        x_ctx = x_ctx + ctx_pos_embed # B x L_unmasked x D_embed

        # Target tokens
        L_masked = mask_indices.shape[1]
        mask_tokens = self._mask_token.expand(B, L_masked, -1) # B x L_masked x D_embed
        mask_tokens = mask_tokens + target_pos_embed # B x L_masked x D_embed
        
        x = torch.cat([x_ctx, mask_tokens], dim=1) # B x L x D_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x) # B x 1496 x D_embed -> B x 1496 x D_enc
        preds = x[:, -L_masked:, :] # B x L_masked x D_enc

        return preds