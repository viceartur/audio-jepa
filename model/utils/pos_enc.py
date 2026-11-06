import torch

from model.jepa_module.whisper import Whisper

w = Whisper()
whisper_model = w.whisper_model

# 2d Sin-Cos positional encoding
def get_2d_sincos_pos_enc(num_patches_f=8, num_patches_t=187, embed_dim=None, n=10000):
    half_dim = embed_dim // 2 # D//2 = H
    
    pos_f = torch.arange(num_patches_f, dtype=torch.float32).unsqueeze(1) # 8 x 1
    pos_t = torch.arange(num_patches_t, dtype=torch.float32).unsqueeze(1) # 187 x 1
    dim_t = torch.arange(half_dim // 2, dtype=torch.float32)
    omega = 1.0 / (n ** (2 * dim_t / half_dim))

    # Frequency encoding
    out_f = torch.zeros((num_patches_f, half_dim)) # 8 x H
    out_f[:, 0::2] = torch.sin(pos_f * omega)
    out_f[:, 1::2] = torch.cos(pos_f * omega)
    
    # Time encoding
    out_t = torch.zeros((num_patches_t, half_dim)) # 187 x H
    out_t[:, 0::2] = torch.sin(pos_t * omega)
    out_t[:, 1::2] = torch.cos(pos_t * omega)
    
    emb_f = out_f.unsqueeze(1).repeat(1, num_patches_t, 1) # 8 x 1 x H -> 8 x 187 x H
    emb_t = out_t.unsqueeze(0).repeat(num_patches_f, 1, 1) # 1 x 187 x H -> 8 x 187 x H
    pos_2d = torch.cat([emb_f, emb_t], dim=-1)  # 8 x 187 x D
    pos_2d = pos_2d.view(-1, embed_dim) # (F * T) x D
    pos_2d = pos_2d.to(dtype=whisper_model.dtype, device=whisper_model.device)
    
    return pos_2d 