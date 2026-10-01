import torch

def get_mask_indices_2d(
        mel_tensor,
        mask_type="random",
        mask_ratio=0.6,
        mask_selection="static",
        patch_h=16,
        patch_w=16,
    ):

    if mask_selection == "static":
        p = mask_ratio
    elif mask_selection == "uniform":
        p = torch.empty(1).uniform_(0.4, mask_ratio).item()
    elif mask_selection == "normal":
        p = torch.empty(1).normal_(mean=mask_ratio, std=0.1).clamp_(0.15, 1.0).item()
    elif mask_selection == "poisson":
        p = torch.poisson(torch.tensor([mask_ratio * 10.0])).item() / 10.0
        p = min(max(p, 0.0), 1.0)
    
    B, C, F, T = mel_tensor.shape # B x Channel(=1) x Freq dim x Time dim (128 x 3000)
    num_patches_f = F // patch_h # 128//16 = 8
    num_patches_t = T // patch_w # 3000//16 = 187
    num_total_patches = num_patches_f * num_patches_t # 8*187 = 1496
    
    if mask_type == "random":
        num_mask = int(p * num_total_patches)
        mask_indices = torch.rand(B, num_total_patches, device=mel_tensor.device).argsort(dim=1)[:, :num_mask]
        
    elif mask_type == "time":
        num_mask_t = int(p * num_patches_t)
        t_indices = torch.rand(B, num_patches_t, device=mel_tensor.device).argsort(dim=1)[:, :num_mask_t]
        grid_mask = torch.zeros((B, num_patches_f, num_patches_t), dtype=torch.bool, device=mel_tensor.device)
        batch_idx = torch.arange(B, device=mel_tensor.device)[:, None]
        grid_mask[batch_idx, :, t_indices] = True
        mask_indices = grid_mask.view(B, -1).nonzero()[:, 1].view(B, -1)
        
    elif mask_type == "freq":
        num_mask_f = int(p * num_patches_f)
        f_indices = torch.rand(B, num_patches_f, device=mel_tensor.device).argsort(dim=1)[:, :num_mask_f]
        grid_mask = torch.zeros((B, num_patches_f, num_patches_t), dtype=torch.bool, device=mel_tensor.device)
        batch_idx = torch.arange(B, device=mel_tensor.device)[:, None]
        grid_mask[batch_idx, f_indices, :] = True
        mask_indices = grid_mask.view(B, -1).nonzero()[:, 1].view(B, -1)
        
    elif mask_type == "time_freq":
        x = 1.0 - torch.sqrt(torch.tensor(max(0.0, 1.0 - p))).item()
        num_t = int(x * num_patches_t)
        num_f = int(x * num_patches_f)
        t_indices = torch.rand(B, num_patches_t, device=mel_tensor.device).argsort(dim=1)[:, :num_t]
        f_indices = torch.rand(B, num_patches_f, device=mel_tensor.device).argsort(dim=1)[:, :num_f]
        grid_mask = torch.zeros((B, num_patches_f, num_patches_t), dtype=torch.bool, device=mel_tensor.device)
        batch_idx = torch.arange(B, device=mel_tensor.device)[:, None]
        grid_mask[batch_idx, :, t_indices] = True
        grid_mask[batch_idx, f_indices, :] = True
        mask_indices = grid_mask.view(B, -1).nonzero()[:, 1].view(B, -1)

    return mask_indices # B x (D*mask_ratio)