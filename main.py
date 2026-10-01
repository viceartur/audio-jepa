import os
from huggingface_hub import login
from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import copy
import math

from dataset import AbstractDataset
from whisper import Whisper
from encoder import AudioEncoder
from predictor import Predictor

CHECKPOINT_PATH = "best_model_weights.pth"
EMBED_DIM = 384
PREDICTOR_HIDDEN = 192

def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable is not set.")
    login(token=hf_token)

    # Dataset
    ds = load_dataset("openslr/librispeech_asr", "clean")
    # keys: ['test', 'train.100', 'train.360', 'validation']

    # Whisper
    w = Whisper()
    whisper_model = w.whisper_model
    processor = w.processor

    # Development
    # ds_train_100 = AbstractDataset(ds["train.100"].select(range(100)), processor)
    # ds_val = AbstractDataset(ds["validation"].select(range(20)), processor)
    # ds_test = AbstractDataset(ds["test"].select(range(20)), processor)

    # Production
    ds_train_100 = AbstractDataset(ds["train.100"], processor)
    ds_val = AbstractDataset(ds["validation"], processor)
    ds_test = AbstractDataset(ds["test"], processor)

    # JEPA model components
    ctx_encoder = AudioEncoder(embed_dim=EMBED_DIM).to(whisper_model.device)
    predictor = Predictor(encoder_dim=EMBED_DIM, embed_dim=PREDICTOR_HIDDEN).to(whisper_model.device)

    # Target Encoder's architecture is the same as Context Encoder,
    # gradients are disabled, parameters are updated via EMA.
    target_encoder = copy.deepcopy(ctx_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Match input and label size for the dataset
    def whisper_collate_fn(batch):
        input_features = [item["input_features"].squeeze(0) for item in batch]
        input_features = torch.stack(input_features) # B x 128 x 3000
        labels = [item["label"] for item in batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        speaker_ids = [item["speaker_id"] for item in batch]
        chapter_ids = [item["chapter_id"] for item in batch]
        
        return {
            "input_features": input_features,
            "labels": labels_padded,
            "speaker_id": speaker_ids,
            "chapter_id": chapter_ids
        }

    batch_size = 2**7
    train_100 = DataLoader(ds_train_100, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn, num_workers=8)
    val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn, num_workers=8)
    test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn, num_workers=8)

    # Training settings
    num_epochs = 20
    loss_fn = nn.MSELoss()

    # Optimizer
    def get_parameter_groups(model_list, weight_decay):
        decay, no_decay = [], []
        for model in model_list:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # No decay for biases and 1D (LayerNorm) parameters
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
            ]

    param_groups = get_parameter_groups([ctx_encoder, predictor], weight_decay=0.05)
    lr = 3e-4
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # Learning rate schedulers
    start_lr = 1e-6
    warmup_steps = 1000
    total_steps = num_epochs * len(train_100)
    cosine_steps = total_steps - warmup_steps
    start_factor = start_lr / lr
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    momentum_schedule = [
        1.0 - 0.004 * (math.cos(math.pi * i / total_steps) + 1) / 2
        for i in range(total_steps)
    ]

    # Training and Validation
    best_val_loss = float('inf')
    global_step = 0

    for e in range(num_epochs):
        train_loss = 0
        ctx_encoder.train()
        predictor.train()

        print(f'Epoch #{e}...\n')

        for batch in tqdm(train_100):
            ctx_encoder.zero_grad()
            predictor.zero_grad()

            # Input mel-spectrogram
            x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16, cache_enabled=True):
                # Context forward path
                x_unmasked, mask_indices, bool_mask = ctx_encoder(x)
                preds = predictor(x_unmasked, mask_indices=mask_indices, bool_mask=bool_mask)

                # Target forward path
                with torch.no_grad():
                    targets = target_encoder(x, mask_indices=mask_indices)

                loss = loss_fn(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ctx_encoder.parameters()) + list(predictor.parameters()),
                max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            # Target Encoder EMA Update
            current_m = momentum_schedule[global_step]
            with torch.no_grad():
                for param_context, param_target in zip(ctx_encoder.parameters(), target_encoder.parameters()):
                    param_target.data.mul_(current_m).add_(param_context.data, alpha=1.0 - current_m)

            global_step += 1

        print(f'Epoch {e}. Total train loss: {train_loss}')

        val_loss = 0
        ctx_encoder.eval()
        predictor.eval()
        target_encoder.eval()

        with torch.no_grad():
            for batch in tqdm(val):
                    x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)

                    with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16, cache_enabled=True):
                        x_unmasked, mask_indices, bool_mask = ctx_encoder(x)
                        preds = predictor(x_unmasked, mask_indices=mask_indices, bool_mask=bool_mask)
                        targets = target_encoder(x, mask_indices=mask_indices)
                        loss = loss_fn(preds, targets)

                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val)
        print(f'Epoch {e}. Total val loss: {val_loss} | Avg val loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            print("Validation loss decreased. Saving model...")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': e,
                'ctx_encoder_state_dict': ctx_encoder.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'target_encoder_state_dict': target_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
                }, CHECKPOINT_PATH)

    # Testing
    print("Testing phase. Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    ctx_encoder.load_state_dict(checkpoint['ctx_encoder_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])

    test_loss = 0
    ctx_encoder.eval()
    predictor.eval()
    target_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(test):
                x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)

                with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16, cache_enabled=True):
                    x_unmasked, mask_indices, bool_mask = ctx_encoder(x)
                    preds = predictor(x_unmasked, mask_indices=mask_indices, bool_mask=bool_mask)    
                    targets = target_encoder(x, mask_indices=mask_indices)     
                    loss = loss_fn(preds, targets)

                test_loss += loss.item()

    print(f'Test loss: {test_loss}')
        

if __name__=="__main__":
    main()