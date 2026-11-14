import os
from huggingface_hub import login
from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from dotenv import load_dotenv

from model.utils.dataset import AbstractDataset
from jepa_module.whisper import Whisper
from jepa_module.encoder import AudioEncoder
from model.ctc_module.ctc_head import CTCHead
from utils.ctc_vocab import (vocab, blank_id, id_to_char)


ENCODER_CHECKPOINT_PATH = "../best_model_weights_20_epochs_clipping.pth"
BEST_MODEL_WEIGHTS_CHECKPOINT = "best_model_weights_ctc_loss.pth"
EMBED_DIM = 384

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")
login(token=hf_token)

# Whisper processor
# Used as an audio and text tokenizer
w = Whisper()
whisper_model = w.whisper_model
processor = w.processor

# Dataset
# keys: ['test', 'train.100', 'train.360', 'validation']
# train.360 used for fine-tuning with 80/10/10 split
ds = load_dataset("openslr/librispeech_asr", "clean")["train.360"]

# 80/10/10 split sorted by speakers
unique_speakers = sorted(list(set(ds["speaker_id"])))
num_speakers = len(unique_speakers)
train_cutoff = int(0.80 * num_speakers)
val_cutoff = int(0.90 * num_speakers)

train_speakers = set(unique_speakers[:train_cutoff])
val_speakers = set(unique_speakers[train_cutoff:val_cutoff])
test_speakers = set(unique_speakers[val_cutoff:])

ds_train_360_split = ds.filter(lambda x: x["speaker_id"] in train_speakers)
ds_val_360_split = ds.filter(lambda x: x["speaker_id"] in val_speakers)
ds_test_360_split = ds.filter(lambda x: x["speaker_id"] in test_speakers)

# Development
# ds_train = AbstractDataset(ds_train_360_split.select(range(100)), processor, tokenize_text_ctc=True)
# ds_val = AbstractDataset(ds_val_360_split.select(range(20)), processor, tokenize_text_ctc=True)
# ds_test = AbstractDataset(ds_test_360_split.select(range(20)), processor, tokenize_text_ctc=True)

# Production
ds_train = AbstractDataset(ds_train_360_split, processor, tokenize_text_ctc=True)
ds_val = AbstractDataset(ds_val_360_split, processor, tokenize_text_ctc=True)
ds_test = AbstractDataset(ds_test_360_split, processor, tokenize_text_ctc=True)

# Load JEPA pretrained encoder
ctx_encoder = AudioEncoder(embed_dim=EMBED_DIM).to(whisper_model.device)
checkpoint = torch.load(ENCODER_CHECKPOINT_PATH)
ctx_encoder.load_state_dict(checkpoint['ctx_encoder_state_dict'])
for param in ctx_encoder.parameters():
    param.requires_grad = False

ctc_head = CTCHead(embed_dim=EMBED_DIM).to(whisper_model.device)

def whisper_collate_fn_ctc(batch):
        input_features = [item["input_features"].squeeze(0) for item in batch]
        input_features = torch.stack(input_features) # B x 128 x 3000
        labels = [item["label"] for item in batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

        upsample_factor = 3
        input_lengths = torch.tensor(
            [(item["input_features"].shape[-1] // 16) * upsample_factor for item in batch],
            dtype=torch.long
        )
            
        target_lengths = torch.tensor(
            [len(label) for label in labels], 
            dtype=torch.long
        )

        speaker_ids = [item["speaker_id"] for item in batch]
        chapter_ids = [item["chapter_id"] for item in batch]
            
        return {
            "input_features": input_features,
            "labels": labels_padded,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
            "speaker_id": speaker_ids,
            "chapter_id": chapter_ids
            }

batch_size = 2**8
train_100 = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn_ctc, num_workers=8)
val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn_ctc, num_workers=8)
test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=whisper_collate_fn_ctc, num_workers=8)


def ctc_train():
    # Training settings
    num_epochs = 20
    ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=False)

    # Optimizer
    lr = 3e-4
    optimizer = torch.optim.Adam(params=ctc_head.parameters(), lr=lr)

    # Training and Validation
    best_val_loss = float('inf')

    for e in range(num_epochs):
        train_loss = 0.0
        ctc_head.train()
        ctx_encoder.eval()

        print(f"Epoch #{e}...\n")

        for batch in tqdm(train_100):
            optimizer.zero_grad(set_to_none=True)

            x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)

            targets = batch["labels"].to(whisper_model.device)
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            with torch.no_grad():
                x_out = ctx_encoder(x, use_mask=False)

            log_probs = ctc_head(x_out)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_100)

        print(f"Epoch {e}. Average train loss: {avg_train_loss:.4f}")

        val_loss = 0
        ctc_head.eval()

        with torch.no_grad():
            for batch in tqdm(val):
                x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)
                targets = batch["labels"].to(whisper_model.device)
                input_lengths = batch["input_lengths"]
                target_lengths = batch["target_lengths"]
                x_out = ctx_encoder(x, use_mask=False)
                log_probs = ctc_head(x_out)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val)
        print(f'Epoch {e}. Total val loss: {val_loss} | Avg val loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            print("Validation loss decreased. Saving CTCHead weights...")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': e,
                'ctc_head_state_dict': ctc_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
                }, BEST_MODEL_WEIGHTS_CHECKPOINT)

    print("Testing phase. Loading model...")
    checkpoint = torch.load(BEST_MODEL_WEIGHTS_CHECKPOINT)
    ctc_head.load_state_dict(checkpoint['ctc_head_state_dict'])

    test_loss = 0
    ctc_head.eval()

    with torch.no_grad():
        for batch in tqdm(test):
                x = batch["input_features"].unsqueeze(1).to(device=whisper_model.device, dtype=torch.float32)
                targets = batch["labels"].to(whisper_model.device)
                input_lengths = batch["input_lengths"]
                target_lengths = batch["target_lengths"]
                x_out = ctx_encoder(x, use_mask=False)
                log_probs = ctc_head(x_out)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

                test_loss += loss.item()

    print(f'Test loss: {test_loss}')

def ctc_greedy_decode(predictions, input_lengths, blank_id):
    """
    predictions:     T x B x V
    input_lengths: 1 x B (value=562)
    blank_id: special token
    """
    decoded = [] # B x Tokens

    for b in range(predictions.size(1)):
        sequence = predictions[
            :input_lengths[b], # 562
            b
            ].argmax(dim=-1).tolist()

        tokens = []
        previous = None

        for token in sequence:
            if token != blank_id and token != previous:
                tokens.append(token)

            previous = token

        decoded.append(tokens)

    return decoded

def decode_labels(tokens):
    output = "".join([id_to_char[t] for t in tokens])
    return output


def ctc_inference(use_beam_search=False):
    checkpoint = torch.load(BEST_MODEL_WEIGHTS_CHECKPOINT)
    ctc_head.load_state_dict(checkpoint["ctc_head_state_dict"])
    ctx_encoder.eval()
    ctc_head.eval()

    # Inference
    with torch.inference_mode():
        for batch in test:
            x = batch["input_features"].unsqueeze(1).to(
                device=whisper_model.device,
                dtype=torch.float32) # 128 x 3000
            input_lengths = batch["input_lengths"] # 1 x B (each item = 562)
            x_out = ctx_encoder(x, use_mask=False) # B x 1496 x D
            preds = ctc_head(x_out) # 562 (T) x B x V

            if use_beam_search:
                pass
            else:
                predictions = ctc_greedy_decode(
                    preds,
                    input_lengths,
                    blank_id
                )

            # Display results
            for i, prediction in enumerate(predictions):
                ground_truth = batch["labels"][i][
                    :batch["target_lengths"][i]
                ].tolist()

                predicted_text = decode_labels(prediction)
                ground_truth_text = decode_labels(ground_truth)

                print(f"\nSpeaker: {batch['speaker_id'][i]}")
                print(f"Prediction:   {predicted_text}")
                print(f"Ground truth: {ground_truth_text}")

            break
        

if __name__=="__main__":
    ctc_inference(use_beam_search=False)