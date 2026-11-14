import torch

vocab = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz") + ["'", " "]
char_to_id = {char: idx for idx, char in enumerate(vocab)}
id_to_char = {idx: char for idx, char in enumerate(vocab)}
blank_id = char_to_id["<blank>"]

def ctc_tokenizer(text):
    text = text.lower()
    unsupported = set(text) - set(char_to_id)
    if unsupported:
        raise ValueError(f"Unsupported characters: {unsupported}")

    return torch.tensor(
        [char_to_id[c] for c in text],
        dtype=torch.long
    )