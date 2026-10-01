from torch.utils.data import Dataset, DataLoader
import torch

class AbstractDataset(Dataset):
    def __init__(self, ds_split, processor):
        self.ds = ds_split
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        audio = sample["audio"]
        inputs = self.processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        )
        input_features = inputs.input_features
        label_out = self.processor(text=sample["text"])
        label = torch.tensor(label_out["input_ids"])
        
        return {
            "input_features": input_features,
            "label": label,
            "speaker_id": sample["speaker_id"],
            "chapter_id": sample["chapter_id"],
        }