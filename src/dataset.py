# src/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.audio_utils import preprocess_audio

class SpeakerPairDataset(Dataset):
    def __init__(self, csv_path, max_len=None):
        """
        csv_path: path to CSV with columns [audio_path_1, audio_path_2, label]
        label: 1 = same speaker, 0 = different speaker
        """
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path1, path2, label = row["audio_1"], row["audio_2"], row["label"]
        
        # Load and preprocess
        audio1 = preprocess_audio(path1)
        audio2 = preprocess_audio(path2)
        
        # Pad to max_len or truncate
        if self.max_len:
            audio1 = self._pad_or_truncate(audio1, self.max_len)
            audio2 = self._pad_or_truncate(audio2, self.max_len)
        
        return {
            "audio_1": torch.from_numpy(audio1).float(),
            "audio_2": torch.from_numpy(audio2).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }
    
    @staticmethod
    def _pad_or_truncate(audio, length):
        if len(audio) >= length:
            return audio[:length]
        else:
            return np.pad(audio, (0, length - len(audio)), mode="constant")

def collate_fn(batch):
    """Custom collate for variable-length audio."""
    # Just pad to max in batch
    max_len = max(b["audio_1"].shape[0] for b in batch)
    audio_1 = torch.stack([
        torch.nn.functional.pad(b["audio_1"], (0, max_len - b["audio_1"].shape[0]))
        for b in batch
    ])
    audio_2 = torch.stack([
        torch.nn.functional.pad(b["audio_2"], (0, max_len - b["audio_2"].shape[0]))
        for b in batch
    ])
    labels = torch.stack([b["label"] for b in batch])
    return {"audio_1": audio_1, "audio_2": audio_2, "label": labels}