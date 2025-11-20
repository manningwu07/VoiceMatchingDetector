import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path

class VoicePipeline(Dataset):
    def __init__(self, data_root, samples_per_epoch=5000, augment=False, speaker_ids=None):
        """
        data_root: Path to folder containing LibriSpeech (e.g., ./data)
        speaker_ids: List of speaker IDs to include (for Train/Val split)
        """
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.speakers = {}
        
        print(f"Scanning dataset at {data_root}...")
        # Performance: Use pathlib's rglob which is often faster on macOS than glob
        # We look for flac files in train-clean-360
        search_path = Path(data_root)
        files = list(search_path.rglob("*.flac"))
        
        if len(files) == 0:
            raise RuntimeError(f"No .flac files found in {data_root}. Did you download the data?")

        print(f"Found {len(files)} audio files. Indexing speakers...")

        for f_path in files:
            # Structure: .../SPEAKER_ID/CHAPTER_ID/FILE.flac
            # Parent is Chapter, Grandparent is Speaker
            s_id = f_path.parent.parent.name
            
            # Only include if it's in our allowed list (or if list is None, take all)
            if speaker_ids is None or s_id in speaker_ids:
                if s_id not in self.speakers:
                    self.speakers[s_id] = []
                self.speakers[s_id].append(str(f_path))
        
        # If speaker_ids was None, we populate it now
        self.speaker_ids = list(self.speakers.keys())
        print(f"Indexed {len(self.speaker_ids)} speakers.")

    def _preprocess(self, path):
        # Load -> Resample(16k) -> Mono -> Normalize
        try:
            sig, fs = torchaudio.load(path)
            if fs != 16000:
                sig = torchaudio.transforms.Resample(fs, 16000)(sig)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
            
            # Peak Normalization
            sig = sig / (torch.max(torch.abs(sig)) + 1e-9)
            return sig
        except Exception as e:
            # Fallback for corrupt files
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, 16000)

    def _augment_audio(self, sig):
        # 1. Additive Noise (Simulates HVAC, computer fans)
        if random.random() > 0.5:
            noise_level = random.uniform(0.001, 0.015)
            noise = torch.randn_like(sig) * noise_level
            sig = sig + noise
            
        # 2. Random Volume Drop (Simulates moving away from mic)
        if random.random() > 0.5:
            vol_scale = random.uniform(0.5, 1.0)
            sig = sig * vol_scale

        return sig

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        is_same = random.random() > 0.5
        
        if is_same:
            spk = random.choice(self.speaker_ids)
            # Ensure speaker has at least 2 files
            if len(self.speakers[spk]) < 2:
                # Fallback strategy if a speaker has only 1 file (rare)
                f1 = f2 = self.speakers[spk][0]
            else:
                f1, f2 = random.sample(self.speakers[spk], 2)
            label = 1.0
        else:
            spk1, spk2 = random.sample(self.speaker_ids, 2)
            f1 = random.choice(self.speakers[spk1])
            f2 = random.choice(self.speakers[spk2])
            label = 0.0

        w1 = self._preprocess(f1)
        w2 = self._preprocess(f2)

        if self.augment:
            w1 = self._augment_audio(w1)
            w2 = self._augment_audio(w2)

        # Cut/Pad to exactly 3 seconds (48k samples)
        max_len = 48000
        def fix_len(w):
            if w.shape[1] > max_len:
                # Random crop for training variation
                start = random.randint(0, w.shape[1] - max_len)
                return w[:, start:start+max_len]
            else:
                pad = max_len - w.shape[1]
                return torch.nn.functional.pad(w, (0, pad))

        return fix_len(w1), fix_len(w2), torch.tensor([label], dtype=torch.float32)