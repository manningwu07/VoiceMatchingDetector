# src/model.py
import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from src.config import PRETRAINED_MODEL_NAME, DEVICE, EMBEDDING_DIM

class SpeakerVerifier(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=128):
        super().__init__()
        self.encoder = EncoderClassifier.from_hparams(
            source=PRETRAINED_MODEL_NAME,
            run_opts={"device": DEVICE},
        )
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Small head: cosine sim -> logit
        self.head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, wav1, wav2):
        """
        wav1, wav2: [B, T] audio tensors
        Returns: logits [B, 1]
        """
        with torch.no_grad():
            emb1 = self.encoder.encode_batch(wav1.to(DEVICE))  # [B, 1, D]
            emb2 = self.encoder.encode_batch(wav2.to(DEVICE))
        
        emb1 = torch.nn.functional.normalize(emb1.squeeze(1), dim=-1)  # [B, D]
        emb2 = torch.nn.functional.normalize(emb2.squeeze(1), dim=-1)
        
        cos_sim = torch.sum(emb1 * emb2, dim=-1, keepdim=True)  # [B, 1]
        logits = self.head(cos_sim)  # [B, 1]
        return logits

def get_embeddings(wav, encoder):
    """Extract speaker embeddings from wav [T]."""
    wav_batch = wav.unsqueeze(0)  # [1, T]
    with torch.no_grad():
        emb = encoder.encode_batch(wav_batch.to(DEVICE))  # [1, 1, D]
    return torch.nn.functional.normalize(emb.squeeze().cpu(), dim=0)  # [D]