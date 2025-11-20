import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class VerificationNet(nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()
        
        # 1. Pretrained Encoder (ECAPA-TDNN)
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa"
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 2. Custom Verification Head (MLP)
        # Input: Concatenation of (u, v, |u-v|, u*v) -> 192 * 4 features
        self.embedding_dim = 192
        input_dim = self.embedding_dim * 4
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output Logit
        )

    def forward_single(self, x):
        # ECAPA expects input normalized, creates embedding
        # We strip the extra batch/time dims to get (B, 192)
        emb = self.encoder.encode_batch(x)
        return emb.squeeze(1)

    def forward(self, x1, x2):
        u = self.forward_single(x1)
        v = self.forward_single(x2)
        
        # Feature Engineering for the Head
        # This captures both distance and angle explicitly
        features = torch.cat([
            u, 
            v, 
            torch.abs(u - v), 
            u * v
        ], dim=1)
        
        logits = self.head(features)
        return logits

class ProbabilityCalibrator:
    """
    Post-hoc calibration using Platt Scaling (Logistic Regression).
    Converts raw logits into well-behaved probabilities.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        self.is_fitted = False

    def fit(self, logits, labels):
        # Expects logits as numpy array (N, 1) and labels as (N,)
        self.calibrator.fit(logits.reshape(-1, 1), labels)
        self.is_fitted = True

    def predict_proba(self, logits):
        if not self.is_fitted:
            # Fallback to standard sigmoid if not calibrated yet
            return 1 / (1 + np.exp(-logits))
        
        # Sklearn returns [prob_class_0, prob_class_1]
        return self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]