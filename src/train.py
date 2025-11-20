import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import VoicePipeline
from src.model import VerificationNet, ProbabilityCalibrator
import sys

def train():
    # Detect Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}...")

    # 1. Setup Data
    train_ds = VoicePipeline("./data", samples_per_epoch=200, augment=True)
    train_loader = DataLoader(train_ds, batch_size=16, num_workers=0) # workers=0 for mac safety

    # 2. Setup Model
    model = VerificationNet(freeze_encoder=True).to(device)
    optimizer = optim.Adam(model.head.parameters(), lr=2e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    model.train()
    all_val_logits = []
    all_val_labels = []

    print("Starting training loop...")
    for epoch in range(5): # Short run for demo
        total_loss = 0
        for x1, x2, label in train_loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Collect for calibration (usually done on a separate validation set)
            # Moving to cpu/numpy for sklearn
            all_val_logits.extend(logits.detach().cpu().numpy().flatten())
            all_val_labels.extend(label.detach().cpu().numpy().flatten())

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 4. Post-Hoc Calibration (Platt Scaling)
    print("\nRunning Platt Scaling Calibration...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(np.array(all_val_logits), np.array(all_val_labels))
    
    print("Calibration complete.")
    
    # 5. Save
    torch.save(model.state_dict(), "voice_auth_model.pth")
    print("Model saved.")
    
    # Test a dummy prediction
    test_logit = 2.5
    prob = calibrator.predict_proba(np.array([test_logit]))
    print(f"Test: Logit {test_logit} -> Calibrated Probability {prob[0]*100:.2f}%")

if __name__ == "__main__":
    train()