import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from src.dataset import VoicePipeline
from src.model import VerificationNet, ProbabilityCalibrator
from pathlib import Path

# --- CONFIGURATION ---
DEVICE = "mps"
BATCH_SIZE = 64         # Increased batch size for stability
NUM_WORKERS = 4         # M4 Pro sweet spot
LEARNING_RATE = 5e-4  # Slightly lower LR for longer training
EPOCHS = 30             # Will take longer per epoch now
SAMPLES_PER_EPOCH = 20000 # We define an "epoch" as seeing 20k pairs

def train():
    print(f"🚀 Starting Large Scale Training on {DEVICE}")
    
    # 1. Discover Speakers First
    # We do a quick scan to get the ID list so we can split Train/Val
    print("Scanning data structure...")
    # Just looking at folders to get IDs quickly without loading all files yet
    data_path = Path("./data/LibriSpeech/train-clean-360")
    if not data_path.exists():
        print("Error: train-clean-360 not found. Did you run download_data.py?")
        return

    # Speaker IDs are the folder names inside train-clean-360
    all_speaker_ids = [d.name for d in data_path.iterdir() if d.is_dir()]
    random.shuffle(all_speaker_ids)
    
    # 90% Train / 10% Val (Validation is crucial so we don't overfit)
    split_idx = int(len(all_speaker_ids) * 0.9)
    train_spks = all_speaker_ids[:split_idx]
    val_spks = all_speaker_ids[split_idx:]
    
    print(f"Total Speakers: {len(all_speaker_ids)}")
    print(f"Train Pool: {len(train_spks)} | Val Pool: {len(val_spks)}")

    # 2. Create Datasets with explicit speaker lists
    train_ds = VoicePipeline("./data", samples_per_epoch=SAMPLES_PER_EPOCH, 
                           augment=True, speaker_ids=train_spks)
    
    val_ds = VoicePipeline("./data", samples_per_epoch=2000, 
                         augment=False, speaker_ids=val_spks)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 3. Model Setup
    model = VerificationNet(freeze_encoder=True).to(DEVICE)
    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Dynamic Learning Rate: slows down as we get closer to solution
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')

    # 4. Training Loop
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_train_loss = 0
        
        for x1, x2, label in train_loader:
            x1, x2, label = x1.to(DEVICE), x2.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for x1, x2, label in val_loader:
                x1, x2, label = x1.to(DEVICE), x2.to(DEVICE), label.to(DEVICE)
                logits = model(x1, x2)
                loss = criterion(logits, label)
                total_val_loss += loss.item()
                
                preds = torch.sigmoid(logits) > 0.5
                correct_val += (preds == label.byte()).sum().item()
                total_val += label.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        dt = time.time() - t0
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | {dt:.1f}s")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "voice_model_large.pth")

    print("Training Complete.")
    # (Run calibration here if desired)

if __name__ == "__main__":
    import random # needed for the main block
    train()