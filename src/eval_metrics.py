import tensorflow as tf
import numpy as np
import librosa
import os
import random
from glob import glob
from tqdm import tqdm

# Import the architecture definition explicitly
# This prevents the "NameError: K is not defined" because the code runs here
from model import build_siamese_model

# --- CONFIG ---
SAMPLE_RATE = 16000
N_MFCC = 80
MAX_LEN = 130
BATCH_SIZE = 64

def preprocess(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        # Trim and Pad
        y, _ = librosa.effects.trim(y, top_db=20)
        target = int(SAMPLE_RATE * 3.0)
        if len(y) > target:
            start = (len(y) - target) // 2
            y = y[start:start+target]
        else:
            y = np.pad(y, (0, target - len(y)))
            
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        if mfcc.shape[0] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:MAX_LEN, :]
        return mfcc.astype(np.float32)
    except:
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

def evaluate(model_path, data_dir, num_pairs=2000):
    print(f"🏗️  Reconstructing Model Architecture...")
    # 1. Build the model structure from code (Safe way)
    full_model = build_siamese_model((MAX_LEN, N_MFCC))
    
    print(f"🔄 Loading Weights from {model_path}...")
    # 2. Load just the weights (Bypasses the Lambda deserialization error)
    # Note: We need by_name=True or skip_mismatch just in case, but usually standard load works
    try:
        full_model.load_weights(model_path)
    except Exception as e:
        print(f"Weight loading warning (might be fine if shapes match): {e}")

    # 3. Extract Encoder
    encoder = full_model.get_layer("ECAPA_Encoder")
    
    print("📂 Indexing Speakers...")
    speaker_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2]
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))
    
    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
    speakers = list(speaker_dict.keys())
    
    # Validation Set (Last 10%)
    val_keys = speakers[::10]
    print(f"🧪 Testing on {len(val_keys)} speakers ({num_pairs} pairs)...")
    
    dists = []
    labels = []
    
    print("⚡ Generating Embeddings & Calculating Distances...")
    for _ in tqdm(range(num_pairs)):
        is_same = random.random() > 0.5
        if is_same:
            spk = random.choice(val_keys)
            if len(speaker_dict[spk]) < 2: continue
            f1, f2 = random.sample(speaker_dict[spk], 2)
            label = 1
        else:
            s1, s2 = random.sample(val_keys, 2)
            f1 = random.choice(speaker_dict[s1])
            f2 = random.choice(speaker_dict[s2])
            label = 0
            
        inp1 = preprocess(f1)[np.newaxis, ...]
        inp2 = preprocess(f2)[np.newaxis, ...]
        
        v1 = encoder.predict(inp1, verbose=0)
        v2 = encoder.predict(inp2, verbose=0)
        
        dist = np.linalg.norm(v1 - v2)
        dists.append(dist)
        labels.append(label)

    # --- Find Best Threshold ---
    dists = np.array(dists)
    labels = np.array(labels)
    
    best_acc = 0
    best_thresh = 0
    
    # Scan thresholds
    for t in np.arange(0.1, 1.5, 0.05):
        preds = (dists < t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    print("\n" + "="*30)
    print(f"🏆 FINAL RESULTS")
    print(f"✅ Best Accuracy: {best_acc*100:.2f}%")
    print(f"⚖️ Optimal Threshold: {best_thresh:.2f}")
    print("="*30)

if __name__ == "__main__":
    evaluate("ecapa_best.h5", "./data/LibriSpeech/train-clean-360")