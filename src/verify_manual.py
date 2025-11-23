import tensorflow as tf
import numpy as np
from model import contrastive_loss, distance_accuracy, L2Normalize
import librosa
import sys
import os

SAMPLE_RATE = 16000
N_MFCC = 80
MAX_LEN = 130

def preprocess(path):
    print(f"  Processing {os.path.basename(path)}...")
    
    # 1. Load Audio
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    
    # 2. VAD (Voice Activity Detection) - CRITICAL FIX
    # Trim silence from beginning and end (top_db=20 is standard for voice)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # If trimmed audio is too short, warn user
    if len(y_trimmed) < 4000:
        print(f"  ⚠️ WARNING: Audio file {path} is mostly silence/noise!")
    
    # Take center 3 seconds if too long, or pad if too short
    target_len = int(SAMPLE_RATE * 3.0)
    if len(y_trimmed) > target_len:
        start = (len(y_trimmed) - target_len) // 2
        y_trimmed = y_trimmed[start:start+target_len]
    else:
        pad_len = target_len - len(y_trimmed)
        y_trimmed = np.pad(y_trimmed, (0, pad_len))

    # 3. Extract MFCC
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    
    # 4. CMVN (Normalize)
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0) + 1e-8
    mfcc = (mfcc - mean) / std
    
    # 5. Length padding (just in case)
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
        
    return mfcc[np.newaxis, ...]

def verify(path1, path2, threshold=0.5):
    print(f"Loading best model...")
    model = tf.keras.models.load_model(
        "siamese_best.h5", 
        custom_objects={
            "contrastive_loss": contrastive_loss,
            "distance_accuracy": distance_accuracy,
            "L2Normalize": L2Normalize
        }
    )
    
    # Get the embedding model (the shared encoder) specifically
    # Layer [2] is usually the base network in a Siamese model
    encoder = model.layers[2] 
    
    f1 = preprocess(path1)
    f2 = preprocess(path2)
    
    print("Computing embeddings...")
    # Get the actual vectors
    vec1 = encoder.predict(f1, verbose=0)[0]
    vec2 = encoder.predict(f2, verbose=0)[0]
    
    # DEBUG: Print Vector Stats
    print("\n--- DEBUG STATS ---")
    print(f"Vec1 Mean: {np.mean(vec1):.4f}, Std: {np.std(vec1):.4f}")
    print(f"Vec2 Mean: {np.mean(vec2):.4f}, Std: {np.std(vec2):.4f}")
    
    # Manually compute distance to be sure
    dist_sq = np.sum(np.square(vec1 - vec2))
    dist = np.sqrt(dist_sq)
    
    print(f"\n🔍 Calculated Distance: {dist:.4f}")
    print(f"📊 Threshold: {threshold}")
    
    if dist < threshold:
        print("✅ SAME PERSON (Low Distance)")
    else:
        print("❌ DIFFERENT PEOPLE (High Distance)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 src/verify_manual.py file1.wav file2.wav [threshold]")
        sys.exit(1)
        
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    verify(sys.argv[1], sys.argv[2], threshold)