import tensorflow as tf
import numpy as np
from model import contrastive_loss, distance_accuracy
import librosa
import sys

SAMPLE_RATE = 16000
N_MFCC = 80
MAX_LEN = 130

def preprocess(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    return mfcc[np.newaxis, ...]

def verify(path1, path2, threshold=0.5):
    print(f"Loading best model...")
    model = tf.keras.models.load_model(
        "siamese_best.keras", 
        custom_objects={
            "contrastive_loss": contrastive_loss,
            "distance_accuracy": distance_accuracy
        }
    )
    
    print(f"Processing audio...")
    f1 = preprocess(path1)
    f2 = preprocess(path2)
    
    print(f"Predicting...")
    distance = float(model.predict([f1, f2], verbose=0)[0][0])
    
    print(f"\n🔍 Distance: {distance:.4f}")
    if distance < threshold:
        print("✅ SAME PERSON")
    else:
        print("❌ DIFFERENT PEOPLE")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_manual.py <file1> <file2> [threshold]")
    else:
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        verify(sys.argv[1], sys.argv[2], threshold)