import tensorflow as tf
import librosa
import numpy as np
import sys

# Constants (Must match training)
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 130

def preprocess(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    return mfcc[np.newaxis, ...] # Add batch dim

def verify(path1, path2):
    print(f"Loading model...")
    model = tf.keras.models.load_model("custom_voice_auth.keras")
    
    print(f"Processing audio...")
    f1 = preprocess(path1)
    f2 = preprocess(path2)
    
    print(f"Predicting...")
    # Prediction is probability of being the SAME person (0-1)
    score = model.predict([f1, f2], verbose=0)[0][0]
    
    # In our specific architecture (Euclidean Distance -> Dense -> Sigmoid), 
    # If the Dense weights are negative (common), high distance = low score (0).
    # Low distance = high score (1).
    
    print(f"\n🔍 Similarity Score: {score:.4f}")
    if score > 0.5:
        print("✅ SAME PERSON")
    else:
        print("❌ DIFFERENT PEOPLE")

if __name__ == "__main__":
    # Usage: python3 verify_manual.py file1.wav file2.wav
    if len(sys.argv) < 3:
        print("Usage: python3 verify_manual.py <file1> <file2>")
    else:
        verify(sys.argv[1], sys.argv[2])