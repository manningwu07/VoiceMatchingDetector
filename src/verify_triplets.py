import tensorflow as tf
import numpy as np
import librosa
import sys
import os

# Define dummy objects so Keras can load the architecture
# We don't actually use them for inference, but load_model needs them
def identity_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, x):
        return tf.math.l2_normalize(x, axis=self.axis)
    def get_config(self):
        return super().get_config() | {"axis": self.axis}

# --- CONFIG ---
SAMPLE_RATE = 16000
N_MFCC = 80
MAX_LEN = 130
THRESHOLD = 0.6  # Triplet models often need slightly higher thresholds

def preprocess(path):
    print(f"  Processing {os.path.basename(path)}...")
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    
    # Trim Silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Pad/Cut to 3s
    target = int(SAMPLE_RATE * 3.0)
    if len(y) > target:
        start = (len(y) - target) // 2
        y = y[start:start+target]
    else:
        y = np.pad(y, (0, target - len(y)))
        
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    # CMVN (Normalize)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
    
    # Pad Shape
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
        
    return mfcc[np.newaxis, ...]

def verify(path1, path2):
    print(f"Loading Triplet Model...")
    
    # 1. Load the Full Triplet Model
    triplet_model = tf.keras.models.load_model(
        "ecapa_best.h5",
        custom_objects={"identity_loss": identity_loss, "L2Normalize": L2Normalize}
    )
    
    # 2. Extract ONLY the Encoder (The Brain)
    # We look for the layer named 'ECAPA_Encoder' which we named in model.py
    encoder = triplet_model.get_layer("ECAPA_Encoder")
    print("✅ Encoder extracted successfully.")

    f1 = preprocess(path1)
    f2 = preprocess(path2)
    
    v1 = encoder.predict(f1, verbose=0)[0]
    v2 = encoder.predict(f2, verbose=0)[0]
    
    # Cosine Distance (1.0 - Similarity)
    # Since vectors are L2 normalized, Euclidean^2 = 2 * (1 - Cosine)
    # But let's just use Euclidean distance as trained
    dist = np.linalg.norm(v1 - v2)
    
    print(f"\n--- RESULTS ---")
    print(f"Files: {os.path.basename(path1)} vs {os.path.basename(path2)}")
    print(f"Distance: {dist:.4f}")
    
    if dist < THRESHOLD:
        print(f"✅ SAME PERSON (Conf: {((THRESHOLD-dist)/THRESHOLD)*100:.1f}%)")
    else:
        print(f"❌ DIFFERENT PERSON")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 src/verify_triplet.py file1.wav file2.wav")
        sys.exit(1)
    verify(sys.argv[1], sys.argv[2])
