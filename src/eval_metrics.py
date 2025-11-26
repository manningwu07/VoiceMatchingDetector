import tensorflow as tf
import numpy as np
import librosa
import os
import random
from tqdm import tqdm
from model import build_siamese_model

# --- CONFIG ---
SAMPLE_RATE = 16000
N_MFCC = 80
MAX_LEN = 130
BATCH_SIZE = 64
DATA_DIR = "./data/LibriSpeech/train-clean-360"

# --- ARCFACE SUPPORT ---
# We need this class definition to reconstruct the model for loading

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        embedding_shape = input_shape[0]
        self.W = self.add_weight(name='W',
                                shape=(embedding_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        super(ArcFace, self).build(input_shape)

    def call(self, inputs):
        x, y = inputs
        # FIX: We must return a Tensor, not a list, so Softmax works.
        # We perform the matmul to ensure the shape is (batch_size, n_classes)
        # and to ensure 'W' is included in the computation graph for weight loading.
        return tf.matmul(x, self.W)

def get_num_classes(data_dir):
    # We must match the class count exactly to load weights
    speakers = [d for d in os.listdir(data_dir) if not d.startswith('.')]
    return len(speakers)

def build_reconstruction(n_classes):
    # Rebuild the EXACT architecture from train_master.py
    base_model = build_siamese_model((MAX_LEN, N_MFCC))
    encoder = base_model.get_layer("ECAPA_Encoder")
    
    audio_inp = tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC))
    label_inp = tf.keras.layers.Input(shape=(n_classes,))
    
    emb = encoder(audio_inp)
    
    # We reconstruct the head just to satisfy the weight loader
    output = ArcFace(n_classes=n_classes)([emb, label_inp])
    output = tf.keras.layers.Softmax()(output)
    
    model = tf.keras.Model([audio_inp, label_inp], output)
    return model

# --- EVALUATION LOGIC ---
def preprocess_tta(path):
    """Returns a batch of variations for a single file"""
    try:
        y_raw, _ = librosa.load(path, sr=SAMPLE_RATE)
        # 1. Base crop (Center)
        target = int(SAMPLE_RATE * 3.0)
        if len(y_raw) > target:
            start = (len(y_raw) - target) // 2
            y_base = y_raw[start:start+target]
        else:
            y_base = np.pad(y_raw, (0, target - len(y_raw)))

        # 2. Shifted crop (Random offset)
        if len(y_raw) > target:
            start = random.randint(0, len(y_raw) - target)
            y_shift = y_raw[start:start+target]
        else:
            y_shift = y_base # Fallback

        batch = []
        for y in [y_base, y_shift]:
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
            # Instance Norm
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
            
            if mfcc.shape[0] < MAX_LEN:
                mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
            else:
                mfcc = mfcc[:MAX_LEN, :]
            batch.append(mfcc)
            
        return np.array(batch, dtype=np.float32)
    except:
        return np.zeros((2, MAX_LEN, N_MFCC), dtype=np.float32)

def evaluate(model_path, data_dir, num_pairs=2000):
    print(f"🏗️  Reconstructing ArcFace Model...")
    
    # 1. Determine shape
    n_classes = get_num_classes(data_dir)
    print(f"   Detected {n_classes} classes/speakers.")
    
    # 2. Build and Load
    full_model = build_reconstruction(n_classes)
    try:
        full_model.load_weights(model_path)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 3. Extract Encoder
    encoder = full_model.get_layer("ECAPA_Encoder")
    
    # 4. Data Indexing
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
    # Use the validation split (every 10th speaker)
    val_keys = speakers[::10]
    
    print(f"🧪 Testing on {len(val_keys)} speakers ({num_pairs} pairs) using TTA...")
    
    dists = []
    labels = []
    
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
            
        # Get TTA batches (shape: [2, 130, 80])
        inp1 = preprocess_tta(f1)
        inp2 = preprocess_tta(f2)
        
        # Predict on batch
        v1_batch = encoder.predict(inp1, verbose=0)
        v2_batch = encoder.predict(inp2, verbose=0)
        
        # Average the embeddings (The Magic Step)
        v1 = np.mean(v1_batch, axis=0)
        v2 = np.mean(v2_batch, axis=0)
        
        # Re-normalize after averaging
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Cosine Distance
        dist = 1.0 - np.dot(v1, v2)
        
        dists.append(dist)
        labels.append(label)

    # --- Find Best Threshold ---
    dists = np.array(dists)
    labels = np.array(labels)
    best_acc = 0
    best_thresh = 0
    
    # Scan thresholds from 0.0 to 1.0
    for t in np.arange(0.0, 1.0, 0.01):
        # In cosine distance: 0 is same, 1 is different.
        # So if dist < t, we predict SAME (1).
        preds = (dists < t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    print("\n" + "="*30)
    print(f"🏆 FINAL TTA RESULTS")
    print(f"✅ Best Accuracy: {best_acc*100:.2f}%")
    print(f"⚖️ Optimal Threshold: {best_thresh:.2f}")
    print("="*30)

if __name__ == "__main__":
    evaluate("ecapa_master.h5", DATA_DIR)