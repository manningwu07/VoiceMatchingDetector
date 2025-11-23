import tensorflow as tf
import numpy as np
import os
import random
import librosa
from model import build_siamese_model, L2Normalize

# --- CONFIG ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     
BATCH_SIZE = 64
PAIRS_PER_EPOCH = 5000 # Enough to see all variance

MODEL_TO_HEAL = "ecapa_best.h5"
HEALED_MODEL = "ecapa_healed.h5"

# --- DATA GEN ---
def preprocess_pipeline(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        # No Augmentation for healing - we want clean stats
        target_len = int(sr * DURATION)
        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start+target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]
        return mfcc.astype(np.float32)
    except:
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

def triplet_generator(speaker_dict, batch_size=32):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch, n_batch = [], [], []
        for _ in range(batch_size):
            spk_pos = random.choice(speakers)
            if len(speaker_dict[spk_pos]) < 2: continue
            
            f1, f2 = random.sample(speaker_dict[spk_pos], 2)
            spk_neg = random.choice(speakers)
            while spk_neg == spk_pos: spk_neg = random.choice(speakers)
            f3 = random.choice(speaker_dict[spk_neg])

            a_batch.append(preprocess_pipeline(f1))
            p_batch.append(preprocess_pipeline(f2))
            n_batch.append(preprocess_pipeline(f3))
        
        yield [np.array(a_batch), np.array(p_batch), np.array(n_batch)], np.zeros((batch_size,))

def identity_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

if __name__ == "__main__":
    print(f"🚑 Healing Model: {MODEL_TO_HEAL}")
    
    # 1. Rebuild and Load Weights
    model = build_siamese_model((MAX_LEN, N_MFCC))
    try:
        model.load_weights(MODEL_TO_HEAL)
        print("✅ Weights loaded.")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        exit()

    # 2. FREEZE WEIGHTS (Critical Step)
    # We only want to update the Moving Averages (Non-Trainable params), not the Kernels
    for layer in model.layers:
        layer.trainable = False
        
    # 3. Compile with 0 Learning Rate (Double Safety)
    model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0))

    # 4. Data Setup
    print("📂 Scanning data...")
    data_path = "./data/LibriSpeech/train-clean-360"
    speaker_dict = {}
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2]
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))
    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}

    # 5. Run "Training" (Healing)
    # We run fit(), which triggers the BN updates, but gradients are 0 so weights don't change.
    print("💉 Injecting data to reset Batch Norm stats...")
    model.fit(
        triplet_generator(speaker_dict, BATCH_SIZE),
        steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
        epochs=3, # Run a few times to stabilize stats
        verbose=1
    )
    
    # 6. Save Healed Model
    model.save(HEALED_MODEL)
    print(f"✅ Healed model saved to: {HEALED_MODEL}")
    print("👉 Now update resume_training.py to load 'ecapa_healed.h5' and RUN.")
