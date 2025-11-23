import tensorflow as tf
import numpy as np
import os
import random
import librosa
import sys
from model import build_siamese_model, L2Normalize

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     
BATCH_SIZE = 64
EPOCHS = 100      
PAIRS_PER_EPOCH = 10000

# File Paths
DATA_DIR = "./data/LibriSpeech/train-clean-360"
CHECKPOINT_PATH = "ecapa_master.h5" # Saves best model here
LOG_DIR = "logs"

# Learning Params
INITIAL_LR = 0.0001 # Standard start
CLIP_NORM = 3.0     # Safety rail (Prevents explosion, allows learning)

# --- ARMORED PREPROCESSING ---
def augment_audio_signal(y, sr):
    """Apply augmentation only if audio is valid"""
    # 1. Noise
    if random.random() > 0.6:
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    # 2. Time Stretch
    if random.random() > 0.7:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
    return y

def preprocess_pipeline(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        # 1. Silence Check (Reject if mostly empty)
        if np.max(np.abs(y)) < 0.005: return None
        
        if augment: y = augment_audio_signal(y, sr)
        
        # 2. Length Check/Pad
        target_len = int(sr * DURATION)
        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start+target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
            
        # 3. MFCC & NaN Check
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        if np.isnan(mfcc).any(): return None # Reject NaNs
        
        # 4. Normalize
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        # 5. Shape Check
        if mfcc.shape[0] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:MAX_LEN, :]
            
        return mfcc.astype(np.float32)
    except:
        return None # Catch generic file errors

# --- ARMORED GENERATOR ---
def triplet_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch, n_batch = [], [], []
        # Keep looping until we fill the batch with VALID data
        while len(a_batch) < batch_size:
            spk_pos = random.choice(speakers)
            if len(speaker_dict[spk_pos]) < 2: continue
            
            file_a, file_p = random.sample(speaker_dict[spk_pos], 2)
            
            spk_neg = random.choice(speakers)
            while spk_neg == spk_pos: spk_neg = random.choice(speakers)
            file_n = random.choice(speaker_dict[spk_neg])

            # Process & Check Validity
            a = preprocess_pipeline(file_a, augment=augment)
            if a is None: continue
            
            p = preprocess_pipeline(file_p, augment=augment)
            if p is None: continue
            
            n = preprocess_pipeline(file_n, augment=augment)
            if n is None: continue
            
            a_batch.append(a); p_batch.append(p); n_batch.append(n)
        
        yield [np.array(a_batch), np.array(p_batch), np.array(n_batch)], np.zeros((batch_size,))

def create_val_set(speaker_dict, n=1000):
    print(f"🔨 Generating Fixed Validation Set ({n} triplets)...")
    gen = triplet_generator(speaker_dict, batch_size=n, augment=False)
    return next(gen)

def identity_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

# --- MAIN ---
if __name__ == "__main__":
    print(f"🚀 TensorFlow Version: {tf.__version__}")
    
    # 1. Data Indexing
    print("📂 Indexing Dataset...")
    speaker_dict = {}
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2]
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))
    
    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
    
    # 2. Deterministic Split (Sort keys to prevent leakage)
    keys = list(speaker_dict.keys())
    keys.sort() 
    
    val_keys = set(keys[::10]) # 10% val
    train_dict = {k: v for k, v in speaker_dict.items() if k not in val_keys}
    val_dict = {k: v for k, v in speaker_dict.items() if k in val_keys}
    
    print(f"📊 Train Speakers: {len(train_dict)} | Val Speakers: {len(val_dict)}")
    val_data = create_val_set(val_dict, n=1500)

    # 3. Model Setup (Auto-Resume)
    model = build_siamese_model((MAX_LEN, N_MFCC))
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"♻️  Found Checkpoint: {CHECKPOINT_PATH}. Resuming...")
        try:
            # Load weights ONLY (safer than loading full model)
            model.load_weights(CHECKPOINT_PATH)
            print("✅ Weights loaded.")
        except Exception as e:
            print(f"⚠️  Checkpoint load failed: {e}. Starting fresh.")
    else:
        print("✨ No checkpoint found. Starting fresh.")

    # 4. Compile with Safety Rails
    print(f"🛡️  Compiling with ClipNorm={CLIP_NORM}")
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=INITIAL_LR,
        clipnorm=CLIP_NORM
    )
    model.compile(loss=identity_loss, optimizer=optimizer)

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

    # 6. Train
    print("🔥 Starting Training...")
    try:
        model.fit(
            triplet_generator(train_dict, BATCH_SIZE, augment=True),
            validation_data=val_data,
            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User.")
        
    print("✅ Done.")