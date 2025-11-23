# Protects against corrupt data during training by adding rigorous checks
# and skipping any triplet with problematic files.

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
EPOCHS = 30 
PAIRS_PER_EPOCH = 10000

# Resume from the GOOD epoch (Epoch 3), not the crashed one
LOAD_PATH = "ecapa_finetuned.h5"
SAVE_PATH = "ecapa_final_armored.h5"
FINE_TUNE_LR = 0.00001 # Even lower for final polish

# --- ARMORED PREPROCESSING ---
def augment_audio_signal(y, sr):
    if random.random() > 0.6:
        noise_amp = 0.0005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    if random.random() > 0.7:
        rate = random.uniform(0.98, 1.02)
        y = librosa.effects.time_stretch(y, rate=rate)
    return y

def preprocess_pipeline(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        # CRITICAL CHECK: If audio is too silent/empty, reject it
        if np.max(np.abs(y)) < 0.005: 
            return None # Reject silence
            
        if augment: y = augment_audio_signal(y, sr)
        
        target_len = int(sr * DURATION)
        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start+target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
            
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        
        # Check for NaNs in MFCC
        if np.isnan(mfcc).any(): return None
        
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]
            
        return mfcc.astype(np.float32)
    except:
        return None # Reject corrupt files

# --- ARMORED GENERATOR ---
def triplet_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch, n_batch = [], [], []
        while len(a_batch) < batch_size:
            # 1. Select Speakers
            spk_pos = random.choice(speakers)
            if len(speaker_dict[spk_pos]) < 2: continue
            
            file_a, file_p = random.sample(speaker_dict[spk_pos], 2)
            
            spk_neg = random.choice(speakers)
            while spk_neg == spk_pos: spk_neg = random.choice(speakers)
            file_n = random.choice(speaker_dict[spk_neg])

            # 2. Preprocess with Checks
            # If ANY file in the triplet returns None (corrupt), we skip the whole triplet
            a = preprocess_pipeline(file_a, augment=augment)
            if a is None: continue
            
            p = preprocess_pipeline(file_p, augment=augment)
            if p is None: continue
            
            n = preprocess_pipeline(file_n, augment=augment)
            if n is None: continue
            
            a_batch.append(a); p_batch.append(p); n_batch.append(n)
        
        yield [np.array(a_batch), np.array(p_batch), np.array(n_batch)], np.zeros((batch_size,))

def create_val_set(speaker_dict, n=1000):
    print(f"🔨 Generating Val Set...")
    gen = triplet_generator(speaker_dict, batch_size=n, augment=False)
    return next(gen)

def identity_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

if __name__ == "__main__":
    print(f"🛡️  Starting Armored Training from: {LOAD_PATH}")
    
    # 1. Load Model
    model = build_siamese_model((MAX_LEN, N_MFCC))
    try:
        model.load_weights(LOAD_PATH)
        print("✅ Loaded healthy weights.")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    # 2. Data Setup
    data_path = "./data/LibriSpeech/train-clean-360"
    speaker_dict = {}
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2]
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))
    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
    
    keys = list(speaker_dict.keys())
    keys.sort() # Deterministic Split
    val_keys = set(keys[::10])
    train_dict = {k: v for k, v in speaker_dict.items() if k not in val_keys}
    val_dict = {k: v for k, v in speaker_dict.items() if k in val_keys}
    val_data = create_val_set(val_dict, n=1500)

    # 3. Compile with ClipValue (Stricter than ClipNorm)
    print("🛡️  Compiling with clipvalue=0.5...")
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=FINE_TUNE_LR,
        clipvalue=0.5  # Clips INDIVIDUAL gradients, safer for NaNs
    )
    
    model.compile(loss=identity_loss, optimizer=optimizer)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    model.fit(
        triplet_generator(train_dict, BATCH_SIZE, augment=True),
        validation_data=val_data,
        steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )