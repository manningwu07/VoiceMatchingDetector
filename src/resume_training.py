import tensorflow as tf
import numpy as np
import os
import random
from glob import glob
from model import L2Normalize # Import custom layer class

# --- CONFIG ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     
BATCH_SIZE = 64  
PAIRS_PER_EPOCH = 10000 
EPOCHS_TO_ADD = 30 # Squeeze out more performance

# Load the HEALTHY model, not the crashed final one
LOAD_PATH = "ecapa_best.h5" 
SAVE_PATH = "ecapa_finetuned.h5"

# Lower LR for fine-tuning
FINE_TUNE_LR = 0.00005 

# --- RE-DEFINE GENERATORS (Copy/Paste for safety) ---
import librosa
def augment_audio_signal(y, sr):
    if random.random() > 0.7:
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    if random.random() > 0.8:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
    return y

def preprocess_pipeline(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        if augment: y = augment_audio_signal(y, sr)
        target_len = int(sr * DURATION)
        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start+target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]
        return mfcc.astype(np.float32)
    except:
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

def triplet_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch, n_batch = [], [], []
        for _ in range(batch_size):
            spk_pos = random.choice(speakers)
            if len(speaker_dict[spk_pos]) < 2:
                file_a = file_p = speaker_dict[spk_pos][0]
            else:
                file_a, file_p = random.sample(speaker_dict[spk_pos], 2)
            spk_neg = random.choice(speakers)
            while spk_neg == spk_pos: spk_neg = random.choice(speakers)
            file_n = random.choice(speaker_dict[spk_neg])

            a = preprocess_pipeline(file_a, augment=augment)
            p = preprocess_pipeline(file_p, augment=augment)
            n = preprocess_pipeline(file_n, augment=augment)
            a_batch.append(a); p_batch.append(p); n_batch.append(n)
        yield [np.array(a_batch), np.array(p_batch), np.array(n_batch)], np.zeros((batch_size,))

# Need dummy identity_loss for loading
def identity_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

# --- MAIN ---
if __name__ == "__main__":
    print(f"♻️  Resuming from healthy checkpoint: {LOAD_PATH}")
    
    # 1. Load the good model
    try:
        model = tf.keras.models.load_model(
            LOAD_PATH,
            custom_objects={"identity_loss": identity_loss, "L2Normalize": L2Normalize}
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        exit()

    # 2. DATA SETUP (Same as before)
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
    keys = list(speaker_dict.keys())
    val_keys = set(keys[::10])
    train_dict = {k: v for k, v in speaker_dict.items() if k not in val_keys}
    val_dict = {k: v for k, v in speaker_dict.items() if k in val_keys}
    
    # One-time validation set gen
    print("🔨 Generating Val Set...")
    val_gen = triplet_generator(val_dict, batch_size=1500, augment=False)
    val_data = next(val_gen)

    # 3. CRITICAL FIX: Recompile with Gradient Clipping
    print("🛡️  Recompiling with Gradient Clipping (global_clipnorm=3.0)...")
    
    # Use legacy optimizer for Mac M-series, but add CLIPNORM
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=FINE_TUNE_LR,
        clipnorm=3.0  # <--- THIS PREVENTS THE CRASH
    )
    
    model.compile(loss=identity_loss, optimizer=optimizer)

    callbacks = [
        # Save to a NEW file so we don't overwrite the known-good one until it improves
        tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ]

    print("🚀 Starting Fine-Tuning...")
    model.fit(
        triplet_generator(train_dict, BATCH_SIZE, augment=True),
        validation_data=val_data,
        steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
        epochs=EPOCHS_TO_ADD,
        callbacks=callbacks
    )
    print("✅ Done. Check ecapa_finetuned.h5")