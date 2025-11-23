import tensorflow as tf
import tensorflow.keras.backend as K
import librosa
import numpy as np
import os
import random
from glob import glob
from model import build_siamese_model, contrastive_loss, L2Normalize

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     

# Reduced batch size for stability, but accumulating gradients would be better if supported easily
BATCH_SIZE = 64  
EPOCHS = 100      
PAIRS_PER_EPOCH = 10000 

CHECKPOINT_PATH = "ecapa_checkpoint.h5"
BEST_MODEL_PATH = "ecapa_best.h5"

# significantly reduced learning rate
INITIAL_LR = 0.0001

# --- METRICS ---
def distance_accuracy(y_true, y_pred):
    return K.mean(tf.equal(tf.cast(y_pred < 0.5, tf.float32), y_true))

# --- REDUCED AUGMENTATION (Initially simpler) ---
def augment_audio_signal(y, sr):
    """Augment RAW audio before MFCC (More realistic)"""
    # 1. Gaussian Noise (Simulate Microphone Hiss) - Reduced probability and amplitude
    if random.random() > 0.7:
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    
    # 2. Time Stretch (Speed up / Slow down) - Reduced range
    if random.random() > 0.8:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
        
    # 3. Pitch Shift (Alter tone slightly) - Reduced steps
    if random.random() > 0.8:
        steps = random.uniform(-1, 1)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        
    return y

def get_mfcc(y, sr):
    # Ensure length
    target_len = int(sr * DURATION)
    if len(y) > target_len:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start+target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    
    # CMVN (Critical for cross-mic performance)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
    return mfcc

def preprocess_pipeline(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        if augment:
            y = augment_audio_signal(y, sr)
        mfcc = get_mfcc(y, sr)
        return mfcc.astype(np.float32)
    except:
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

# --- DATA GENERATOR ---
def pair_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())
    while True:
        x1, x2, y = [], [], []
        for _ in range(batch_size):
            is_same = random.random() > 0.5
            if is_same:
                spk = random.choice(speakers)
                f1, f2 = random.sample(speaker_dict[spk], 2) if len(speaker_dict[spk]) > 1 else (speaker_dict[spk][0], speaker_dict[spk][0])
                lbl = 1.0
            else:
                s1, s2 = random.sample(speakers, 2)
                f1 = random.choice(speaker_dict[s1])
                f2 = random.choice(speaker_dict[s2])
                lbl = 0.0
            
            # Augment BOTH independently to force channel invariance
            feat1 = preprocess_pipeline(f1, augment=augment)
            feat2 = preprocess_pipeline(f2, augment=augment)
            
            # Check shapes
            if feat1.shape[0] != MAX_LEN: feat1 = np.resize(feat1, (MAX_LEN, N_MFCC))
            if feat2.shape[0] != MAX_LEN: feat2 = np.resize(feat2, (MAX_LEN, N_MFCC))
                
            x1.append(feat1); x2.append(feat2); y.append(lbl)
        
        yield [np.array(x1), np.array(x2)], np.array(y)

def create_val_set(speaker_dict, n=1000):
    print(f"🔨 Generating Fixed Validation Set ({n} pairs)...")
    # Reuse the generator logic once
    gen = pair_generator(speaker_dict, batch_size=n, augment=False)
    return next(gen)

# --- MAIN ---
if __name__ == "__main__":
    print("Scanning dataset...")
    data_path = "./data/LibriSpeech/train-clean-360"
    speaker_dict = {}
    # Fast scan using scandir
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2] # Adapt based on folder structure
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))

    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
    keys = list(speaker_dict.keys())
    
    # 90/10 Split
    val_keys = set(keys[::10])
    train_dict = {k: v for k, v in speaker_dict.items() if k not in val_keys}
    val_dict = {k: v for k, v in speaker_dict.items() if k in val_keys}
    
    print(f"Train: {len(train_dict)} spk, Val: {len(val_dict)} spk")
    val_data = create_val_set(val_dict, n=1500)

    input_shape = (MAX_LEN, N_MFCC)
    model = build_siamese_model(input_shape)
    
    # Adjusted contrastive loss margin
    def relaxed_contrastive_loss(y_true, y_pred):
        return contrastive_loss(y_true, y_pred, margin=0.5)

    model.compile(
        loss=relaxed_contrastive_loss,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=INITIAL_LR),
        metrics=[distance_accuracy]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    ]

    print("🚀 Starting ECAPA-TDNN Training...")
    model.fit(
        pair_generator(train_dict, BATCH_SIZE, augment=True),
        validation_data=val_data,
        steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    model.save("ecapa_final.h5")