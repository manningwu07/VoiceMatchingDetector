import tensorflow as tf
import tensorflow.keras.backend as K
import librosa
import numpy as np
import os
import random
from glob import glob
from model import build_siamese_model, L2Normalize

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     

BATCH_SIZE = 64  
EPOCHS = 100      
PAIRS_PER_EPOCH = 10000 

CHECKPOINT_PATH = "ecapa_checkpoint.h5"
BEST_MODEL_PATH = "ecapa_best.h5"
INITIAL_LR = 0.0001

# --- AUGMENTATION ---
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
        
        # Ensure length logic inline
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

# --- TRIPLET GENERATOR ---
def triplet_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch, n_batch = [], [], []
        
        for _ in range(batch_size):
            # 1. Choose Anchor Speaker
            spk_pos = random.choice(speakers)
            
            # 2. Choose Positive (Same Speaker)
            if len(speaker_dict[spk_pos]) < 2:
                file_a = file_p = speaker_dict[spk_pos][0]
            else:
                file_a, file_p = random.sample(speaker_dict[spk_pos], 2)
            
            # 3. Choose Negative (Diff Speaker)
            spk_neg = random.choice(speakers)
            while spk_neg == spk_pos:
                spk_neg = random.choice(speakers)
            file_n = random.choice(speaker_dict[spk_neg])

            a = preprocess_pipeline(file_a, augment=augment)
            p = preprocess_pipeline(file_p, augment=augment)
            n = preprocess_pipeline(file_n, augment=augment)
            
            a_batch.append(a)
            p_batch.append(p)
            n_batch.append(n)
        
        # Output: [A, P, N], Dummy_Y (Model loss is calculated internally, Y is ignored)
        yield [np.array(a_batch), np.array(p_batch), np.array(n_batch)], np.zeros((batch_size,))

def create_val_set(speaker_dict, n=1000):
    print(f"🔨 Generating Fixed Validation Set ({n} triplets)...")
    gen = triplet_generator(speaker_dict, batch_size=n, augment=False)
    return next(gen)

# --- LOSS FUNCTION WRAPPER ---
# Keras requires a loss function signature (y_true, y_pred).
# Since our model outputs the *loss value itself* as prediction,
# we just return y_pred (the calculated loss).
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

# --- MAIN ---
if __name__ == "__main__":
    print("Scanning dataset...")
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
    
    val_data = create_val_set(val_dict, n=1500)

    input_shape = (MAX_LEN, N_MFCC)
    model = build_siamese_model(input_shape)
    
    model.compile(
        loss=identity_loss,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=INITIAL_LR)
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    ]

    print("🚀 Starting Triplet ECAPA-TDNN Training...")
    model.fit(
        triplet_generator(train_dict, BATCH_SIZE, augment=True),
        validation_data=val_data,
        steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    model.save("ecapa_triplet_final.h5")