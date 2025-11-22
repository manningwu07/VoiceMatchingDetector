import tensorflow as tf
import tensorflow.keras.backend as K
import librosa
import numpy as np
import os
import random
from glob import glob
from model import L2Normalize, build_siamese_model, contrastive_loss

# --- CONFIGURATION ---
# (Updated to match your screenshot settings)
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 80       
MAX_LEN = 130

BATCH_SIZE = 384  
EPOCHS = 200      
PAIRS_PER_EPOCH = 2000 # Kept at 20k to prevent overfitting (resampling)

# UPDATED: Use .h5 to avoid 'options' arguments error in ModelCheckpoint
CHECKPOINT_PATH = "siamese_checkpoint.h5"
BEST_MODEL_PATH = "siamese_best.h5"

INITIAL_LR = 0.005  # Updated to match the new learning rate

print(f"🚀 TensorFlow Version: {tf.__version__} (Mac Optimized)")

# --- DISTANCE ACCURACY ---
def distance_accuracy(y_true, y_pred):
    # y_pred is distance. < 0.5 means Same (1.0).
    prediction = tf.cast(y_pred < 0.5, tf.float32)
    return K.mean(tf.equal(prediction, y_true))

# --- AUGMENTATION (Fixed) ---
def augment_mfcc(mfcc):
    """
    SpecAugment style: Noise + Freq Masking + Time Masking.
    Robust and crash-free.
    """
    # 1. Additive Noise
    if random.random() > 0.5:
        noise = np.random.randn(*mfcc.shape) * random.uniform(0.005, 0.02)
        mfcc = mfcc + noise

    # 2. Frequency Masking (Block out bands of frequencies)
    if random.random() > 0.5:
        n_mels = mfcc.shape[1]
        # Mask up to 15% of frequencies
        F = random.randint(1, int(n_mels * 0.15)) 
        f0 = random.randint(0, n_mels - F)
        mfcc[:, f0:f0+F] = 0.0

    # 3. Time Masking (Block out time segments) -> Replaces buggy Time Stretch
    if random.random() > 0.5:
        time_steps = mfcc.shape[0]
        # Mask up to 15% of time steps
        T = random.randint(1, int(time_steps * 0.15))
        t0 = random.randint(0, time_steps - T)
        mfcc[t0:t0+T, :] = 0.0

    return mfcc.astype(np.float32)

# --- FEATURE EXTRACTION ---
def preprocess_audio(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = mfcc.T

        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]

        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

# --- DATA GENERATOR ---
def pair_generator(speaker_dict, batch_size=32, augment=False):
    speakers = list(speaker_dict.keys())

    while True:
        x1_batch = []
        x2_batch = []
        y_batch = []

        for _ in range(batch_size):
            is_same = random.random() > 0.5

            if is_same:
                spk = random.choice(speakers)
                if len(speaker_dict[spk]) < 2:
                    f1 = f2 = speaker_dict[spk][0]
                else:
                    f1, f2 = random.sample(speaker_dict[spk], 2)
                label = 1.0
            else:
                s1, s2 = random.sample(speakers, 2)
                f1 = random.choice(speaker_dict[s1])
                f2 = random.choice(speaker_dict[s2])
                label = 0.0

            feat1 = preprocess_audio(f1)
            feat2 = preprocess_audio(f2)

            if augment:
                feat1 = augment_mfcc(feat1)
                feat2 = augment_mfcc(feat2)

            x1_batch.append(feat1)
            x2_batch.append(feat2)
            y_batch.append(label)

        yield [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Scanning dataset...")
    data_path = "./data/LibriSpeech/train-clean-360"

    speaker_dict = {}
    files = glob(data_path + "/**/*.flac", recursive=True)
    for f in files:
        spk_id = f.split(os.sep)[-3]
        if spk_id not in speaker_dict:
            speaker_dict[spk_id] = []
        speaker_dict[spk_id].append(f)

    speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
    print(f"Found {len(speaker_dict)} valid speakers.")

    # Split: Train vs Val (Every 10th speaker is validation)
    all_speakers = list(speaker_dict.keys())
    val_speakers_ids = set(all_speakers[::10]) 
    train_speakers = {k: v for k, v in speaker_dict.items() if k not in val_speakers_ids}
    val_speakers_dict = {k: v for k, v in speaker_dict.items() if k in val_speakers_ids}
    
    print(f"Train speakers: {len(train_speakers)}, Val speakers: {len(val_speakers_dict)}")

    # Model Setup
    input_shape = (MAX_LEN, N_MFCC)

    # Need custom objects because of the specific loss/metric functions
    custom_objects = {
        "contrastive_loss": contrastive_loss,
        "distance_accuracy": distance_accuracy,
        "L2Normalize": L2Normalize
    }

    # Load or Initialize Model (Best, Checkpoint, then Fresh)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"🔄 Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
        try:
            siamese_model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
        except Exception as e:
            print(f"❌ CRITICAL: Could not load checkpoint ({e}).")
            print("🛑 Stopping to prevent overwrite. Fix the model or delete checkpoint manually.")
            exit(1)
    elif os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
        try:
            siamese_model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
        except Exception as e:
            print(f"❌ CRITICAL: Could not load checkpoint ({e}).")
            print("🛑 Stopping to prevent overwrite. Fix the model or delete checkpoint manually.")
            exit(1)
    else:
        print("✨ No checkpoint found. Starting fresh training...")
        siamese_model = build_siamese_model(input_shape)

    # Reactive Learning Rate (Aggressive)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.6,       # Cut LR in half
        patience=2,       # If no improvement for 2 epochs...
        min_delta=0.005,  # ...where improvement must be at least 0.005
        min_lr=1e-6,
        verbose=1
    )

    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=INITIAL_LR),
        metrics=[distance_accuracy]
    )

    siamese_model.summary()

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False, # Keep latest for resuming
        save_weights_only=False,
        verbose=1
    )

    best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True, # Keep absolute best
        save_weights_only=False,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting Training on M4 Pro...")
    try:
        history = siamese_model.fit(
            pair_generator(train_speakers, BATCH_SIZE, augment=True),
            validation_data=pair_generator(val_speakers_dict, BATCH_SIZE, augment=False),
            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
            validation_steps=2000 // BATCH_SIZE, # Smaller val steps for speed
            epochs=EPOCHS,
            callbacks=[checkpoint_cb, best_model_cb, early_stopping, lr_reduce]
        )

        siamese_model.save("custom_voice_auth_final.h5")
        print("\n✅ Training Complete & Saved.")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted. Saving progress...")
        siamese_model.save(CHECKPOINT_PATH)
        print("✅ Progress saved.")