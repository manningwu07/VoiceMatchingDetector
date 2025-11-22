import tensorflow as tf
import tensorflow.keras.backend as K
import librosa
import numpy as np
import os
import random
from glob import glob
from model import build_siamese_model, contrastive_loss

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 80
MAX_LEN = 130

BATCH_SIZE = 384
EPOCHS = 300
PAIRS_PER_EPOCH = 1000 #20000
CHECKPOINT_PATH = "siamese_checkpoint.keras"
BEST_MODEL_PATH = "siamese_best.keras"

ADAM_LR = 0.0003

print(f"🚀 TensorFlow Version: {tf.__version__} (Mac Optimized)")

# --- DISTANCE ACCURACY ---
def distance_accuracy(y_true, y_pred):
    prediction = tf.cast(y_pred < 0.5, tf.float32)
    return K.mean(tf.equal(prediction, y_true))

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
def pair_generator(speaker_dict, batch_size=32, augment=True):
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

            # ADD AUGMENTATION
            if augment:
                feat1 = augment_mfcc(feat1)
                feat2 = augment_mfcc(feat2)

            x1_batch.append(feat1)
            x2_batch.append(feat2)
            y_batch.append(label)

        yield [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)

def augment_mfcc(mfcc):
    """Add noise and time-stretch effects to MFCC features"""
    # 1. Additive Gaussian noise (simulates background noise)
    if random.random() > 0.7:
        noise = np.random.randn(*mfcc.shape) * random.uniform(0.01, 0.05)
        mfcc = mfcc + noise
    
    # 2. Time stretch (speaker speaks faster/slower)
    if random.random() > 0.7:
        scale = random.uniform(0.95, 1.05)
        mfcc = mfcc[::int(1/scale), :] if scale > 1.0 else np.repeat(mfcc, int(scale), axis=0)
        # Re-pad to MAX_LEN
        if mfcc.shape[0] < MAX_LEN:
            pad = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        else:
            mfcc = mfcc[:MAX_LEN, :]
    
    # 3. Random frequency masking (block out frequency bands)
    if random.random() > 0.7:
        n_mfcc = mfcc.shape[1]
        mask_start = random.randint(0, n_mfcc - 5)
        mask_end = mask_start + random.randint(2, 8)
        mfcc[:, mask_start:mask_end] *= random.uniform(0.1, 0.5)
    
    return mfcc.astype(np.float32)

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

    # Split speakers into train/val
    all_speakers = list(speaker_dict.keys())
    val_speakers = set(all_speakers[::10])  # Every 10th speaker for validation
    train_speakers = {k: v for k, v in speaker_dict.items() if k not in val_speakers}
    val_speakers_dict = {k: v for k, v in speaker_dict.items() if k in val_speakers}
    
    print(f"Train speakers: {len(train_speakers)}, Val speakers: {len(val_speakers_dict)}")

    input_shape = (MAX_LEN, N_MFCC)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
        siamese_model = tf.keras.models.load_model(
            CHECKPOINT_PATH,
            custom_objects={"contrastive_loss": contrastive_loss, "distance_accuracy": distance_accuracy}
        )
    else:
        print("✨ No checkpoint found. Starting fresh training...")
        siamese_model = build_siamese_model(input_shape)
        siamese_model.compile(
            loss=contrastive_loss,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=ADAM_LR),
            metrics=[distance_accuracy]
        )

    siamese_model.summary()

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False,  # Keep latest for resuming
        save_weights_only=False,
        verbose=1
    )

    best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,  # Only save if validation loss improves
        save_weights_only=False,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_delta=0.005,
        min_lr=1e-5,
        verbose=1
    )

    print("\nStarting Training on M4 Pro...")
    try:
        history = siamese_model.fit(
            pair_generator(train_speakers, BATCH_SIZE, augment=True),
            validation_data=pair_generator(val_speakers_dict, BATCH_SIZE, augment=False),
            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
            validation_steps=5000 // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint_cb, best_model_cb, early_stopping, lr_reduce]
        )

        siamese_model.save("custom_voice_auth_final.keras")
        print("\n✅ Training Complete & Saved.")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted. Saving progress...")
        siamese_model.save(CHECKPOINT_PATH)
        print("✅ Progress saved.")