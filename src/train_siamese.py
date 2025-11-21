import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as K
import librosa
import numpy as np
import os
import random
from glob import glob
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# MFCC Parameters: The "Fingerprint" of the audio
SAMPLE_RATE = 16000
DURATION = 3 # Seconds
N_MFCC = 40  # How many feature dimensions to extract
MAX_LEN = 130 # Expected time steps for 3 seconds of MFCCs

BATCH_SIZE = 64
EPOCHS = 15
PAIRS_PER_EPOCH = 1000 # How many pairs to generate per epoch
CHECKPOINT_PATH = "siamese_checkpoint.keras"

print(f"🚀 TensorFlow Version: {tf.__version__} (Mac Optimized)")

# --- 1. FEATURE EXTRACTION (The "From Scratch" Part) ---
def preprocess_audio(path):
    """
    1. Load Audio
    2. Strip Silence
    3. Compute MFCCs (The voice texture)
    4. Pad/Crop to fixed size
    """
    try:
        # Load and resample
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Extract MFCCs
        # Shape: (N_MFCC, Time) -> (40, 130)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # Transpose to (Time, Feats) for the LSTM/CNN
        mfcc = mfcc.T 
        
        # Fix Length (Pad or Crop)
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:MAX_LEN, :]
            
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros((MAX_LEN, N_MFCC), dtype=np.float32)

# --- 2. DATA GENERATOR ---
def pair_generator(speaker_dict, batch_size=32):
    """
    Infinite generator that yields pairs of audio features.
    Half match (Label 1), Half mismatch (Label 0).
    """
    speakers = list(speaker_dict.keys())
    
    while True:
        x1_batch = []
        x2_batch = []
        y_batch = []
        
        for _ in range(batch_size):
            is_same = random.random() > 0.5
            
            if is_same:
                # Positive Pair (Same Speaker)
                spk = random.choice(speakers)
                f1, f2 = random.sample(speaker_dict[spk], 2)
                label = 1.0
            else:
                # Negative Pair (Different Speakers)
                s1, s2 = random.sample(speakers, 2)
                f1 = random.choice(speaker_dict[s1])
                f2 = random.choice(speaker_dict[s2])
                label = 0.0
            
            # Process on the fly
            feat1 = preprocess_audio(f1)
            feat2 = preprocess_audio(f2)
            
            x1_batch.append(feat1)
            x2_batch.append(feat2)
            y_batch.append(label)
            
        yield [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)

# --- 3. BUILD MODEL (The Siamese Architecture) ---
def build_base_network(input_shape):
    """
    The shared encoder. It learns to map audio to a 128-dim vector.
    We use a 1D CNN + LSTM hybrid (Classic Speech Architecture).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Conv Block 2
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # LSTM to capture temporal sequence
    x = layers.LSTM(128, return_sequences=False)(x)
    
    # Projection Head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    return models.Model(inputs, x, name="Shared_Encoder")

def build_siamese_model(input_shape):
    # 1. Define Inputs
    input_a = layers.Input(shape=input_shape, name="Input_A")
    input_b = layers.Input(shape=input_shape, name="Input_B")
    
    # 2. Shared Encoder (Weights are same for both paths)
    base_net = build_base_network(input_shape)
    vect_a = base_net(input_a)
    vect_b = base_net(input_b)
    
    # 3. Euclidean Distance Layer
    # We calculate the distance between the two vectors
    def euclidean_distance(vectors):
        (featsA, featsB) = vectors
        sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

    distance = layers.Lambda(euclidean_distance)([vect_a, vect_b])
    
    # 4. Output (Sigmoid: 0=Diff, 1=Same)
    # We invert distance logic: small distance -> high prob
    # Dense layer learns the threshold
    outputs = layers.Dense(1, activation="sigmoid")(distance)
    
    return models.Model(inputs=[input_a, input_b], outputs=outputs)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Load Data Paths
    print("Scanning dataset...")
    data_path = "./data/LibriSpeech/train-clean-360" 
    
    speaker_dict = {}
    files = glob(data_path + "/**/*.flac", recursive=True)
    for f in files:
        spk_id = f.split(os.sep)[-3]
        if spk_id not in speaker_dict: speaker_dict[spk_id] = []
        speaker_dict[spk_id].append(f)
        
    speaker_dict = {k:v for k,v in speaker_dict.items() if len(v) >= 2}
    print(f"Found {len(speaker_dict)} valid speakers.")
    
    # B. MODEL SETUP (Resume or Create)
    input_shape = (MAX_LEN, N_MFCC) 
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
        # load_model restores: Architecture + Weights + Optimizer State
        siamese_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    else:
        print("✨ No checkpoint found. Starting fresh training...")
        siamese_model = build_siamese_model(input_shape)
        siamese_model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"]
        )
    
    siamese_model.summary()
    
    # C. CALLBACKS (The Auto-Saver)
    # This saves the model every time 'val_loss' improves or every epoch.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False, # Set True if you have validation data, False to save latest
        save_weights_only=False, # False = Save entire model (optimizer + architecture)
        verbose=1 # Print "Saving model..."
    )

    # D. TRAIN
    print("\nStarting Training on MPS/CPU...")
    try:
        history = siamese_model.fit(
            pair_generator(speaker_dict, BATCH_SIZE),
            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint_cb] # <--- Add the callback here
        )
        
        # Save final version separately just in case
        siamese_model.save("custom_voice_auth_final.keras")
        print("\n✅ Training Complete & Saved.")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving progress...")
        siamese_model.save(CHECKPOINT_PATH)
        print("✅ Progress saved. You can resume later.")