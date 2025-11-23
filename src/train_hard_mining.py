import tensorflow as tf
import numpy as np
import os
import random
import librosa
from model import build_siamese_model

# --- CONFIG ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 80       
MAX_LEN = 130     
BATCH_SIZE = 64   # Reduced to 64 for stability during hard mining
EPOCHS = 30       
STEPS_PER_EPOCH = 150

LOAD_PATH = "ecapa_final_armored.h5" 
SAVE_PATH = "ecapa_hard_mining_fixed.h5"
LR = 3e-6         # SUPER LOW LR to prevent collapse

# --- DATA ---
def augment_audio(y, sr):
    if random.random() > 0.6:
        noise = np.random.normal(0, 0.001, len(y))
        y = y + noise
    return y

def preprocess(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        if np.max(np.abs(y)) < 0.005: return None
        if augment: y = augment_audio(y, sr)
        
        target = int(SAMPLE_RATE * DURATION)
        if len(y) > target:
            start = random.randint(0, len(y) - target)
            y = y[start:start+target]
        else:
            y = np.pad(y, (0, target - len(y)))
            
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        
        if mfcc.shape[0] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:MAX_LEN, :]
        return mfcc.astype(np.float32)
    except:
        return None

def get_speaker_data():
    print("📂 Indexing Data...")
    data_path = "./data/LibriSpeech/train-clean-360"
    speaker_dict = {}
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".flac"):
                spk = root.split(os.sep)[-2]
                if spk not in speaker_dict: speaker_dict[spk] = []
                speaker_dict[spk].append(os.path.join(root, f))
    return {k:v for k,v in speaker_dict.items() if len(v) >= 2}

# --- MODEL ---
class HardMiningModel(tf.keras.Model):
    def __init__(self, base_model, margin=1.0):
        super().__init__()
        self.encoder = base_model.get_layer("ECAPA_Encoder")
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="val_acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def train_step(self, data):
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            anc_emb = self.encoder(anchors, training=True)
            pos_emb = self.encoder(positives, training=True)
            
            # --- Batch Hard Logic ---
            # 1. Pos Dist
            pos_dist = tf.reduce_sum(tf.square(anc_emb - pos_emb), axis=1)
            
            # 2. Neg Dist (All vs All)
            cross_dot = tf.matmul(anc_emb, tf.transpose(pos_emb))
            anc_sq = tf.reduce_sum(tf.square(anc_emb), axis=1, keepdims=True)
            pos_sq = tf.reduce_sum(tf.square(pos_emb), axis=1, keepdims=True)
            cross_dist = anc_sq - 2.0 * cross_dot + tf.transpose(pos_sq)
            cross_dist = tf.maximum(cross_dist, 0.0)
            
            # Mask diagonal (Self)
            batch_size = tf.shape(anchors)[0]
            diag_mask = tf.eye(batch_size) * 1e9
            masked_neg_dist = cross_dist + diag_mask
            
            # Hardest Negative
            hardest_neg_dist = tf.reduce_min(masked_neg_dist, axis=1)
            
            # Loss
            loss = tf.maximum(pos_dist - hardest_neg_dist + self.margin, 0.0)
            loss = tf.reduce_mean(loss)

        # Gradient Clipping to prevent Collapse
        grads = tape.gradient(loss, self.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # REAL VALIDATION: Check both Positive and Negative pairs
        anchors, positives = data[0], data[1]
        
        anc_emb = self.encoder(anchors, training=False)
        pos_emb = self.encoder(positives, training=False)
        
        # 1. Positive Distance (Should be small)
        pos_dist = tf.reduce_sum(tf.square(anc_emb - pos_emb), axis=1)
        
        # 2. Negative Distance (Should be large)
        # We shift positives by 1 to create artificial negatives
        neg_emb = tf.roll(pos_emb, shift=1, axis=0) 
        neg_dist = tf.reduce_sum(tf.square(anc_emb - neg_emb), axis=1)
        
        # Accuracy: How often is Pos < Neg?
        correct = tf.cast(pos_dist < neg_dist, tf.float32)
        self.acc_tracker.update_state(correct)
        
        return {"val_acc": self.acc_tracker.result()}

# --- GENERATOR ---
def pair_generator(speaker_dict, batch_size=64):
    speakers = list(speaker_dict.keys())
    while True:
        a_batch, p_batch = [], []
        while len(a_batch) < batch_size:
            spk = random.choice(speakers)
            if len(speaker_dict[spk]) < 2: continue
            f1, f2 = random.sample(speaker_dict[spk], 2)
            
            a = preprocess(f1, augment=True)
            if a is None: continue
            p = preprocess(f2, augment=True)
            if p is None: continue
            
            a_batch.append(a); p_batch.append(p)
                
        yield (np.array(a_batch), np.array(p_batch))

# --- MAIN ---
if __name__ == "__main__":
    print(f"🔥 Starting Stabilized Hard Mining from: {LOAD_PATH}")
    
    full_model = build_siamese_model((MAX_LEN, N_MFCC))
    try:
        full_model.load_weights(LOAD_PATH)
        print("✅ Weights loaded.")
    except:
        print("❌ CRITICAL: No weights found. Stop.")
        exit()

    mining_model = HardMiningModel(full_model, margin=0.6) # Gentle Margin
    
    # CLIPNORM IS CRITICAL HERE
    mining_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR, clipnorm=1.0))

    spk_dict = get_speaker_data()
    keys = sorted(list(spk_dict.keys()))
    train_dict = {k: spk_dict[k] for k in keys if k not in keys[::10]}
    val_dict = {k: spk_dict[k] for k in keys if k in keys[::10]}

    train_ds = tf.data.Dataset.from_generator(
        lambda: pair_generator(train_dict, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 130, 80), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, 130, 80), dtype=tf.float32)
        )
    )
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: pair_generator(val_dict, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 130, 80), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, 130, 80), dtype=tf.float32)
        )
    )

    mining_model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=50,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_acc', verbose=1, save_weights_only=True)
        ]
    )
    print("✅ Done.")