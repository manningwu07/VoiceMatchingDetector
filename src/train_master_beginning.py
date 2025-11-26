# Use Adam/RMS prop to speed up convergence
# However, adam will drive into a cliff because of large gradients from ArcFace
# So we swap to SGD after a few epochs to prevent gradient explosions
# This is a known issue with margin-based losses

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
BATCH_SIZE = 128  
EPOCHS = 10       
LR = 5e-5         

SAVE_PATH = "ecapa_master.h5"

# --- 1. ROBUST DATASET ---
def get_dataset_info():
    print("📂 Indexing Data...")
    data_path = "./data/LibriSpeech/train-clean-360"
    files = []
    labels = []
    
    speakers = sorted([d for d in os.listdir(data_path) if not d.startswith('.')])
    label_map = {spk: i for i, spk in enumerate(speakers)}
    NUM_CLASSES = len(speakers)
    
    for spk in speakers:
        spk_dir = os.path.join(data_path, spk)
        for root, dirs, f_names in os.walk(spk_dir):
            for f in f_names:
                if f.endswith(".flac"):
                    files.append(os.path.join(root, f))
                    labels.append(label_map[spk])
    
    # Shuffle now to ensure mixing
    c = list(zip(files, labels))
    random.shuffle(c)
    files, labels = zip(*c)
    
    print(f"✅ Found {len(files)} files | {NUM_CLASSES} Speakers")
    return list(files), list(labels), NUM_CLASSES

def augment_audio(y, sr):
    if random.random() > 0.7:
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
            
        if np.isnan(mfcc).any(): return None
        return mfcc.astype(np.float32)
    except:
        return None

def train_generator(files, labels, batch_size, n_classes):
    idx = 0
    while True:
        batch_x = []
        batch_y = [] # Indices
        batch_y_hot = [] # OneHot
        
        while len(batch_x) < batch_size:
            if idx >= len(files):
                idx = 0
                c = list(zip(files, labels))
                random.shuffle(c)
                files, labels = zip(*c)
            
            f = files[idx]
            l = labels[idx]
            idx += 1
            
            x = preprocess(f, augment=True)
            if x is None: continue
            
            batch_x.append(x)
            batch_y.append(l)
            batch_y_hot.append(tf.keras.utils.to_categorical(l, n_classes))
            
        yield [np.array(batch_x), np.array(batch_y_hot)], np.array(batch_y_hot)

# --- 2. VALIDATION CALLBACK (FIXED) ---
class VerificationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_files, val_labels):
        super().__init__()
        self.val_files = val_files
        self.val_labels = val_labels
        self.encoder_model = None 

    def on_epoch_end(self, epoch, logs=None):
        if self.encoder_model is None:
            input_layer = self.model.layers[0].input 
            encoder_layer = self.model.get_layer("ECAPA_Encoder")
            self.encoder_model = encoder_layer
        
        pos_dists = []
        neg_dists = []
        
        # Test 500 pairs
        for i in range(500):
            # Pos Pair
            idx = random.randint(0, len(self.val_files)-1)
            f1, l1 = self.val_files[idx], self.val_labels[idx]
            
            f2 = None
            # Find match (Safe Loop)
            for _ in range(50):
                idx2 = random.randint(0, len(self.val_files)-1)
                if self.val_labels[idx2] == l1:
                    f2 = self.val_files[idx2]
                    break
            
            if f2 is None: continue # Skip if no pair found
            
            x1 = preprocess(f1)
            x2 = preprocess(f2)
            if x1 is None or x2 is None: continue
            
            v1 = self.encoder_model(x1[np.newaxis, ...], training=False)
            v2 = self.encoder_model(x2[np.newaxis, ...], training=False)
            
            v1 = tf.nn.l2_normalize(v1, axis=1)
            v2 = tf.nn.l2_normalize(v2, axis=1)
            dist = 1.0 - tf.reduce_sum(v1 * v2)
            pos_dists.append(dist)
            
            # Neg Pair
            f3 = None
            for _ in range(50):
                idx3 = random.randint(0, len(self.val_files)-1)
                if self.val_labels[idx3] != l1:
                    f3 = self.val_files[idx3]
                    break
            
            if f3 is None: continue
            
            x3 = preprocess(f3)
            if x3 is None: continue
            v3 = self.encoder_model(x3[np.newaxis, ...], training=False)
            v3 = tf.nn.l2_normalize(v3, axis=1)
            dist_n = 1.0 - tf.reduce_sum(v1 * v3)
            neg_dists.append(dist_n)

        # Calculate Threshold
        thresholds = np.arange(0, 2.0, 0.05)
        best_acc = 0
        best_thresh = 0
        
        pos_dists = np.array(pos_dists)
        neg_dists = np.array(neg_dists)
        
        if len(pos_dists) > 0 and len(neg_dists) > 0:
            for t in thresholds:
                tp = np.sum(pos_dists < t)
                tn = np.sum(neg_dists >= t)
                acc = (tp + tn) / (len(pos_dists) + len(neg_dists))
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = t
                    
        print(f"\n📊 [Verification] Best Acc: {best_acc*100:.2f}% | Thresh: {best_thresh:.2f}")
        logs['val_verification_acc'] = best_acc

# --- 3. ARCFACE LAYER ---
class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        # Force L2 regularization to prevent weight explosion
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        embedding_shape = input_shape[0]
        self.W = self.add_weight(name='W',
                                shape=(embedding_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        super(ArcFace, self).build(input_shape)

    def call(self, inputs):
        x, y = inputs
        # Force Float32 for numerical stability on M4
        x = tf.cast(x, tf.float32)
        
        x = tf.nn.l2_normalize(x, axis=1, epsilon=1e-6)
        W = tf.nn.l2_normalize(self.W, axis=0, epsilon=1e-6)
        
        logits = tf.matmul(x, W)
        
        # AGGRESSIVE CLIPPING (The Fix)
        logits = tf.clip_by_value(logits, -0.95, 0.95)
        
        theta = tf.math.acos(logits)
        target_logits = tf.math.cos(theta + self.m)
        
        logits = logits * self.s
        target_logits = target_logits * self.s
        
        out = logits * (1 - y) + target_logits * y
        return out

# --- 4. BUILD MODEL ---
def build_arcface_model(n_classes):
    base_model = build_siamese_model((MAX_LEN, N_MFCC))
    encoder = base_model.get_layer("ECAPA_Encoder")
    
    audio_inp = tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC))
    label_inp = tf.keras.layers.Input(shape=(n_classes,))
    
    emb = encoder(audio_inp)
    
    output = ArcFace(n_classes=n_classes)([emb, label_inp])
    output = tf.keras.layers.Softmax()(output)
    
    return tf.keras.Model([audio_inp, label_inp], output)

# --- MAIN ---
if __name__ == "__main__":
    files, labels, n_classes = get_dataset_info()
    
    split = int(len(files) * 0.9)
    train_files, val_files = files[:split], files[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    print("🏗️  Building ArcFace...")
    model = build_arcface_model(n_classes)
    
    # --- RESUME LOGIC ---
    if os.path.exists(SAVE_PATH):
        print(f"♻️  Resuming from checkpoint: {SAVE_PATH}")
        # We need to build the model first to load weights
        # Compile it to ensure variables are initialized
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        try:
            model.load_weights(SAVE_PATH)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("Starting from scratch...")
    # --------------------

    # RE-COMPILE with Safety Clips (Critical Fix)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=LR,
            clipnorm=2.0  # <--- PREVENTS EXPLOSIONS
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Put verification callback FIRST to ensure logs are updated before Checkpoint
    callbacks = [
        VerificationCallback(val_files, val_labels),
        tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_verification_acc', mode='max', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)
    ]
    
    print("🔥 Starting Training...")
    model.fit(
        train_generator(train_files, train_labels, BATCH_SIZE, n_classes),
        steps_per_epoch=len(train_files) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("✅ Done.")