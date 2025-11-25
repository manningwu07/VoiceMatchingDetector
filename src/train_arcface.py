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
BATCH_SIZE = 64   
EPOCHS = 15       
STEPS_PER_EPOCH = 200 

LOAD_PATH = "ecapa_hard_mining.h5" 
SAVE_PATH = "ecapa_arcface.h5"
LR = 1e-4 

# --- DATA UTILS ---
def get_all_files():
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
                    
    print(f"Found {len(files)} files across {NUM_CLASSES} classes.")
    return files, labels, NUM_CLASSES

def augment_audio(y, sr):
    if random.random() > 0.5:
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

def data_generator(files, labels, batch_size, n_classes):
    indices = np.arange(len(files))
    while True:
        np.random.shuffle(indices)
        batch_x = []
        batch_y = []
        
        for i in indices:
            path = files[i]
            label = labels[i]
            
            x = preprocess(path, augment=True)
            if x is None: continue
            
            batch_x.append(x)
            batch_y.append(label)
            
            if len(batch_x) == batch_size:
                # One-Hot Encoding
                y_onehot = tf.keras.utils.to_categorical(batch_y, num_classes=n_classes)
                
                # CRITICAL FIX:
                # Inputs: [Audio, Labels] -> for ArcFace math
                # Targets: Labels -> for CrossEntropy Loss
                yield [np.array(batch_x), y_onehot], y_onehot
                
                batch_x, batch_y = [], []

# --- ARCFACE LAYER (FIXED SHAPE) ---
class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        # input_shape[0] is the embedding shape (Batch, 192)
        embedding_shape = input_shape[0]
        self.W = self.add_weight(name='W',
                                shape=(embedding_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        super(ArcFace, self).build(input_shape)

    def call(self, inputs):
        x, y = inputs
        # Normalize
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        # Dot Product
        logits = tf.matmul(x, W)
        # Clip
        logits = tf.clip_by_value(logits, -0.999, 0.999)
        # ArcFace Math
        theta = tf.math.acos(logits)
        target_logits = tf.math.cos(theta + self.m)
        # Scale
        logits = logits * self.s
        target_logits = target_logits * self.s
        # Combine
        out = logits * (1 - y) + target_logits * y
        return out

# --- MODEL BUILDER (FIXED INPUTS) ---
def build_arcface_model(base_model, n_classes):
    # 1. Create FRESH Inputs (Single Head, not Triplet)
    audio_input = tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC), name="audio_input")
    label_input = tf.keras.layers.Input(shape=(n_classes,), name='label_input')
    
    # 2. Extract the Trained Encoder
    # Since we loaded weights into 'base_model', this layer has the smart weights
    encoder = base_model.get_layer("ECAPA_Encoder")
    
    # 3. Connect Graph
    emb = encoder(audio_input)
    
    # 4. ArcFace Head
    output = ArcFace(n_classes=n_classes)( [emb, label_input] )
    output = tf.keras.layers.Softmax()(output)
    
    return tf.keras.Model([audio_input, label_input], output)

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Indexing Dataset for Classification...")
    files, labels, NUM_CLASSES = get_all_files()
    
    # Split
    total = len(files)
    idx_val = int(total * 0.1)
    val_files = files[:idx_val]
    val_labels = labels[:idx_val]
    train_files = files[idx_val:]
    train_labels = labels[idx_val:]
    
    print(f"Classes: {NUM_CLASSES}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # 1. Load OLD Model (Triplet)
    print(f"♻️  Loading Weights from {LOAD_PATH}...")
    dummy_siamese = build_siamese_model((MAX_LEN, N_MFCC)) 
    dummy_siamese.load_weights(LOAD_PATH)
    
    # 2. Build NEW Model (ArcFace)
    print("🏗️  Building ArcFace Head...")
    arc_model = build_arcface_model(dummy_siamese, NUM_CLASSES)
    
    # 3. Compile
    arc_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train
    print("🔥 Starting ArcFace Training...")
    arc_model.fit(
        data_generator(train_files, train_labels, BATCH_SIZE, NUM_CLASSES),
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=data_generator(val_files, val_labels, BATCH_SIZE, NUM_CLASSES),
        validation_steps=20,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                SAVE_PATH, 
                save_best_only=True, 
                monitor='val_accuracy', 
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
        ]
    )
    
    # Save JUST the encoder for verification later
    encoder = arc_model.get_layer("ECAPA_Encoder")
    encoder.save_weights("ecapa_arcface_final.h5")
    print("✅ Done. Encoder extracted to ecapa_arcface_final.h5")