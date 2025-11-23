import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as K

# --- UTILS ---
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, x):
        return K.l2_normalize(x, axis=self.axis)
    def get_config(self):
        return super().get_config() | {"axis": self.axis}

def euclidean_distance(vectors):
    a, b = vectors
    # Added K.epsilon() inside sqrt is standard, but let's be safer
    sum_sq = K.sum(K.square(a - b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_sq, K.epsilon()))

# --- ROBUST CONTRASTIVE LOSS ---
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    
    # 1. Squared Distance
    sq_pred = K.square(y_pred)
    
    # 2. Positive Loss (Pull together)
    # We add a small regularizer to prevent total collapse to 0
    pos_loss = sq_pred
    
    # 3. Negative Loss (Push apart)
    # The margin logic: if dist > margin, loss is 0.
    neg_loss = K.square(K.maximum(margin - y_pred, 0.0))
    
    # 4. Total
    return K.mean(y_true * pos_loss + (1.0 - y_true) * neg_loss)

# --- ECAPA-TDNN BLOCKS (Same as before) ---
def se_block(x, filters, ratio=8):
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Reshape((1, filters))(s)
    s = layers.Dense(filters // ratio, activation='relu', use_bias=False)(s)
    s = layers.Dense(filters, activation='sigmoid', use_bias=False)(s)
    return layers.Multiply()([x, s])

def res2net_block(x, filters, kernel_size, dilation, scale=4):
    inp = x
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, filters)
    if inp.shape[-1] != filters:
        inp = layers.Conv1D(filters, 1, padding='same')(inp)
    return layers.Add()([inp, x])

# --- MAIN ARCHITECTURE ---
def build_base_network(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Feature Extraction
    x = layers.Conv1D(128, 5, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # ECAPA Blocks
    x1 = res2net_block(x, 128, kernel_size=3, dilation=2)
    x2 = res2net_block(x1, 128, kernel_size=3, dilation=3)
    x3 = res2net_block(x2, 128, kernel_size=3, dilation=4)

    # Aggregation
    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Conv1D(256, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Pooling
    mean = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    std = layers.Lambda(lambda t: K.std(t, axis=1))(x)
    x = layers.Concatenate()([mean, max_p, std])
    
    # Embedding Head
    x = layers.Dense(192, activation=None)(x)
    x = layers.BatchNormalization()(x)
    
    # CRITICAL: We enforce L2 Normalize. 
    # If the inputs to this are all zero, it outputs NaNs or Zeros.
    # We add a small epsilon inside the normalization layer just in case,
    # but strictly speaking K.l2_normalize handles it.
    x = L2Normalize(name="l2_norm")(x)
    
    return models.Model(inputs, x, name="ECAPA_Encoder")

def build_siamese_model(input_shape):
    inp_a = layers.Input(shape=input_shape, name="Input_A")
    inp_b = layers.Input(shape=input_shape, name="Input_B")
    base = build_base_network(input_shape)
    fa = base(inp_a)
    fb = base(inp_b)
    dist = layers.Lambda(euclidean_distance, name="distance")([fa, fb])
    return models.Model([inp_a, inp_b], dist, name="SiameseECAPA")