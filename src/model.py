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
    sum_sq = K.sum(K.square(a - b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_sq, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    sq = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0.0))
    return K.mean(y_true * sq + (1.0 - y_true) * margin_sq)

# --- ECAPA-TDNN BLOCKS ---
def se_block(x, filters, ratio=8):
    """Squeeze-and-Excitation block to weight channels adaptively"""
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Reshape((1, filters))(s)
    s = layers.Dense(filters // ratio, activation='relu', use_bias=False)(s)
    s = layers.Dense(filters, activation='sigmoid', use_bias=False)(s)
    return layers.Multiply()([x, s])

def res2net_block(x, filters, kernel_size, dilation, scale=4):
    """Res2Net-style block with Dilation for long context"""
    inp = x
    # 1x1 Conv to adjust channels
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Dilated Conv (The magic part)
    x = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 1x1 Conv out
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # SE Attn
    x = se_block(x, filters)
    
    # Skip connection
    if inp.shape[-1] != filters:
        inp = layers.Conv1D(filters, 1, padding='same')(inp)
        
    return layers.Add()([inp, x])

# --- MAIN ARCHITECTURE ---
def build_base_network(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Initial Feature Extraction
    x = layers.Conv1D(128, 5, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. ECAPA-TDNN Blocks (Dilations: 2, 3, 4)
    # Captures 3ms, 10ms, 50ms contexts simultaneously
    x1 = res2net_block(x, 128, kernel_size=3, dilation=2)
    x2 = res2net_block(x1, 128, kernel_size=3, dilation=3)
    x3 = res2net_block(x2, 128, kernel_size=3, dilation=4)

    # 3. Multi-scale Feature Aggregation
    x = layers.Concatenate()([x1, x2, x3])
    
    # 4. 1x1 Conv
    x = layers.Conv1D(256, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 5. Attentive Statistics Pooling (Simplified)
    # Instead of just Average, we take Mean + Std + Max to capture all stats
    mean = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    std = layers.Lambda(lambda t: K.std(t, axis=1))(x)
    x = layers.Concatenate()([mean, max_p, std])
    
    # 6. Embedding Head
    x = layers.Dense(192, activation=None)(x) # Standard ECAPA size
    x = layers.BatchNormalization()(x)
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