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

# --- TRIPLET LOSS ---
# This is computed inside the model structure for easier Keras integration
def triplet_loss(inputs, margin=1.0):
    anchor, positive, negative = inputs
    
    # Distance(Anchor, Positive)
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    
    # Distance(Anchor, Negative)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    
    # Loss: max(pos_dist - neg_dist + margin, 0)
    basic_loss = pos_dist - neg_dist + margin
    loss = K.maximum(basic_loss, 0.0)
    return loss

# --- ECAPA-TDNN BLOCKS (UNCHANGED) ---
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
    
    x = layers.Conv1D(128, 5, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x1 = res2net_block(x, 128, kernel_size=3, dilation=2)
    x2 = res2net_block(x1, 128, kernel_size=3, dilation=3)
    x3 = res2net_block(x2, 128, kernel_size=3, dilation=4)

    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Conv1D(256, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    mean = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    std = layers.Lambda(lambda t: K.std(t, axis=1))(x)
    x = layers.Concatenate()([mean, max_p, std])
    
    x = layers.Dense(192, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = L2Normalize(name="l2_norm")(x)
    
    return models.Model(inputs, x, name="ECAPA_Encoder")


def build_siamese_model(input_shape):
    # Triplet Inputs
    inp_a = layers.Input(shape=input_shape, name="Anchor")
    inp_p = layers.Input(shape=input_shape, name="Positive")
    inp_n = layers.Input(shape=input_shape, name="Negative")

    base = build_base_network(input_shape)
    
    emb_a = base(inp_a)
    emb_p = base(inp_p)
    emb_n = base(inp_n)

    # --- Calculations inside the graph ---
    # 1. Calculate Distances
    pos_dist = K.sum(K.square(emb_a - emb_p), axis=1)
    neg_dist = K.sum(K.square(emb_a - emb_n), axis=1)
    
    # 2. Calculate Loss (Margin 1.0)
    basic_loss = pos_dist - neg_dist + 1.0
    loss = K.maximum(basic_loss, 0.0)
    
    # 3. Calculate Accuracy (Is Positive closer than Negative?)
    # If pos_dist < neg_dist, then we successfully identified the speaker.
    accuracy = K.mean(K.cast(pos_dist < neg_dist, tf.float32))
    
    # Construct Model
    model = models.Model([inp_a, inp_p, inp_n], loss, name="TripletModel")
    
    # 4. Inject the Metric
    model.add_metric(accuracy, name="triplet_accuracy")
    
    return model