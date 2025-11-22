import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as K

# --- CUSTOM LAYER (Replaces Lambda for safe serialization) ---
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return K.l2_normalize(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    sq = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0.0))
    return K.mean(y_true * sq + (1.0 - y_true) * margin_sq)

def euclidean_distance(vectors):
    a, b = vectors
    sum_sq = K.sum(K.square(a - b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_sq, K.epsilon()))

def build_base_network(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 7, padding="same", strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation=None)(x)
    
    # UPDATED: Use class instead of Lambda
    x = L2Normalize(name="l2_norm")(x)
    
    return models.Model(inputs, x, name="Shared_Encoder")

def build_siamese_model(input_shape):
    inp_a = layers.Input(shape=input_shape, name="Input_A")
    inp_b = layers.Input(shape=input_shape, name="Input_B")

    base = build_base_network(input_shape)
    fa = base(inp_a)
    fb = base(inp_b)

    dist = layers.Lambda(euclidean_distance, name="distance")([fa, fb])
    return models.Model([inp_a, inp_b], dist, name="SiameseContrastive")

# --- METRICS ---
def distance_accuracy(y_true, y_pred):
    prediction = tf.cast(y_pred < 0.5, tf.float32)
    return K.mean(tf.equal(prediction, y_true))