import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as K

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    sq = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0.0))
    return K.mean(y_true * sq + (1.0 - y_true) * margin_sq)

def euclidean_distance(vectors):
    a, b = vectors
    sum_sq = K.sum(K.square(a - b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_sq, K.epsilon()))

def l2_normalize_layer(v):
    """Named function for L2 normalization (serializable)"""
    return K.l2_normalize(v, axis=1)

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
    x = layers.Lambda(l2_normalize_layer, name="l2_norm")(x)  # ← Now named function
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
    """
    y_true: 1 for same, 0 for different
    y_pred: distance
    
    If distance < 0.5, we predict "same" (1).
    If distance > 0.5, we predict "different" (0).
    """
    # cast boolean (dist < 0.5) to float (1.0 or 0.0)
    prediction = tf.cast(y_pred < 0.5, tf.float32)
    return K.mean(tf.equal(prediction, y_true))