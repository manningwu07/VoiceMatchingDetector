import tensorflow as tf
from model import build_siamese_model

input_shape = (130, 67)
model = build_siamese_model(input_shape)

# Generate and save the diagram
tf.keras.utils.plot_model(
    model,
    to_file="siamese_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",  # Top to Bottom
    expand_nested=True,
    dpi=150
)

print("✅ Architecture diagram saved to: siamese_architecture.png")