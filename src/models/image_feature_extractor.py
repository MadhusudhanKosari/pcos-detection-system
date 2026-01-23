import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def get_image_feature_extractor():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all layers (CPU-friendly)
    base_model.trainable = False

    # Global average pooling to get feature vector
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    return model
