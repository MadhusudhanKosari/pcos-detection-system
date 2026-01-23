import os
import tensorflow as tf


def create_image_dataset(image_dir, img_size=(224, 224), batch_size=32):
    """
    Creates a TensorFlow dataset from folder-based image labels.
    """

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset
