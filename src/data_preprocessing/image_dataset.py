import os
import tensorflow as tf
from src.data_preprocessing.image_preprocessing import preprocess_image

PCOS_DIR = "data/raw/images/pcos"
NON_PCOS_DIR = "data/raw/images/non_pcos"

def get_image_paths_and_labels():
    image_paths = []
    labels = []

    # PCOS images -> label 1
    for file in os.listdir(PCOS_DIR):
        image_paths.append(os.path.join(PCOS_DIR, file))
        labels.append(1)

    # Non-PCOS images -> label 0
    for file in os.listdir(NON_PCOS_DIR):
        image_paths.append(os.path.join(NON_PCOS_DIR, file))
        labels.append(0)

    return image_paths, labels


def create_image_dataset(batch_size=32, shuffle=True):
    image_paths, labels = get_image_paths_and_labels()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
