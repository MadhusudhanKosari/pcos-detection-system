import numpy as np
import tensorflow as tf

from src.data_preprocessing.image_dataset_for_training import create_image_dataset
from src.models.image_feature_extractor import get_image_feature_extractor


def extract_and_save_image_features():
    # 1. Load dataset
    dataset = create_image_dataset(
        "data/raw/images",
        batch_size=16
    )

    # 2. Load feature extractor
    feature_extractor = get_image_feature_extractor()

    all_features = []
    all_labels = []

    # 3. Extract features batch by batch
    for images, labels in dataset:
        features = feature_extractor(images)
        features = features.numpy()

        all_features.append(features)
        all_labels.append(labels.numpy())

    # 4. Convert to NumPy arrays
    X = np.vstack(all_features)
    y = np.vstack(all_labels).ravel()

    # 5. Save features and labels
    np.save("data/processed/image_features.npy", X)
    np.save("data/processed/image_labels.npy", y)

    print("Image feature extraction complete.")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)


if __name__ == "__main__":
    extract_and_save_image_features()
