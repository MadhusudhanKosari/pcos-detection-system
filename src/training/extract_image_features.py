import numpy as np
from src.models.image_feature_extractor import get_image_feature_extractor
from src.data_preprocessing.image_dataset import create_image_dataset

def extract_and_save_features(output_path, batch_size=32):
    model = get_image_feature_extractor()
    dataset = create_image_dataset(batch_size=batch_size, shuffle=False)

    all_features = []
    all_labels = []

    for images, labels in dataset:
        features = model(images)
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    np.save(f"{output_path}_features.npy", all_features)
    np.save(f"{output_path}_labels.npy", all_labels)

    print("Features saved:")
    print("Features shape:", all_features.shape)
    print("Labels shape:", all_labels.shape)


if __name__ == "__main__":
    extract_and_save_features("outputs/image")
