from src.data_preprocessing.image_dataset_for_training import create_image_dataset

dataset = create_image_dataset("data/raw/images", batch_size=8)

for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Labels:", labels.numpy())
