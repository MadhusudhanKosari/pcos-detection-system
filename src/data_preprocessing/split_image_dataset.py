import os
import shutil
import random

RAW_BASE = "data/raw/images"
TARGET_BASE = "data/image_dataset"

CLASSES = ["pcos", "non_pcos"]
TRAIN_RATIO = 0.8   # 80% train, 20% validation

random.seed(42)

def split_class(class_name):
    src_dir = os.path.join(RAW_BASE, class_name)
    images = os.listdir(src_dir)

    images = [img for img in images if img.lower().endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_dir = os.path.join(TARGET_BASE, "train", class_name)
    val_dir = os.path.join(TARGET_BASE, "val", class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, img))

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

def main():
    for cls in CLASSES:
        split_class(cls)

    print("\nDataset splitting completed successfully.")

if __name__ == "__main__":
    main()
