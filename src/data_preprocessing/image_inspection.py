import os
from PIL import Image

PCOS_DIR = "data/raw/images/pcos"
NON_PCOS_DIR = "data/raw/images/non_pcos"

def inspect_images(folder, label, max_samples=5):
    print(f"\nInspecting {label} images...")
    count = 0

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        try:
            with Image.open(file_path) as img:
                print(f"{file} -> size: {img.size}, mode: {img.mode}")
                count += 1
        except Exception as e:
            print(f"ERROR reading {file}: {e}")

        if count >= max_samples:
            break

if __name__ == "__main__":
    inspect_images(PCOS_DIR, "PCOS")
    inspect_images(NON_PCOS_DIR, "NON-PCOS")
