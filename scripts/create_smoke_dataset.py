import os
import numpy as np
from PIL import Image

def generate_smoke_dataset():
    # Base directory matching standard MVTec layout structure
    base_dir = "data/mvtec/bottle"
    
    # Define standard directory branches
    train_dir = os.path.join(base_dir, "train", "good")
    test_good_dir = os.path.join(base_dir, "test", "good")
    test_defect_dir = os.path.join(base_dir, "test", "broken") # Anomaly category
    
    # Ensure all directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)

    print("🛠️ Generating synthetic MVTec bottle dataset for smoke testing...")

    # Generate a few placeholder 256x256 RGB images
    # We use random data since the objective is to verify code execution, not ML performance.
    img_size = (256, 256, 3)
    num_samples = 2

    # Generate Training data (Only 'good' samples)
    for i in range(num_samples):
        random_pixels = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        img = Image.fromarray(random_pixels)
        img.save(os.path.join(train_dir, f"{i:03d}.png"))

    # Generate Testing data ('good' baseline validation samples)
    for i in range(num_samples):
        random_pixels = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        img = Image.fromarray(random_pixels)
        img.save(os.path.join(test_good_dir, f"{i:03d}.png"))

    # Generate Testing data ('broken' anomaly validation samples)
    for i in range(num_samples):
        random_pixels = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        img = Image.fromarray(random_pixels)
        img.save(os.path.join(test_defect_dir, f"{i:03d}.png"))

    print(f"Smoke dataset structure successfully created at: {base_dir}")
    print(f"├── Train ('good'): {num_samples} images")
    print(f"└── Test ('good' + 'broken'): {num_samples * 2} images")

if __name__ == "__main__":
    generate_smoke_dataset()