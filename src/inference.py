import os
import torch
import cv2
import numpy as np
from model import AnomalyAE
import matplotlib.pyplot as plt

from config_utils import load_config, resolve_paths


def generate_heatmap(model_path, image_path, img_size=256):
    model = AnomalyAE.load_from_checkpoint(model_path).eval()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        reconstruction = model(input_tensor)

    residual = torch.abs(input_tensor - reconstruction).squeeze().permute(1, 2, 0).numpy()
    residual_gray = np.mean(residual, axis=2)

    eps = 1e-8
    residual_norm = (residual_gray - residual_gray.min()) / (residual_gray.max() - residual_gray.min() + eps)
    heatmap = cv2.applyColorMap((residual_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[1].imshow(reconstruction.squeeze().permute(1, 2, 0).numpy())
    axes[1].set_title("Reconstruction")
    axes[2].imshow(heatmap)
    axes[2].set_title("Anomaly Heatmap")
    plt.show()


def default_checkpoint_from_config(config):
    paths = resolve_paths(config)
    return os.path.join(paths["output_dir"], "last.ckpt")


if __name__ == "__main__":
    cfg = load_config()
    checkpoint = default_checkpoint_from_config(cfg)
    img_size = cfg.get("train_params", {}).get("img_size", 256)

    sample_image = "data/mvtec/bottle/test/broken_large/000.png"
    generate_heatmap(checkpoint, sample_image, img_size=img_size)
