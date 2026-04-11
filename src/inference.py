import os
import torch
import cv2
import numpy as np
from pathlib import Path
from model import AnomalyAE
import matplotlib.pyplot as plt

try:
    import mlflow
except Exception:  # optional dependency
    mlflow = None

from config_utils import load_config, resolve_paths




def _to_uint8(img):
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def generate_heatmap(model_path, image_path, img_size=256, save_dir=None, show=True):
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

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        recon_img = _to_uint8(reconstruction.squeeze().permute(1, 2, 0).numpy())
        orig_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(save_dir / f"{stem}_orig.png"), orig_bgr)
        cv2.imwrite(str(save_dir / f"{stem}_recon.png"), cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"{stem}_heatmap.png"), heatmap)
        fig.savefig(str(save_dir / f"{stem}_panel.png"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def save_prediction_artifacts(model_path, image_paths, img_size=256, out_dir=None, show=False):
    if out_dir is None:
        out_dir = "artifacts/predictions"

    for img_path in image_paths:
        generate_heatmap(model_path, img_path, img_size=img_size, save_dir=out_dir, show=show)

    if mlflow and mlflow.active_run():
        mlflow.log_artifacts(out_dir, artifact_path="predictions")


def default_checkpoint_from_config(config):
    paths = resolve_paths(config)
    return os.path.join(paths["output_dir"], "last.ckpt")


if __name__ == "__main__":
    cfg = load_config()
    checkpoint = default_checkpoint_from_config(cfg)
    img_size = cfg.get("train_params", {}).get("img_size", 256)
    paths = resolve_paths(cfg)

    sample_image = "data/mvtec/bottle/test/broken_large/000.png"
    artifacts_dir = os.path.join(paths["output_dir"], "artifacts", "predictions")
    save_prediction_artifacts(checkpoint, [sample_image], img_size=img_size, out_dir=artifacts_dir, show=True)
