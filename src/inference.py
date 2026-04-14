import os
import glob
import argparse
from contextlib import nullcontext
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


def _resolve_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_heatmap(model_path, image_path, img_size=256, save_dir=None, show=True, device=None):
    device = _resolve_device(device)
    model = AnomalyAE.load_from_checkpoint(model_path, map_location=device).to(device).eval()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    input_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(device)

    with torch.no_grad():
        reconstruction = model(input_tensor)

    input_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    recon_np = reconstruction.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    residual = np.abs(input_np - recon_np)
    residual_gray = np.mean(residual, axis=2)

    eps = 1e-8
    residual_norm = (residual_gray - residual_gray.min()) / (residual_gray.max() - residual_gray.min() + eps)
    heatmap = cv2.applyColorMap((residual_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[1].imshow(recon_np)
    axes[1].set_title("Reconstruction")
    axes[2].imshow(heatmap)
    axes[2].set_title("Anomaly Heatmap")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        recon_img = _to_uint8(recon_np)
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


def save_prediction_artifacts(model_path, image_paths, img_size=256, out_dir=None, show=False, device=None):
    if out_dir is None:
        out_dir = "artifacts/predictions"

    for img_path in image_paths:
        generate_heatmap(model_path, img_path, img_size=img_size, save_dir=out_dir, show=show, device=device)

    if mlflow and mlflow.active_run():
        mlflow.log_artifacts(out_dir, artifact_path="predictions")


def default_checkpoint_from_config(config):
    paths = resolve_paths(config)
    return os.path.join(paths["output_dir"], "last.ckpt")


def collect_test_samples(data_dir, category, num_samples=6, prefer_defect=True):
    test_root = os.path.join(data_dir, category, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    defect_paths = []
    good_paths = []
    for sub in sorted(os.listdir(test_root)):
        sub_path = os.path.join(test_root, sub)
        if not os.path.isdir(sub_path):
            continue
        imgs = sorted(glob.glob(os.path.join(sub_path, "*.png")))
        if sub == "good":
            good_paths.extend(imgs)
        else:
            defect_paths.extend(imgs)

    ordered = defect_paths + good_paths if prefer_defect else good_paths + defect_paths
    if not ordered:
        raise ValueError(f"No PNG images found under: {test_root}")
    return ordered[:num_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AE reconstruction/heatmap artifacts for test images.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt file. Defaults to output_dir/last.ckpt")
    parser.add_argument("--num-samples", type=int, default=6, help="How many test images to visualize")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows")
    parser.add_argument("--no-prefer-defect", action="store_true", help="Pick good samples first instead of defect-first")
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda or cpu (default: auto)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow/DagsHub artifact logging")
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    args = parser.parse_args()

    cfg = load_config()
    checkpoint = args.checkpoint or default_checkpoint_from_config(cfg)
    img_size = cfg.get("train_params", {}).get("img_size", 256)
    paths = resolve_paths(cfg)
    category = cfg.get("experiment", {}).get("category", "bottle")

    artifacts_dir = os.path.join(paths["output_dir"], "artifacts", "predictions")
    samples = collect_test_samples(
        paths["data_dir"],
        category=category,
        num_samples=args.num_samples,
        prefer_defect=not args.no_prefer_defect,
    )

    run_ctx = nullcontext()
    if mlflow and not args.no_mlflow and not mlflow.active_run():
        mlflow.set_tracking_uri(paths["log_dir"])
        mlflow.set_experiment("quali-trace-inference")
        run_ctx = mlflow.start_run(run_name=args.run_name)

    with run_ctx:
        save_prediction_artifacts(
            checkpoint,
            samples,
            img_size=img_size,
            out_dir=artifacts_dir,
            show=args.show,
            device=args.device,
        )

    print(f"Saved {len(samples)} predictions to: {artifacts_dir}")
