import os
import time
import torch
import numpy as np
import mlflow

from config_utils import load_config, resolve_paths
from models import build_model


def run_benchmark(config, checkpoint_path, device="cuda", img_size=256):
    model = build_model(config.get("model", {}), config.get("train_params", {}))
    model.load(checkpoint_path, map_location=device)
    model.to(device).eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    for _ in range(10):
        _ = model.score(dummy_input, device=torch.device(device))

    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.score(dummy_input, device=torch.device(device))
        latencies.append(time.perf_counter() - start)

    print(f"Device: {device}")
    print(f"Average Latency: {np.mean(latencies) * 1000:.2f} ms")
    print(f"Throughput: {1 / np.mean(latencies):.2f} FPS")

    if mlflow.active_run():
        mlflow.log_metric("inference_latency_ms", np.mean(latencies) * 1000)
        mlflow.log_metric("throughput_fps", 1 / np.mean(latencies))


if __name__ == "__main__":
    cfg = load_config()
    paths = resolve_paths(cfg)
    checkpoint = os.path.join(paths["output_dir"], "last.pt")
    img_size = cfg.get("train_params", {}).get("img_size", 256)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(cfg, checkpoint, device=default_device, img_size=img_size)
