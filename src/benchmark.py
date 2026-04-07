import torch
import time
from model import AnomalyAE
import numpy as np

import mlflow




def run_benchmark(checkpoint_path, device="cuda"):
    # load the trained model, here AE but more generic later
    model = AnomalyAE.load_from_checkpoint(checkpoint_path)
    model.to(device).eval()
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # warm-up start
    for _ in range(10):
        _ = model(dummy_input)
        
    # timed runs
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append(time.perf_counter() - start)
        
    print(f"Device: {device}")
    print(f"Average Latency: {np.mean(latencies)*1000:.2f} ms")
    print(f"Throughput: {1/np.mean(latencies):.2f} FPS")

    if mlflow.active_run():
        mlflow.log_metric("inference_latency_ms", np.mean(latencies) * 1000)
        mlflow.log_metric("throughput_fps", 1 / np.mean(latencies))

if __name__ == "__main__":
    run_benchmark("models/last.ckpt")