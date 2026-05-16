import argparse
import csv
import json
import os

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from config_utils import load_config, resolve_data_info, resolve_paths
from data_loader import MVTecDataModule
from model import AnomalyAE


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_checkpoint_from_config(config: dict) -> str:
    paths = resolve_paths(config)
    return os.path.join(paths["output_dir"], "last.ckpt")


def score_autoencoder(model: AnomalyAE, dataloader, device: torch.device) -> tuple[list[float], list[int]]:
    model.to(device).eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x_hat = model(x)
            batch_scores = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))

            scores.extend(batch_scores.detach().cpu().tolist())
            labels.extend(y.detach().cpu().int().tolist())

    return scores, labels


def save_scores_csv(scores: list[float], labels: list[int], output_dir: str, filename: str = "evaluation_scores.csv") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "label", "score"])
        writer.writeheader()
        for idx, (score, label) in enumerate(zip(scores, labels)):
            writer.writerow({"sample_index": idx, "label": label, "score": score})

    return path


def save_report(report: dict, output_dir: str, filename: str = "evaluation_report.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return path


def register_candidate_model(model: AnomalyAE, registry_name: str, candidate_tag: str):
    model.cpu().eval()
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=registry_name,
    )

    if model_info.registered_model_version:
        client = MlflowClient()
        client.set_model_version_tag(
            name=registry_name,
            version=model_info.registered_model_version,
            key="status",
            value=candidate_tag,
        )

    mlflow.set_tag("model_registry_name", registry_name)
    mlflow.set_tag("model_candidate_status", candidate_tag)
    return model_info


def evaluate(config_path: str = "config/config.yaml", checkpoint: str | None = None, device: str | None = None) -> dict:
    config = load_config(config_path)
    paths = resolve_paths(config)
    data_info = resolve_data_info(config, paths)
    train_cfg = config["train_params"]
    eval_cfg = config.get("evaluation", {})
    category = config.get("experiment", {}).get("category", "bottle")

    checkpoint_path = checkpoint or default_checkpoint_from_config(config)
    threshold = float(eval_cfg.get("auroc_threshold", 0.85))
    output_dir = os.path.join(paths["output_dir"], "evaluation")
    device_obj = _resolve_device(device)

    dm = MVTecDataModule(
        root_dir=paths["data_dir"],
        category=category,
        batch_size=train_cfg["batch_size"],
        img_size=train_cfg["img_size"],
    )
    dm.setup("test")

    model = AnomalyAE.load_from_checkpoint(checkpoint_path, map_location=device_obj)
    scores, labels = score_autoencoder(model, dm.val_dataloader(), device_obj)

    if len(set(labels)) < 2:
        raise ValueError("AUROC requires both normal and anomalous samples in the test labels.")

    auroc = float(roc_auc_score(labels, scores))
    passed = auroc >= threshold

    scores_path = save_scores_csv(scores, labels, output_dir)
    report = {
        "checkpoint": checkpoint_path,
        "category": category,
        "data": data_info,
        "num_samples": len(labels),
        "num_normal": int(sum(label == 0 for label in labels)),
        "num_anomaly": int(sum(label == 1 for label in labels)),
        "auroc": auroc,
        "auroc_threshold": threshold,
        "passed": passed,
    }
    report_path = save_report(report, output_dir)

    mlflow.set_tracking_uri(paths["log_dir"])
    mlflow.set_experiment("quali-trace-evaluation")

    with mlflow.start_run(run_name=f"evaluate-{category}"):
        mlflow.log_params(
            {
                "category": category,
                "checkpoint": checkpoint_path,
                "data_name": data_info["name"],
                "data_version": data_info["version"],
                "data_source": data_info["source"],
                "data_dir": data_info["data_dir"],
                "gate_auroc_threshold": threshold,
            }
        )
        mlflow.log_metric("test_auroc", auroc)
        mlflow.log_metric("gate_passed", int(passed))
        mlflow.log_artifact(report_path, artifact_path="evaluation")
        mlflow.log_artifact(scores_path, artifact_path="evaluation")
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")

        if passed and eval_cfg.get("register_on_pass", True):
            register_candidate_model(
                model=model,
                registry_name=eval_cfg.get("model_registry_name", "QualiTrace-AnomalyAE"),
                candidate_tag=eval_cfg.get("candidate_tag", "Candidate"),
            )

    if not passed:
        raise RuntimeError(
            f"Evaluation gate failed: AUROC={auroc:.4f} below threshold={threshold:.4f}. "
            f"Report: {report_path}"
        )

    print(f"Evaluation gate passed: AUROC={auroc:.4f} >= threshold={threshold:.4f}")
    print(f"Report saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AE anomaly detection and register passing models.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Defaults to output_dir/last.ckpt")
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    evaluate(config_path=args.config, checkpoint=args.checkpoint, device=args.device)
