import json
import os

import mlflow
import torch
from sklearn.metrics import roc_auc_score

from config_utils import resolve_data_info
from data_loader import MVTecDataModule
from data_validation import raise_if_validation_failed, save_validation_report, validate_mvtec_dataset
from models import build_model
from .registry import register_runner


def _resolve_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _score_dataloader(model, dataloader, device):
    scores = []
    labels = []

    for x, y in dataloader:
        batch_scores = model.score(x, device=device)
        scores.extend(batch_scores.detach().cpu().tolist())
        labels.extend(y.detach().cpu().int().tolist())

    return scores, labels


@register_runner("embedding", "patchcore")
def run_patchcore_experiment(config: dict, paths: dict):
    train_cfg = config["train_params"]
    model_cfg = config.get("model", {})
    exp_cfg = config.get("experiment", {})
    validation_cfg = config.get("data_validation", {})
    data_info = resolve_data_info(config, paths)

    category = exp_cfg.get("category", "bottle")
    device = _resolve_device(model_cfg.get("device"))
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_tracking_uri(paths["log_dir"])
    mlflow.set_experiment("quali-trace-patchcore")

    with mlflow.start_run(run_name=f"patchcore-{category}"):
        mlflow.log_params(
            {
                "model_family": "embedding",
                "model_architecture": "patchcore",
                "backbone": model_cfg.get("backbone", "wide_resnet50_2"),
                "pretrained": model_cfg.get("pretrained", True),
                "layers": ",".join(model_cfg.get("layers", ["layer2", "layer3"])),
                "category": category,
                "data_name": data_info["name"],
                "data_version": data_info["version"],
                "data_source": data_info["source"],
                "data_dir": data_info["data_dir"],
            }
        )

        if validation_cfg.get("enabled", True):
            validation_report = validate_mvtec_dataset(
                data_dir=paths["data_dir"],
                category=category,
                expected_channels=validation_cfg.get("expected_channels", 3),
                min_train_test_ratio=validation_cfg.get("min_train_test_ratio"),
                max_train_test_ratio=validation_cfg.get("max_train_test_ratio"),
            )
            report_path = save_validation_report(validation_report, output_dir)
            mlflow.log_artifact(report_path, artifact_path="data_validation")

            if validation_cfg.get("fail_on_error", True):
                raise_if_validation_failed(validation_report)

        dm = MVTecDataModule(
            root_dir=paths["data_dir"],
            category=category,
            batch_size=train_cfg["batch_size"],
            img_size=train_cfg["img_size"],
        )
        dm.setup()

        model = build_model(model_cfg, train_cfg)
        memory_bank = model.fit_memory_bank(dm.train_dataloader(), device=device)
        scores, labels = _score_dataloader(model, dm.val_dataloader(), device=device)

        if len(set(labels)) < 2:
            raise ValueError("AUROC requires both normal and anomalous samples in the test labels.")

        auroc = float(roc_auc_score(labels, scores))
        checkpoint_path = os.path.join(output_dir, "last.pt")
        model.save(
            checkpoint_path,
            metadata={
                "category": category,
                "img_size": train_cfg["img_size"],
                "model_cfg": model_cfg,
                "train_cfg": train_cfg,
                "data": data_info,
            },
        )

        score_path = os.path.join(output_dir, "patchcore_scores.json")
        with open(score_path, "w") as f:
            json.dump({"scores": scores, "labels": labels, "auroc": auroc}, f, indent=2)

        mlflow.log_metric("val_auroc", auroc)
        mlflow.log_metric("memory_bank_patches", int(memory_bank.shape[0]))
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")
        mlflow.log_artifact(score_path, artifact_path="evaluation")

    print(f"PatchCore training complete. AUROC={auroc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
