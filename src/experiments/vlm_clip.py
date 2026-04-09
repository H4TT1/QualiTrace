import os
from typing import List

import mlflow
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor

from data_loader import MVTecDataset
from .registry import register_runner


class CLIPAnomalyScorer:
    def __init__(self, backbone: str, device: str):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(backbone)
        self.model = CLIPModel.from_pretrained(backbone).to(device).eval()

    def score_image(self, image: Image.Image, prompts: List[str]) -> float:
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)[0]

        # Prompt order: [normal, anomaly]
        return float(probs[1].item())


@register_runner("vlm", "clip")
def run_vlm_clip_experiment(config: dict, paths: dict):
    model_cfg = config.get("model", {})
    exp_cfg = config.get("experiment", {})
    vlm_cfg = config.get("vlm_params", {})

    category = exp_cfg.get("category", "bottle")
    backbone = model_cfg.get("backbone", "openai/clip-vit-base-patch32")
    normal_prompt = vlm_cfg.get("normal_prompt", "a photo of a normal {category}")
    anomaly_prompt = vlm_cfg.get("anomaly_prompt", "a photo of a defective {category}")

    formatted_prompts = [
        normal_prompt.format(category=category),
        anomaly_prompt.format(category=category),
    ]

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = vlm_cfg.get("device", default_device)

    test_ds = MVTecDataset(
        root_dir=paths["data_dir"],
        category=category,
        transform=None,
        is_train=False,
    )

    scorer = CLIPAnomalyScorer(backbone=backbone, device=device)

    labels = []
    scores = []

    for image_path, label in zip(test_ds.image_paths, test_ds.labels):
        image = Image.open(image_path).convert("RGB")
        score = scorer.score_image(image, formatted_prompts)

        labels.append(label)
        scores.append(score)

    if len(set(labels)) < 2:
        raise ValueError("AUC requires at least two classes in labels.")

    auc = roc_auc_score(labels, scores)
    print(f"VLM backbone: {backbone}")
    print(f"Category: {category}")
    print(f"Prompts: {formatted_prompts}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Num samples: {len(labels)}")

    if mlflow.active_run():
        mlflow.log_param("model_family", "vlm")
        mlflow.log_param("model_architecture", "clip")
        mlflow.log_param("vlm_backbone", backbone)
        mlflow.log_param("category", category)
        mlflow.log_param("normal_prompt", formatted_prompts[0])
        mlflow.log_param("anomaly_prompt", formatted_prompts[1])
        mlflow.log_metric("vlm_roc_auc", float(auc))

    out_dir = paths["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"vlm_scores_{category}.npy"), np.array(scores, dtype=np.float32))
    np.save(os.path.join(out_dir, f"vlm_labels_{category}.npy"), np.array(labels, dtype=np.int32))
