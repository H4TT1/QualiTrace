import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights, resnet18, wide_resnet50_2
from torchvision.models._utils import IntermediateLayerGetter

from .registry import register_model


BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT),
}


class PatchCoreModel(nn.Module):
    def __init__(
        self,
        backbone="wide_resnet50_2",
        pretrained=True,
        layers=None,
        max_memory_patches=20000,
        distance_chunk_size=4096,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Expected one of: {list(BACKBONES.keys())}")

        builder, default_weights = BACKBONES[backbone]
        weights = default_weights if pretrained else None
        base_model = builder(weights=weights)
        self.layers = layers or ["layer2", "layer3"]
        self.max_memory_patches = max_memory_patches
        self.distance_chunk_size = distance_chunk_size
        self.memory_bank = None

        return_layers = {layer: layer for layer in self.layers}
        self.feature_extractor = IntermediateLayerGetter(base_model, return_layers=return_layers)
        self.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def extract_patch_features(self, x):
        features = self.feature_extractor(x)
        ordered = [features[layer] for layer in self.layers]
        target_size = ordered[0].shape[-2:]
        resized = [
            feat if feat.shape[-2:] == target_size else F.interpolate(feat, size=target_size, mode="bilinear")
            for feat in ordered
        ]
        embedding = torch.cat(resized, dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def patch_matrix(self, x):
        embedding = self.extract_patch_features(x)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])

    @torch.no_grad()
    def fit_memory_bank(self, dataloader, device):
        self.to(device).eval()
        patches = []

        for x, _ in dataloader:
            patches.append(self.patch_matrix(x.to(device)).cpu())

        memory_bank = torch.cat(patches, dim=0)
        if self.max_memory_patches and memory_bank.shape[0] > self.max_memory_patches:
            indices = torch.randperm(memory_bank.shape[0])[: self.max_memory_patches]
            memory_bank = memory_bank[indices]

        self.memory_bank = memory_bank
        return memory_bank

    @torch.no_grad()
    def score(self, x, device):
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is not fitted or loaded.")

        self.to(device).eval()
        memory_bank = self.memory_bank.to(device)
        patch_features = self.patch_matrix(x.to(device))
        distances = []

        for chunk in patch_features.split(self.distance_chunk_size):
            distances.append(torch.cdist(chunk, memory_bank).min(dim=1).values)

        patch_scores = torch.cat(distances)
        patches_per_image = patch_scores.numel() // x.shape[0]
        return patch_scores.reshape(x.shape[0], patches_per_image).max(dim=1).values

    def save(self, path, metadata=None):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "memory_bank": self.memory_bank,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])
        self.memory_bank = checkpoint["memory_bank"]
        return checkpoint.get("metadata", {})


@register_model("embedding", "patchcore")
def build_patchcore(model_cfg: dict, train_cfg: dict):
    return PatchCoreModel(
        backbone=model_cfg.get("backbone", "wide_resnet50_2"),
        pretrained=model_cfg.get("pretrained", True),
        layers=model_cfg.get("layers", ["layer2", "layer3"]),
        max_memory_patches=model_cfg.get("max_memory_patches", 20000),
        distance_chunk_size=model_cfg.get("distance_chunk_size", 4096),
    )
