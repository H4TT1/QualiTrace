import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .registry import register_model


MODEL_VARIANTS = {
    "lite": [16, 32, 64, 128],
    "base": [32, 64, 128, 256],
    "deep": [32, 64, 128, 256, 512],
}


class AnomalyAE(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        loss_type="combined",
        model_variant="base",
        mse_weight=0.5,
        ssim_weight=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model_variant '{model_variant}'. "
                f"Expected one of: {list(MODEL_VARIANTS.keys())}"
            )

        channels = MODEL_VARIANTS[model_variant]
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.encoder = self._build_encoder(channels)
        self.decoder = self._build_decoder(channels)

    def _uses_ssim(self):
        return self.hparams.loss_type in {"ssim", "combined"}

    @staticmethod
    def _build_encoder(channels):
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.ReLU(),
                ]
            )
            in_ch = out_ch
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(channels):
        layers = []
        rev_channels = list(reversed(channels))

        for idx in range(len(rev_channels) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        rev_channels[idx],
                        rev_channels[idx + 1],
                        3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ]
            )

        layers.extend(
            [
                nn.ConvTranspose2d(
                    rev_channels[-1],
                    3,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Sigmoid(),
            ]
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def _compute_losses(self, x_hat, x):
        mse_loss = F.mse_loss(x_hat, x)
        ssim_val = self.ssim_metric(x_hat, x) if self._uses_ssim() else x_hat.new_tensor(0.0)
        ssim_loss = 1 - ssim_val

        if self.hparams.loss_type == "mse":
            loss = mse_loss
        elif self.hparams.loss_type == "ssim":
            loss = ssim_loss
        elif self.hparams.loss_type == "combined":
            loss = self.hparams.mse_weight * mse_loss + self.hparams.ssim_weight * ssim_loss
        else:
            raise ValueError(
                f"Unknown loss_type '{self.hparams.loss_type}'. Expected: mse, ssim, combined"
            )

        return loss, mse_loss, ssim_val

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)

        loss, mse_loss, ssim_val = self._compute_losses(x_hat, x)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", mse_loss)
        self.log("train_ssim", ssim_val)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        val_loss, val_mse, val_ssim = self._compute_losses(x_hat, x)

        score = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)) + (1 - val_ssim)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mse", val_mse)
        self.log("val_ssim", val_ssim)
        return {"val_loss": val_loss, "scores": score, "labels": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


@register_model("ae", "conv_ae")
def build_conv_ae(model_cfg: dict, train_cfg: dict):
    return AnomalyAE(
        lr=train_cfg["lr"],
        loss_type=train_cfg.get("loss_type", "combined"),
        model_variant=model_cfg.get("variant", "base"),
        mse_weight=train_cfg.get("mse_weight", 0.5),
        ssim_weight=train_cfg.get("ssim_weight", 0.5),
    )
