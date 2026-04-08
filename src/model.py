import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

class AnomalyAE(pl.LightningModule):
    def __init__(self, lr=1e-3, loss_type="combined"):
        super().__init__()
        # hparams now tracks lr and loss_type for MLOps comparisons
        self.save_hyperparameters()
        
        # SSIM focuses on texture/structure rather than just pixel intensity
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # Encoder: Downsample to a tight bottleneck
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16
            nn.ReLU()
        )
        
        # Decoder: Upsample back to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        
        # calculate component losses
        mse_loss = F.mse_loss(x_hat, x)
        # SSIM is 1 for perfect match, so we use (1 - SSIM) as a loss
        ssim_val = self.ssim_metric(x_hat, x)
        ssim_loss = 1 - ssim_val
        
        if self.hparams.loss_type == "mse":
            loss = mse_loss
        elif self.hparams.loss_type == "ssim":
            loss = ssim_loss
        else: # "combined"
            loss = 0.5 * mse_loss + 0.5 * ssim_loss
            
        # Log all components for experiment tracking in DagsHub
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", mse_loss)
        self.log("train_ssim", ssim_val)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        
        val_mse = F.mse_loss(x_hat, x)
        val_ssim = self.ssim_metric(x_hat, x)

        score = torch.mean((x_hat - x)**2, dim=(1, 2, 3)) + (1 - self.ssim_metric(x_hat, x))
        
        self.log("val_loss", val_mse, prog_bar=True)
        self.log("val_ssim", val_ssim)
        return {"val_loss": val_mse, "scores": score, "labels": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)