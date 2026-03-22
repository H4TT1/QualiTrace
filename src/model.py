import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class AnomalyAE(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
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
            nn.Sigmoid() # Keep pixel values in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch # MVTec loader returns (img, label)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        val_loss = F.mse_loss(x_hat, x)
        
        # Generate Anomaly Score (MSE per image) / another similarity metric ? 
        score = torch.mean((x_hat - x)**2, dim=(1, 2, 3))
        
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss, "scores": score, "labels": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)