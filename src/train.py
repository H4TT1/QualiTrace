import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import AnomalyAE
from data_loader import MVTecDataModule
import yaml

def train():
    # load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # resolve environment 
    env = os.environ.get("QUALITRACE_ENV", config.get("environment", "local"))
    if env not in config["paths"]:
        raise ValueError(f"Unknown environment '{env}'. Expected one of: {list(config['paths'].keys())}")
    paths = config["paths"][env]

    # setup data
    dm = MVTecDataModule(
        root_dir=paths["data_dir"],
        category="bottle", 
        batch_size=config['train_params']['batch_size']
    )

    # model
    model = AnomalyAE(lr=config['train_params']['lr'])

    # setup mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name="quali-trace-autoencoder",
        tracking_uri=paths["log_dir"],
    )

    # checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["output_dir"],
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=config['train_params']['epochs'],
        accelerator="auto", # will find gpu on kaggle
        logger=mlf_logger,
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=paths["output_dir"],
    )

    # run
    trainer.fit(model, dm)

if __name__ == "__main__":
    train()