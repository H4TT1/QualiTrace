import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
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

    # trainer
    trainer = pl.Trainer(
        max_epochs=config['train_params']['epochs'],
        accelerator="auto", # will find gpu on kaggle
        logger=mlf_logger,
        devices=1
    )

    # run
    trainer.fit(model, dm)

if __name__ == "__main__":
    train()