import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from model import AnomalyAE
from data_loader import MVTecDataModule
import yaml

def train():
    # load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # setup data
    dm = MVTecDataModule(
        root_dir=config['paths']['data_dir'],
        category="bottle", 
        batch_size=config['train_params']['batch_size']
    )

    # model
    model = AnomalyAE(lr=config['train_params']['lr'])

    # setup mlflow logger
    mlf_logger = MLFlowLogger(experiment_name="QualiTrace_Reconstruction")

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