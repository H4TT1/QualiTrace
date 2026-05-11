import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loader import MVTecDataModule
from models import build_model
from config_utils import resolve_data_info
from .registry import register_runner


@register_runner("ae", "conv_ae")
def run_ae_experiment(config: dict, paths: dict):
    train_cfg = config["train_params"]
    model_cfg = config.get("model", {})
    exp_cfg = config.get("experiment", {})
    data_info = resolve_data_info(config, paths)

    category = exp_cfg.get("category", "bottle")
    model_variant = model_cfg.get("variant", "base")
    loss_type = train_cfg.get("loss_type", "combined")

    dm = MVTecDataModule(
        root_dir=paths["data_dir"],
        category=category,
        batch_size=train_cfg["batch_size"],
        img_size=train_cfg["img_size"],
    )

    model = build_model(model_cfg, train_cfg)

    mlf_logger = MLFlowLogger(
        experiment_name="quali-trace-autoencoder",
        tracking_uri=paths["log_dir"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["output_dir"],
        filename=f"{category}-{model_variant}-{loss_type}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="auto",
        logger=mlf_logger,
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=paths["output_dir"],
    )

    for key, value in {
        "data_name": data_info["name"],
        "data_version": data_info["version"],
        "data_source": data_info["source"],
        "data_dir": data_info["data_dir"],
        "category": category,
    }.items():
        mlf_logger.experiment.log_param(mlf_logger.run_id, key, value)

    trainer.fit(model, dm)
