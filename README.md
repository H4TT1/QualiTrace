# QualiTrace
End-to-End MLOps Pipeline for Industrial Defect Detection

## Generic experiment launcher
`src/train.py` is now a generic launcher. It picks the experiment runner from:

- `model.family`
- `model.architecture`

Current built-ins:

- `ae/conv_ae` (trainable autoencoder)
- `vlm/clip` (zero-shot CLIP anomaly scoring)

## Config-driven experiments
Use `config/config.yaml` to change experiments from Kaggle without editing code:

- `model.family`: `ae` or `vlm`
- `model.architecture`: `conv_ae` or `clip`
- `model.variant` for AE: `lite`, `base`, `deep`
- `train_params.loss_type` for AE: `mse`, `ssim`, `combined`
- `model.backbone` for VLM (for example `openai/clip-vit-base-patch32`)
- `vlm_params.normal_prompt` / `vlm_params.anomaly_prompt`
- `experiment.category` (`bottle`, `capsule`, ...)

## Run
```bash
python3 src/train.py
```

## How to add a new model/experiment
1. Add a model builder in `src/models/<your_model>.py` and register it with:
```python
@register_model("your_family", "your_arch")
def build_your_model(model_cfg, train_cfg):
    ...
```
2. Add an experiment runner in `src/experiments/<your_runner>.py` and register it with:
```python
@register_runner("your_family", "your_arch")
def run_your_experiment(config, paths):
    ...
```
3. Ensure your runner module is imported from `src/experiments/__init__.py`.
4. Set matching values in `config/config.yaml`.
