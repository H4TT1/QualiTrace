# QualiTrace
End-to-End MLOps Pipeline for Industrial Defect Detection

## Generic experiment launcher
`src/train.py` is now a generic launcher. It picks the experiment runner from:

- `model.family`
- `model.architecture`

Current built-ins:

- `embedding/patchcore` (PatchCore anomaly detection)
- `vlm/clip` (zero-shot CLIP anomaly scoring)

## Config-driven experiments
Use `config/config.yaml` to change experiments from Kaggle without editing code:

- `model.family`: `embedding` or `vlm`
- `model.architecture`: `patchcore` or `clip`
- `model.backbone` for PatchCore: `wide_resnet50_2` or `resnet18`
- `model.layers` for PatchCore feature extraction, for example `["layer2", "layer3"]`
- `model.vlm_backbone` for VLM (for example `openai/clip-vit-base-patch32`)
- `vlm_params.normal_prompt` / `vlm_params.anomaly_prompt`
- `experiment.category` (`bottle`, `capsule`, ...)

## Run
```bash
python3 src/train.py
```

## CI/CD with Kaggle GPU
The project supports a GitHub Actions to Kaggle workflow for full GPU training:

```text
[ Developer ] --- git push ---> [ GitHub Actions ]
                                      | (Kaggle API trigger)
                                      v
                                [ Kaggle GPU Kernel ]
                                 |-- 1. Data Validation (src/data_validation.py)
                                 |-- 2. Pluggable Training (src/train.py)
                                 |-- 3. Automated Evaluation (src/evaluate.py)
                                 `-- 4. Metrics & Artifacts ---> [ DagsHub / MLflow ]
```

There are two workflows:

- `.github/workflows/cml.yml`: CPU smoke test for PRs and pushes. It trains for 1 epoch on synthetic data and posts a CML loss plot.
- `.github/workflows/kaggle-train.yml`: full Kaggle GPU pipeline. It packages the repo, uploads it as a Kaggle dataset, pushes a GPU kernel, follows logs, and downloads outputs.

Required GitHub secrets:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

For DagsHub/MLflow logging and model registry access from Kaggle, configure the matching MLflow credentials in the Kaggle runtime or as Kaggle secrets:

```text
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

The Kaggle workflow runs automatically on pushes to `main` and can also be launched manually from GitHub Actions.

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
