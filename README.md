---
header-includes:
  - \usepackage[linesnumbered,ruled,vlined]{algorithm2e}
---

# CSFS — Cluster-based Sequential Feature Selection for Renewable Energy Prediction

A research-focused Python toolkit for preprocessing, feature selection and AutoML-driven model training for renewable
energy forecasting (WT and PV power).

Table of contents

- [Project Description](#project-description)
- [Why it is useful](#why-it-is-useful)
- [Getting started](#getting-started)
    - [Requirements](#requirements)
    - [Necessary steps](#necessary-steps)
- [Usage examples](#usage-examples)
    - [Run a single experiment](#run-a-single-experiment)
    - [Orchestrate multiple experiments](#orchestrate-multiple-experiments)
- [Project layout](#project-layout)
- [CSFS algorithm pseudocode](#csfs-algorithm-pseudocode)
- [If you need help...](#if-you-need-help)

## Project Description

This project provides the implementation of Cluster-based Sequential Feature Selection (CSFS) and supporting utilities
to build reproducible experiments for renewable energy forecasting. It integrates dataset utilities, pre-processing,
feature selection algorithms (SFS, CSFS and simple filter methods), and AutoML-backed model selection (via FLAML) with
light wrappers for common model types (MLP, GP, LightGBM, XGBoost, RandomForest).

Primary capabilities

- Load datasets and standardize preprocessing pipelines
- Extract feature sets by tag (forecast-available, digital-twin, cyclical/circular, etc.)
- Run SFS and CSFS feature selection with optional HPO per step
- Use FLAML AutoML to train and tune models with custom estimator wrappers
- Log experiments to a filesystem logger

## Why it is useful

- Research-ready: Designed for reproducible experiments and logging of results/artifacts
- Flexible: Supports multiple domains (wind and pv energy) and model types
- Scalable experiments: HPO (optionally with warm-starting) and parallel runs
- Ready-to-extend: Extensible by adding new datasets, models, or feature selection methods

## Getting started

### Requirements

- Python version 3.11
- The project metadata is defined in `pyproject.toml`. It is recommended to use a virtual environment.
- The lockfile `uv.lock` ensures cross-platform compatibility of dependencies. We recommend using
  the [uv package manager](https://docs.astral.sh/uv/).

### Necessary steps

#### 1. Set up project and dependencies (using uv)

1. Clone the repository.
2. If necessary, install uv: See the
   official [uv install instructions](https://docs.astral.sh/uv/getting-started/installation/)
3. In the project root directory, run `uv sync` command to create and update the virtual environment `.venv`

#### 2. Download datasets

- The project itself does not include the datasets, but they are publicly available:
    - PVOD dataset: https://www.doi.org/10.11922/sciencedb.01094
        - Needs to be downloaded and extracted.
    - EDP dataset: https://www.edp.com/en/innovation/data
        - The EDP OpenData website offers multiple several datasets. We require to download the following:
            - "Wind Turbine SCADA Signals 2016"
            - "On-site MetMast SCADA 2016"
            - "SCADA Signals of the Wind Turb" (2017 SCADA Signals)
            - "SCADA Data of Meteorological Mast on Site 2017"
            - "Logbook of Historical Failures" (Failure Logbook 2016)
            - "Record of failures history from 2017"
        - All Excel files need to be placed in a single folder.

#### 3. Configuration

- In the config file `config/constants.py`, set the dataset paths accordingly:
    - `Paths.PVOD_DATASET_PATH` for the PVOD dataset
    - `Paths.EDP_DATASET_PATH` for the EDP dataset
- The project expects datasets under `datasets/` by default (see `config/constants.Paths`). Example dataset subfolders:
  `datasets/pvod`, `datasets/edp_dataset`.
- Neptune logging is disabled by default. If you want Neptune, set `Constants.USE_NEPTUNE_LOGGER = True` and provide
  credentials (see `config/constants.py`). This is an experimental feature, so use at your own risk.

## Usage examples

### Run a single experiment

The `main.py` script provides a comprehensive CLI. Important flags:

- `--name`: experiment name (required)
- `--domain`: `wind` or `pv` (required)
- `--asset_id`: dataset-specific asset/station id (required)
- `--model`: one of the models in `config/constants.py` (mlp, lgbm, gp, xgboost, rf)
- `--features`: `forecast_available` or `digital_twin`
- `--fs_method`: feature selection method (SFS, CSFS, mutual_info, f_value, RF_FI)
- `--n_features`: number of features to select
- `--random_seed`: random seed (required)

See `uv run .src/main.py -h` for the full list of options.

Example command to run a CSFS experiment with MLP model on wind turbine dataset, and FI-based clustering:

```bash
uv run ./src/main.py
--name wind-T11_mlp_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set
--hpo_train_time_limit 60
--hpo_max_iter 25
--hpo_early_stop
--warm_starts
--warmup_max_iter 25
--warmup_early_stop
--bootstrapping
--n_bootstrap_samples 100
--n_jobs 2
--random_seed 27
--domain wind
--asset_id T11
--model mlp
--features digital_twin
--n_features 2
--hpo_mode per_feature_set
--fs_method CSFS
--fast_mode
--direction backward
--clustering_method feature_importance
--group_size 3
```

### Orchestrate multiple experiments

The `experiment_runner.py` script can be used to run multiple experiments in parallel.
The cpu cores to be used, cpu cores per experiment, and experiment ids are configurable in the script.

## Project layout

- `config/` — constants and path helpers (`config/constants.py`)
- `data/` — dataset loading and preprocessing utilities
- `feature_selection/` — CSFS, SFS and filter-based feature selection implementations
- `models/` — model wrappers and FLAML estimator integrations with HPO search spaces
- `training/` — logging helpers and experiment runners
- `utils/` — evaluation, plotting and misc. helpers
- `main.py` — CLI entrypoint to run single experiments end-to-end
- `experiment_runner.py` — small orchestration helper for parallel experiment execution
- `run_experiments.sh` - definition of all experiments conducted for the paper

## CSFS algorithm pseudocode

![](./CSFS%20Pseudocode.png "Pseudocode for the Cluster-based Sequential Feature Selection Algorithm")

## If you need help...

...please open an issue on the GitHub repository. Include details about your environment, steps to reproduce, and any
error messages.
