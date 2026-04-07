# CSFS — Cluster-based Sequential Feature Selection for Renewable Energy Prediction

A research-focused Python toolkit for preprocessing, feature selection and AutoML-driven model training for renewable
energy forecasting (WT and PV power).

Table of contents

- [Project Description](#project-description)
- [Why it is useful](#why-it-is-useful)
- [Getting started](#getting-started)
    - [General Requirements](#general-requirements)
    - [Necessary steps](#necessary-steps)
- [Usage examples](#usage-examples)
    - [Re-run the experiments from the paper](#re-run-the-experiments-from-the-paper)
    - [Run a custom experiment](#run-a-custom-experiment)
    - [Generate report](#generate-report)
- [Source directory layout](#source-directory-layout)
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
- Run SFS and CSFS feature selection with configurable hyperparameter tuning
- Log experiments to a filesystem logger
- Reproducing analysis in the paper

## Why it is useful

- The proposed feature selection method is easy to use (inherits from scikit-learn modules)
- Reproducibility of the claims
- Flexible: multiple domains (wind and pv energy) and model types
- Scalable experiments with parallel runs
- Ready-to-extend: extensible by adding new datasets, models, or feature selection methods

## Getting started

### General Requirements

- Python version 3.11
- The project metadata is defined in `pyproject.toml`.
- The lockfile `uv.lock` ensures cross-platform compatibility of dependencies. We recommend using
  the [uv package manager](https://docs.astral.sh/uv/).

### Necessary steps

We provide two options to set up and run the project:
- Using Docker (recommended for ease of setup and isolation)
- Manual local setup (if you prefer to run directly on your machine)

#### 1. Set up project and dependencies (using uv)

<details>
<summary><strong>Docker</strong></summary>

No manual setup needed.

</details>

<details>
<summary><strong>Local</strong></summary>

1. Clone the repository.
2. If necessary, install uv: See the
   official [uv install instructions](https://docs.astral.sh/uv/getting-started/installation/)
3. In the project root directory, run `uv sync` command to create and update the virtual environment `.venv`

</details>

#### 2. Download datasets

- You can skip this step if you just want to use the provided logs from the experiments in the paper, instead of
  re-running them.
- The project itself does not include the datasets, but they are publicly available:
    - PVOD dataset: https://www.doi.org/10.11922/sciencedb.01094
        - Needs to be downloaded and extracted.
        - The folder must be named `PVOD`
    - EDP dataset: https://www.edp.com/en/innovation/data
        - The EDP OpenData website offers several datasets. We require to download the following (one excel file each):
            - "Wind Turbine SCADA Signals 2016"
            - "On-site MetMast SCADA 2016"
            - "SCADA Signals of the Wind Turb" (2017 SCADA Signals)
            - "SCADA Data of Meteorological Mast on Site 2017"
            - "Logbook of Historical Failures" (Failure Logbook 2016)
            - "Record of failures history from 2017"
        - All Excel files need to be placed in a single folder called `edp`.

#### 3. Configuration

Create an .env file in the project root directory and set the following variables:

- `LOGS_DIR`: This is where the logs and results of the experiments will be saved, including for instance the best hyperparameters, metrics, selected features, ...
- `DATASETS_DIR`: The dataset path should contain the two subfolders `edp` and `PVOD` with the respective downloaded (and extracted) datasets. Folder can be empty if you only want to use the provided logs from the experiments in the paper.
- `EXP_CONFIG_DIR`: Should contain a base configuration `base.yaml` and a sub-directory `matrices` with configurable experiment parameter ranges and constraints.
- `REPORT_DIR`: This is where the reports (as Jupyter Notebooks) will be saved.
- `LITERATURE_STUDY_DIR`: Contains the csv-files with the literature study.

An example .env file is provided: `.env.example`

## Usage examples

### Re-run the experiments from the paper

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose up run_paper_experiments
```

</details>

<details>
<summary><strong>Local</strong></summary>

```bash
uv run ./src/run_paper_experiments.py run paper_grid.yaml
```

</details>

This will launch all experiments defined in `paper_grid.yaml` in `${EXP_CONFIG_PATH}/matrices`.
The base configuration is configured in `${EXP_CONFIG_PATH}/base.yaml`.
The logs and results will be saved to `${LOGS_DIR}`. You can also download the original logs from [Figshare](https://figshare.com/s/43f9aebe17725b29eeed).


### Run a custom experiment

You can also run custom experiments by specifying the desired configuration in `${EXP_CONFIG_PATH}/custom_config.yaml`.

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose up --rm main --config_name custom_config
```

</details>

<details>
<summary><strong>Local</strong></summary>

```bash
uv run ./src/main.py --config_name custom_config
```

</details>

### Generate report
Requires that the experiments finished and the resulting logs are stored in `${LOGS_DIR}`. Alternatively, you can download the original logs from our [Figshare repository](https://figshare.com/s/43f9aebe17725b29eeed).

The two Jupyter Notebooks and a pickled results pandas dataframe will be saved to `${REPORT_DIR}`. The first notebook takes only a few seconds, while the second notebook takes about 4 minutes on a MacBook Air M2.

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose up generate_report
```

</details>

<details>
<summary><strong>Local</strong></summary>

```bash
uv run ./src/evaluate_notebooks.py
```

</details>

## Source directory layout

- `config/` — constants and path helpers (`config/constants.py`)
- `data/` — dataset loading and preprocessing utilities
- `experiments/` - contains scripts for managing and launching experiments
- `feature_selection/` — CSFS, SFS and filter-based feature selection implementations
- `models/` — model wrappers and FLAML estimator integrations with HPO search spaces
- `mplstyles/` - Matplotlib configuration files
- `notebooks/` - Notebooks with literature review and experiment results analysis
- `results_analysis` - Utils used in the notebooks for results analysis
- `training/` — logging helpers for logging to filesystem during training and feature selection
- `utils/` — evaluation, plotting and misc. helpers
- `main.py` — CLI entrypoint to run a single experiment
- `run_paper_experiments.py` — CLI tool for running multiple experiments in parallel, e.g., to reproduce the paper results
- `run_single_experiment.py` - Contains logic for execution of a single experiment (used in `main.py`)

## CSFS algorithm pseudocode

![](./CSFS%20Pseudocode.png "Pseudocode for the Cluster-based Sequential Feature Selection Algorithm")

## If you need help...

...please open an issue on the GitHub repository. Include details about your environment, steps to reproduce, and any
error messages.
