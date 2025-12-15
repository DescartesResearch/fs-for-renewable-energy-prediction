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
    - [Generate report](#generate-report)
    - [Run a custom experiment](#run-a-custom-experiment)
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

DOCKER:
No manual setup needed.

MANUAL LOCAL SETUP:

1. Clone the repository.
2. If necessary, install uv: See the
   official [uv install instructions](https://docs.astral.sh/uv/getting-started/installation/)
3. In the project root directory, run `uv sync` command to create and update the virtual environment `.venv`

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

<details>
<summary><strong>Docker</strong></summary>
Adapt the environment variables in the `.env` file.
</details>

<details>
<summary><strong>Local</strong></summary>

- In the config file `config/constants.py`, set the path to the datasets folder `Paths.DATASETS` correctly.
</details>
The dataset path should contain the two subfolders `edp` and `PVOD` with the respective downloaded (and extracted)
datasets. Folder can be empty if you only want to use the provided logs from the experiments in the paper.
The logs path is where the experiment logs will be saved. You can also use the provided logs in the following Zenodo
repository, if you want to skip running the experiments yourself:
The report path is where the generated figures, tables, and numbers from the paper will be saved.

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
uv run ./src/main.py
            --run_multiple_experiments
            --n_jobs 4
            --first_cpu_idx 0
            --last_cpu_idx 15
            --script_path "./src/paper_experiments.sh"
            --first_experiment_idx 0
            --last_experiment_idx 559
```

</details>

The `n_jobs` variable defines how many CPU cores will be used per experiment. All CPU cores between `first_cpu_idx` and
`last_cpu_idx` (incl.) will be used to run experiments in parallel.
In the example above, CPU cores 0 to 15 (16 cores) will be used, with 4 cores per experiment, resulting in 4 experiments
running in parallel.
The `paper_experiments.sh` script contains the definition of all experiments (in total: 560) conducted for the paper.

### Generate report

After the experiments have finished and all experiment logs are saved to the logs directory (defined above), you can
generate the report (figures, tables, numbers).
Alternatively, instead of re-running the experiments yourself, you can use the provided logs in the following Zenodo
repository:

The report generation script can be executed as follows:

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose up generate_report
```

</details>


<details>
<summary><strong>Local</strong></summary>

```bash
uv run ./src/main.py --generate_report
```

</details>

### Run a custom experiment

You can also run custom experiments by specifying the desired parameters directly via CLI arguments.

Example command to run a CSFS experiment with MLP model on wind turbine dataset, and FI-based clustering:

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose run --rm main
--run_single_experiment
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

</details>

<details>
<summary><strong>Local</strong></summary>

```bash
uv run ./src/main.py
--run_single_experiment
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

</details>

## Project layout

- `config/` — constants and path helpers (`config/constants.py`)
- `data/` — dataset loading and preprocessing utilities
- `feature_selection/` — CSFS, SFS and filter-based feature selection implementations
- `models/` — model wrappers and FLAML estimator integrations with HPO search spaces
- `training/` — logging helpers and experiment runners
- `utils/` — evaluation, plotting and misc. helpers
- `main.py` — CLI entrypoint to re-run experiments and generate report
- `run_single_experiment.py` — Runner for single custom experiments
- `run_multiple_experiments.py` - Orchestrator for multiple experiments in parallel

## CSFS algorithm pseudocode

![](./CSFS%20Pseudocode.png "Pseudocode for the Cluster-based Sequential Feature Selection Algorithm")

## If you need help...

...please open an issue on the GitHub repository. Include details about your environment, steps to reproduce, and any
error messages.
