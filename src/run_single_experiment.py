import os
import shutil
import sys
import datetime
import time
import logging
import torch
from pytorch_lightning import seed_everything
import numpy as np
import pandas as pd

from data.data_processing import DataFrameProcessor, DataUtils
from data.feature_names import get_features_by_tags
from data import get_dataset
from feature_selection.feature_selection import get_feature_selector
from models.models import get_automl_with_registered_models, needs_cyclical_encoding
from training.logging import FSLogger
from utils.eval import evaluate_on_test_set, data_leakage
from utils.misc import limit_memory, flatten_dict
from config.constants import Constants

# Create logging logger (not to be confused with FSLogger used for experiment artifact and results logging)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Formatter
formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                              datefmt='%d.%m.%y %H:%M:%S')

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Set torch matmul precision and limit memory usage
torch.set_float32_matmul_precision("high")
limit_memory(Constants.MEMORY_LIMIT)

if torch.cuda.is_available():
    # print("Using GPU.")
    accelerator = "gpu"
    device = torch.device("cuda:0")
else:
    # print("Using CPU.")
    accelerator = "cpu"
    device = torch.device("cpu")


def transform_custom_params(params: dict) -> dict:
    """
    Transform custom params so FLAML gets {"domain": value} format,
    except for 'x_scaler' and 'y_transform', which remain unchanged.
    """
    exclude_keys = set()  # {"x_scaler", "y_transform"}
    transformed = {}

    for est_name, est_params in params.items():
        new_params = {}
        for key, value in est_params.items():
            if key in exclude_keys:
                new_params[key] = value
            else:
                new_params[key] = {"domain": value}
        transformed[est_name] = new_params

    return transformed


def validate_experiment_args(args):
    # General required settings:
    if args.name is None:
        raise ValueError("Experiment name must be specified.")
    if args.fs_method not in Constants.FS_METHODS:
        raise ValueError(f"Invalid feature selection method: {args.fs_method}. Must be in {Constants.FS_METHODS}.")
    if args.domain not in Constants.DOMAINS:
        raise ValueError(f"Invalid domain: {args.domain}. Must be in {Constants.DOMAINS}.")
    if args.asset_id is None:
        raise ValueError("Asset ID must be specified.")
    if not args.model in Constants.MODELS:
        raise ValueError(f"Invalid models: {args.model}. Must be in {Constants.MODELS}.")
    if args.features not in Constants.FEATURE_SET_TYPES:
        raise ValueError(f"Invalid feature set type: {args.features}. Must be in {Constants.FEATURE_SET_TYPES}.")
    if args.n_features is None or args.n_features <= 0:
        raise ValueError("Number of features to select must be a positive integer.")

    if args.random_seed is None:
        raise ValueError("Random seed must be specified.")

    if args.fs_method in ["SFS", "CSFS"]:
        if args.hpo_mode not in Constants.HPO_MODES or args.feature_level_hpo_mode not in Constants.HPO_MODES:
            raise ValueError(f"Invalid HPO mode: {args.hpo_mode}. Must be in {Constants.HPO_MODES}.")
        if args.direction not in ["forward", "backward"]:
            raise ValueError(f"Invalid direction: {args.direction}. Must be in ['forward', 'backward'].")
        if args.clustering_method not in Constants.CLUSTERING_METHODS:
            raise ValueError(
                f"Invalid clustering method: {args.clustering_method}. Must be in {Constants.CLUSTERING_METHODS}.")
        if args.clustering_method in ["random", "feature_importance"]:
            if args.group_size is None:
                raise ValueError(f"group_size must be specified when using "
                                 f"{args.clustering_method} clustering method.")
        else:
            if args.group_size is not None:
                raise ValueError(f"group_size should not be specified when using "
                                 f"{args.clustering_method} clustering method.")
    else:
        for v in [args.hpo_mode, args.direction, args.clustering_method, args.group_size]:
            if v is not None:
                raise ValueError(f"{v} should not be specified when not using SFS or CSFS as fs_method.")

    # Confidence interval settings:
    if args.cv:
        if args.bootstrapping:
            raise ValueError("Cannot use both cross-validation and bootstrapping at the same time.")
        if args.n_folds is None:
            raise ValueError("Number of folds must be specified when using cross-validation.")
    if args.bootstrapping:
        if args.n_bootstrap_samples is None:
            raise ValueError("Number of bootstrap samples must be specified when using bootstrapping.")

    # HPO settings:
    if args.hpo_mode != 'off':
        if not ((args.hpo_max_iter is not None) ^ (args.hpo_time_budget is not None)):
            raise ValueError("You must specify either hpo_max_iter or hpo_time_budget, but not both.")
    if args.warm_starts:
        if not ((args.warmup_max_iter is not None) ^ (args.warmup_time_budget is not None)):
            raise ValueError("You must specify either warmup_max_iter or warmup_time_budget, but not both.")


def run_experiment(args):
    validate_experiment_args(args)
    seed_everything(args.random_seed)
    model_name = args.model
    args.metrics = Constants.METRICS

    fslogger = FSLogger(
        args.name,
        model_name,
    )

    if len(os.listdir(fslogger.log_dir)) > 0:
        if args.name == "debug" or args.overwrite:
            logging.warning(f"Experiment with name {args.name} already exists in {fslogger.log_dir}. Overwriting.")
            shutil.rmtree(fslogger.log_dir)
            fslogger.log_dir.mkdir(parents=True, exist_ok=True)
        elif args.resume_at_iteration is not None:
            logging.info(f"Resuming experiment {args.name} at iteration {args.resume_at_iteration}.")
        else:
            raise ValueError(f"Experiment with name {args.name} already exists in {fslogger.log_dir}.")

    # Add file handler for console logs
    log_filename = fslogger.log_dir / f'console_{datetime.datetime.now().strftime("%y-%m-%d_%H-%M")}.log'
    log_filename.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    dataset = get_dataset(config=vars(args))
    dataset_df = dataset.get_dataframe()
    feature_tags = dataset.get_feature_tags()

    _datetimes = dataset_df['Timestamp']

    # Only keep features according to the selected feature set type
    if args.features == "forecast_available":
        inclusion_tags = {"forecast_available"}
        exclusion_tags = {"meta", "target", "power_proxy", "system_state"}
    else:
        inclusion_tags = {"forecast_available", "system_state"}
        exclusion_tags = {"meta", "target", "power_proxy"}

    # Handle cyclical encoding requirement
    if needs_cyclical_encoding(model_name):
        inclusion_tags.add("cyclical_encoding")
        exclusion_tags.add("circular")
    else:
        inclusion_tags.add("circular")
        exclusion_tags.add("cyclical_encoding")

    features = get_features_by_tags(
        feature_tags,
        tags_include_any=inclusion_tags,
        tags_exclude=exclusion_tags
    )

    _X = DataFrameProcessor.filter_columns(features)(dataset_df)
    logging.info(
        f"X: Removed following meta, target, ... columns: {dataset_df.columns.difference(_X.columns)}")
    _y = DataFrameProcessor.filter_columns(get_features_by_tags(
        feature_tags,
        tags_include_any={"target"}, ))(dataset_df)

    logging.info("Dataset loaded with shape X: {}, y: {}".format(_X.shape, _y.shape))
    logging.info("Features used: {}".format(_X.columns.tolist()))
    logging.info("Target used: {}".format(_y.columns.tolist()))

    split_indices = DataUtils.get_time_split_indices(_datetimes,
                                                     split_ratio=(1 - args.test_ratio, args.test_ratio),
                                                     gap=pd.to_timedelta(args.gap))
    for split_idx, split_typ in zip(split_indices, ["Train+Validation", "Test"]):
        logging.info(f"{split_typ}: index=[{split_idx[0]}, {split_idx[-1]}]")
        logging.info(f"{split_typ}: Datetime=[{_datetimes.iloc[split_idx[0]]}, {_datetimes.iloc[split_idx[-1]]}]")

    train_val_idx = split_indices[0]
    test_idx = split_indices[1]

    X_train_val = _X.iloc[train_val_idx]
    y_train_val = _y.iloc[train_val_idx]
    datetimes_train_val = _datetimes.iloc[train_val_idx]
    X_test = _X.iloc[test_idx]
    y_test = _y.iloc[test_idx]
    datetimes_test = _datetimes.iloc[test_idx]

    # Make sure no data leakage occurs
    data_leakage_value = data_leakage(
        test_data=datetimes_test.to_frame("Timestamp"),
        train_data=datetimes_train_val.to_frame("Timestamp"),
        datetime_col="Timestamp",
        group_col=None,
    )
    if data_leakage_value != 0:
        raise ValueError(f"Data leakage detected. Data leakage value: {data_leakage_value}")

    if args.bootstrapping:
        test_bootstrap_indices = np.random.randint(low=0,
                                                   high=len(test_idx),
                                                   size=(args.n_bootstrap_samples, len(test_idx)))
    else:
        test_bootstrap_indices = None

    automl_logdir = fslogger.log_dir / "automl" if fslogger.log_dir is not None else None
    automl_logdir.mkdir(parents=True, exist_ok=True)

    # Create AutoML settings from base settings and model-specific settings
    BASE_AUTOML_SETTINGS = {
        'time_budget': args.hpo_time_budget,
        'max_iter': args.hpo_max_iter,
        'early_stop': args.hpo_early_stop,
        'eval_method': 'holdout',
        'split_ratio': 0.2,
        'split_type': 'time',
        'metric': 'mse',  # CustomFLAMLMetrics.median_absolute_error,
        'task': 'regression',
        'verbose': 2,
        'n_jobs': args.n_jobs,
        'train_time_limit': args.hpo_train_time_limit,
        'retrain_full': args.hpo_mode == 'per_feature_set',
    }

    automl_settings = {
        **BASE_AUTOML_SETTINGS,
        **{
            "estimator_list": [model_name],
        }
    }
    custom_params = transform_custom_params(Constants.CUSTOM_PARAMS)
    if custom_params.get(model_name, None) is not None:
        automl_settings['custom_hp'] = {
            model_name: custom_params[model_name],
        }
    automl_ws_settings = automl_settings.copy()
    automl_ws_settings.update(
        {
            'time_budget': args.warmup_time_budget,
            'max_iter': args.warmup_max_iter,
            'early_stop': args.warmup_early_stop,
            'retrain_full': False,  # Not needed for warmup
        }
    )

    # Configuration for FS method
    fs_config = {
        'fs_method': args.fs_method,
        'n_features': args.n_features,
        'estimator_name': model_name,
        'fixed_features': [],
        'direction': args.direction,
        'metrics': Constants.METRICS,
        'aggregation_mode': 'median',
        # cv=cv,
        'val_ratio': 1 / 3,
        'gap': pd.to_timedelta(args.gap),
        'datetimes': datetimes_train_val,
        'bootstrap_sample_size': args.n_bootstrap_samples,
        'n_features_to_select': args.n_features,
        'verbose': 3,
        'hpo_mode': args.hpo_mode,
        'feature_level_hpo_mode': args.feature_level_hpo_mode,
        'automl_settings': automl_settings,
        'automl_ws_settings': automl_ws_settings,
        'automl_log_dir': automl_logdir,
        'warm_starts': args.warm_starts,
        'random_seed': args.random_seed,
        'scoring': 'neg_rmse',
        'fast_mode': args.fast_mode,
        'early_stopping': False,
        'clustering_config': {"clustering_method": args.clustering_method,
                              'group_size': args.group_size,
                              },
        'resume_at_iteration': args.resume_at_iteration,
        'task': dataset.get_task(),
    }

    feature_selector = get_feature_selector(fslogger, fs_config)

    logging.info("Start feature selection process.")
    fs_start_time = time.perf_counter()
    feature_selector.fit(
        X=X_train_val.copy(),
        y=y_train_val.copy().values.ravel(),
    )
    fs_end_time = time.perf_counter()
    fs_runtime = fs_end_time - fs_start_time
    logging.info(
        f"Feature selection finished in {str(datetime.timedelta(seconds=fs_runtime))}. Selected features: {feature_selector.get_feature_names_out()}")

    fslogger.log_metrics({"fs_runtime": fs_runtime})

    # fslogger.save_object("csfs", feature_selector)
    fslogger.save_object("feature_names_out", feature_selector.get_feature_names_out())

    logging.info("Start fitting on test set.")
    test_fit_kwargs = automl_settings.copy()
    test_fit_kwargs['retrain_full'] = True
    if args.warm_starts:
        logging.info("Start warmup")
        warmup_est = get_automl_with_registered_models(models_to_register=[model_name])
        warmup_est.fit(X_train_val.copy().loc[:, feature_selector.get_feature_names_out()],
                       y_train_val.copy().values.ravel(),
                       **automl_ws_settings,
                       )
        logging.info("Finished warmup.")
        test_fit_kwargs['starting_points'] = warmup_est.best_config_per_estimator

    est = get_automl_with_registered_models(models_to_register=[model_name])

    test_fit_start_time = time.perf_counter()
    est.fit(X_train_val.copy().loc[:, feature_selector.get_feature_names_out()],
            y_train_val.copy().values.ravel(),
            **test_fit_kwargs,
            log_file_name=automl_logdir / "testing" if automl_logdir is not None else None,
            )
    test_fit_end_time = time.perf_counter()
    test_fit_duration = test_fit_end_time - test_fit_start_time
    logging.info(f"Test fit finished in {str(datetime.timedelta(seconds=test_fit_duration))}.")
    test_results = dict()
    test_results["duration"] = test_fit_duration
    test_results["n_trials"] = est._search_states[model_name].total_iter
    fslogger.log_metrics(flatten_dict(test_results, parent_key="testing"))

    testing_metrics = evaluate_on_test_set(
        est,
        X_test=X_test.copy().loc[:, feature_selector.get_feature_names_out()],
        y_test=y_test.copy().values.ravel(),
        scoring=Constants.METRICS,
        bootstrap_sample_size=args.n_bootstrap_samples,
        test_set_bootstrap=args.bootstrapping,
    )

    fslogger.log_metrics_array(flatten_dict(testing_metrics, parent_key="testing"))

    fslogger.log_metrics_array(
        {
            "train_val_idx": train_val_idx,
            "test_idx": test_idx,
            "test_bootstrap_indices": test_bootstrap_indices,
        }
    )
    fslogger.save_object("exp_args", args)
    fslogger.save()
    fslogger.finalize("success")
