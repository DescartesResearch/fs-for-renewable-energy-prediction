import warnings
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

from training.logging import FSLogger
from utils.eval import aggregate

rename_fs_method = {
    "CSFS feature_importance": "CSFS\n(FI)",
    "CSFS correlation": "CSFS\n(corr.)",
    "CSFS random": "CSFS\n(rnd.)",
    "CSFS singletons": "SFS",
    "mutual_info": "MI",
    "f_value": "F-value",
    "RF_FI": "RF FI",
}

def exp_config_to_name_and_args(
        ds,
        asset_id,
        model,
        n_features,
        feature_set,
        fs_method: Literal["CSFS", "mutual_info", "f_value", "RF_FI"],
        clustering_method,
        gs,
        hpo_mode,
):
    exp_name = f"{ds}-{asset_id}_{model}_n-{n_features}_{feature_set}"
    exp_args = f"--domain {ds} --asset_id {asset_id} --model {model} --features {feature_set} --n_features {n_features} --fs_method {fs_method}"
    if fs_method == "CSFS":
        gs_str = f"-gs{gs}" if gs is not None else ""
        gs_param = f" --group_size {gs}" if gs is not None else ""
        exp_name += f"_csfs-{clustering_method}{gs_str}_{hpo_mode}"
        exp_args += f"--hpo_mode {hpo_mode} --fast_mode --direction backward --clustering_method {clustering_method}{gs_param}"
    else:
        assert clustering_method is None
        assert gs is None
        assert hpo_mode is None
        exp_name += f"_{fs_method}"

    return exp_name, exp_args


def requires_clustering(fs_method: Literal["CSFS", "mutual_info", "f_value", "RF_FI"]):
    return fs_method == "CSFS"


def parse_model(exp_name):
    parts = exp_name.split("_")

    # At minimum: ds-asset, model, n-#, feature_set, fs_method...
    if len(parts) < 5:
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    # --- Parse "{ds}-{asset_id}"
    ds_asset = parts[0]
    if "-" not in ds_asset:
        raise ValueError(f"Could not parse dataset and asset_id from '{ds_asset}'")
    ds, asset_id = ds_asset.split("-", 1)

    # --- Model
    model = parts[1]

    return model


def experiment_generator(
        ds_tuples: list[tuple] = None,
        models: list[str] = None,
        hpo_modes: list[str] = None,
        fs_methods: list[str] = None,
        clustering_method_tuples: list[tuple] = None,
        feature_sets: list[str] = None,
        n_features_list: list[int] = None,
):
    if ds_tuples is None:
        ds_tuples = [("wind", "T11"), ("pv", "01")]
    if models is None:
        models = ["mlp", "xgboost", "lgbm", "rf"]
    if hpo_modes is None:
        hpo_modes = ["per_feature_set"]
    if clustering_method_tuples is None:
        clustering_method_tuples = [("feature_importance", "3"), ("correlation", None), ("random", 3),
                                    ("singletons", None)]
    if fs_methods is None:
        fs_methods = ["CSFS", "mutual_info", "f_value", "RF_FI"]
    if feature_sets is None:
        feature_sets = ["digital_twin", "forecast_available"]
    if n_features_list is None:
        n_features_list = [2, 3, 5, 8, 10]

    for ds, asset_id in ds_tuples:
        for model in models:
            for feature_set in feature_sets:
                for n_features in n_features_list:
                    for fs_method in fs_methods:
                        for hpo_mode in (hpo_modes if requires_clustering(fs_method) else [None]):
                            for clustering_method, gs in (
                                    clustering_method_tuples if requires_clustering(fs_method) else [(None, None)]):
                                exp_name, exp_args = exp_config_to_name_and_args(ds, asset_id, model, n_features,
                                                                                 feature_set, fs_method,
                                                                                 clustering_method, gs, hpo_mode)
                                yield exp_name, exp_args


def _get_fitted_models_count(logger: FSLogger):
    validation_results = logger.fetch_all_values_from_disk("validation")
    if validation_results is None or len(validation_results) == 0:
        return None
    fit_count = 1  # first baseline fit with all features
    for iteration, iteration_results in validation_results.items():
        assert "baseline" in iteration_results
        fit_count += (
                len(iteration_results) - 1)  # don't count baseline, because it's just copied from the last iteration
    return fit_count


def _get_explanation(logger: FSLogger):
    selected_clusters = logger.fetch_values('selected_cluster_id')
    if selected_clusters is None:
        return None
    final_clusters = logger.fetch_object('final_clusters')
    fast_stop_iterations = logger.fetch_values('fast_stop_iteration')
    if isinstance(fast_stop_iterations, int):
        fast_stop_iterations = [fast_stop_iterations]
    fast_stop_iterations = [] if fast_stop_iterations is None else fast_stop_iterations
    feature_level_iterations = logger.fetch_values('feature_level_iteration')
    if isinstance(feature_level_iterations, int):
        feature_level_iterations = [feature_level_iterations]
    feature_level_iterations = [] if feature_level_iterations is None else feature_level_iterations
    feature_level_none_non_inferior_iterations = logger.fetch_values('feature_level_none_non-inferior_iteration')
    if isinstance(feature_level_none_non_inferior_iterations, int):
        feature_level_none_non_inferior_iterations = [feature_level_none_non_inferior_iterations]
    feature_level_none_non_inferior_iterations = [] if feature_level_none_non_inferior_iterations is None else feature_level_none_non_inferior_iterations
    string = ""
    for iteration, cluster_id in enumerate(selected_clusters):
        string += f"Iteration {iteration} | Cluster (id={cluster_id}): {final_clusters[cluster_id]}"
        baseline_res = logger.fetch_values(f"validation/{iteration}/baseline/neg_rmse")
        if baseline_res is None:
            raise RuntimeError(f"[{logger.name}] Cannot fetch baseline results for iteration {iteration}")
        if iteration in fast_stop_iterations:
            string += f"\nReason: Passed non-inferiority test -> stopped early."
        elif iteration in feature_level_none_non_inferior_iterations:
            string += f"\nReason: Best median score. Didn't pass non-inferiority test."
        elif iteration in feature_level_iterations:
            string += f"\nReason: Passed non-inferiority test -> stopped early."
        for cluster_id, results_dict in logger.fetch_all_values_from_disk(f"validation/{iteration}").items():
            # print(results_dict)

            rmse = np.median(results_dict["rmse"])
            diff = results_dict["neg_rmse"] - baseline_res
            diff_median = np.median(diff)
            diff_lower_bound = np.quantile(diff, q=0.05)
            diff_upper_bound = np.quantile(diff, q=0.95)
            string += f"\nCluster ID: {cluster_id} | RMSE: {rmse} | Difference Median (P05-P95) = {diff_median:.3f} ({diff_lower_bound:.3f} - {diff_upper_bound:.3f})"
        string += "\n\n"
    return string


def _get_feature_level_info(logger: FSLogger, information: Literal[
    'total_iterations', 'safe_removals', 'fallbacks', 'total_features_evaluated']):
    if information == 'total_iterations':
        iterations = logger.fetch_values("feature_level_iteration")
        if hasattr(iterations, '__len__'):
            return len(iterations)
        else:
            return 0
    elif information == 'safe_removals':
        total_iterations = logger.fetch_values("feature_level_iteration")
        if total_iterations is None:
            return 0
        else:
            non_safe_removals = logger.fetch_values("feature_level_none_non-inferior_iteration")
            if non_safe_removals is None:
                return len(total_iterations)
    elif information == 'fallbacks':
        non_safe_removals = logger.fetch_values("feature_level_none_non-inferior_iteration")
        if non_safe_removals is None:
            return 0
        elif hasattr(non_safe_removals, '__len__'):
            return len(non_safe_removals)
        else:
            return 1
    elif information == 'total_features_evaluated':
        evaluated_features = logger.fetch_values("feature_level_evaluated_features")
        if hasattr(evaluated_features, '__len__'):
            return sum(evaluated_features)
        else:
            return 0
    else:
        raise ValueError(f"Invalid requested feature-level information: {information}")


def _get_fast_stops(logger: FSLogger):
    fast_stop_iterations = logger.fetch_values("fast_stop_iteration")
    if fast_stop_iterations is None:
        number_of_fast_stops = 0
    elif hasattr(fast_stop_iterations, '__len__'):
        number_of_fast_stops = len(fast_stop_iterations)
    else:
        number_of_fast_stops = 1

    return number_of_fast_stops


def _get_total_iterations(logger: FSLogger):
    selected_clusters = logger.fetch_values("selected_cluster_id")
    if selected_clusters is None:
        return None
    else:
        return len(selected_clusters)


class DataExtractor:
    ARGS_COLS = ["clustering_method", "domain", "fs_method", "n_features", "model", "name", "features"]
    TESTING_SERIES_COLS = ["amde", "ame", "mae", "mdae", "mde", "me", "n_trials", "neg_rmse", "r2", "rmse"]
    VALUE_COLS = {
        "fs_runtime": "fs_runtime"
    }
    OBJECT_COLS = {
        "selected_features": "feature_names_out"
    }
    COMPUTATION_COLS = {"fitted_models": _get_fitted_models_count,
                        "explanation": _get_explanation,
                        "feature_level_entries": lambda logger: _get_feature_level_info(logger, 'total_iterations'),
                        "fallbacks": lambda logger: _get_feature_level_info(logger, 'fallbacks'),
                        "feature_level_total_evaluated": lambda logger: _get_feature_level_info(logger,
                                                                                                'total_features_evaluated'),
                        "fast_stops": _get_fast_stops,
                        "total_iterations": _get_total_iterations, }

    def __init__(self, ds_tuples: list[tuple] = None,
                 models: list[str] = None,
                 hpo_modes: list[str] = None,
                 fs_methods: list[str] = None,
                 clustering_method_tuples: list[tuple] = None,
                 feature_sets: list[str] = None,
                 n_features_list: list[int] = None, ):
        self.exp_names = [exp_name for exp_name, _ in
                          experiment_generator(ds_tuples, models, hpo_modes, fs_methods, clustering_method_tuples,
                                               feature_sets, n_features_list)]

    def _load(self, column: str, logger: FSLogger, aggregation_method: str = None):
        if column in self.TESTING_SERIES_COLS:
            values = logger.fetch_values(f"testing/{column}")
            return aggregate(aggregation_mode=aggregation_method, arr=values)
        elif column in self.VALUE_COLS:
            return logger.fetch_values(self.VALUE_COLS[column])
        elif column in self.OBJECT_COLS:
            return logger.fetch_object(self.OBJECT_COLS[column])
        elif column in self.COMPUTATION_COLS:
            return self.COMPUTATION_COLS[column](logger)
        else:
            raise NotImplementedError(f"Invalid column {column}")

    def get_dataframe(self, columns: list[str], aggregation_method: str) -> pd.DataFrame:
        res = defaultdict(list)
        for exp_name in self.exp_names:
            model = parse_model(exp_name)
            fslogger = FSLogger(exp_name, model)
            if fslogger.fetch_string("status") != "success":
                warnings.warn(f"Could not fetch {exp_name}")
                continue
            exp_args = fslogger.fetch_object("exp_args")
            for column in columns:
                if column in self.ARGS_COLS:
                    res[column].append(getattr(exp_args, column))
                else:
                    res[column].append(self._load(column, fslogger, aggregation_method))

        all_results_df = pd.DataFrame.from_dict(res)



        all_results_df.loc[:, "fs_method_plot"] = all_results_df.loc[:, "fs_method"]
        _requires_clustering_mask = all_results_df["fs_method"].apply(requires_clustering)
        all_results_df.loc[_requires_clustering_mask, "fs_method_plot"] += (
                " " + all_results_df.loc[_requires_clustering_mask, "clustering_method"])
        all_results_df.loc[:, "fs_method_plot"] = all_results_df.loc[:, "fs_method_plot"].replace(rename_fs_method)
        for (domain, features), ds_scenario in zip(
                [("wind", "digital_twin"), ("wind", "forecast_available"), ("pv", "digital_twin"),
                 ("pv", "forecast_available")], ["WT-S1", "WT-S2", "PV-S1", "PV-S2"]):
            all_results_df.loc[
                (all_results_df.domain == domain) & (all_results_df.features == features), "ds_plot"] = ds_scenario
        return all_results_df
