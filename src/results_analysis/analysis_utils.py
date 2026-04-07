from typing import Literal

import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm.notebook import tqdm
import warnings

from training.logging import FSLogger
from utils import aggregate
from experiments import (
    MatrixGenerator,
    get_nested_value,
)


class AnalysisUtils:
    @staticmethod
    def get_fitted_models_count(logger: FSLogger):
        validation_results = logger.fetch_all_values_from_disk("validation")
        if validation_results is None or len(validation_results) == 0:
            return None
        fit_count = 1  # first baseline fit with all features
        for iteration, iteration_results in validation_results.items():
            assert "baseline" in iteration_results
            fit_count += (
                len(iteration_results) - 1
            )  # don't count baseline, because it's just copied from the last iteration
        return fit_count

    @staticmethod
    def get_explanation(logger: FSLogger):
        selected_clusters = logger.fetch_values("selected_cluster_id")
        if selected_clusters is None:
            return None
        elif not hasattr(selected_clusters, "__len__"):
            selected_clusters = [selected_clusters]
        final_clusters = logger.fetch_object("final_clusters")
        fast_stop_iterations = logger.fetch_values("fast_stop_iteration")
        if isinstance(fast_stop_iterations, int):
            fast_stop_iterations = [fast_stop_iterations]
        fast_stop_iterations = (
            [] if fast_stop_iterations is None else fast_stop_iterations
        )
        feature_level_iterations = logger.fetch_values("feature_level_iteration")
        if isinstance(feature_level_iterations, int):
            feature_level_iterations = [feature_level_iterations]
        feature_level_iterations = (
            [] if feature_level_iterations is None else feature_level_iterations
        )
        feature_level_none_non_inferior_iterations = logger.fetch_values(
            "feature_level_none_non-inferior_iteration"
        )
        if isinstance(feature_level_none_non_inferior_iterations, int):
            feature_level_none_non_inferior_iterations = [
                feature_level_none_non_inferior_iterations
            ]
        feature_level_none_non_inferior_iterations = (
            []
            if feature_level_none_non_inferior_iterations is None
            else feature_level_none_non_inferior_iterations
        )
        string = ""
        for iteration, cluster_id in enumerate(selected_clusters):
            string += f"Iteration {iteration} | Cluster (id={cluster_id}): {final_clusters[cluster_id]}"
            baseline_res = logger.fetch_values(
                f"validation/{iteration}/baseline/neg_rmse"
            )
            if baseline_res is None:
                raise RuntimeError(
                    f"[{logger.name}] Cannot fetch baseline results for iteration {iteration}"
                )
            if iteration in fast_stop_iterations:
                string += f"\nReason: Passed non-inferiority test -> stopped early."
            elif iteration in feature_level_none_non_inferior_iterations:
                string += (
                    f"\nReason: Best median score. Didn't pass non-inferiority test."
                )
            elif iteration in feature_level_iterations:
                string += f"\nReason: Passed non-inferiority test -> stopped early."
            for cluster_id, results_dict in logger.fetch_all_values_from_disk(
                f"validation/{iteration}"
            ).items():
                # print(results_dict)

                rmse = np.median(results_dict["rmse"])
                diff = results_dict["neg_rmse"] - baseline_res
                diff_median = np.median(diff)
                diff_lower_bound = np.quantile(diff, q=0.05)
                diff_upper_bound = np.quantile(diff, q=0.95)
                string += f"\nCluster ID: {cluster_id} | RMSE: {rmse} | Difference Median (P05-P95) = {diff_median:.3f} ({diff_lower_bound:.3f} - {diff_upper_bound:.3f})"
            string += "\n\n"
        return string

    @staticmethod
    def get_feature_level_info(
        logger: FSLogger,
        information: Literal[
            "total_iterations", "safe_removals", "fallbacks", "total_features_evaluated"
        ],
    ):
        if information == "total_iterations":
            iterations = logger.fetch_values("feature_level_iteration")
            if hasattr(iterations, "__len__"):
                return len(iterations)
            else:
                return 0
        elif information == "safe_removals":
            total_iterations = logger.fetch_values("feature_level_iteration")
            if total_iterations is None:
                return 0
            else:
                non_safe_removals = logger.fetch_values(
                    "feature_level_none_non-inferior_iteration"
                )
                if non_safe_removals is None:
                    return len(total_iterations)
        elif information == "fallbacks":
            non_safe_removals = logger.fetch_values(
                "feature_level_none_non-inferior_iteration"
            )
            if non_safe_removals is None:
                return 0
            elif hasattr(non_safe_removals, "__len__"):
                return len(non_safe_removals)
            else:
                return 1
        elif information == "total_features_evaluated":
            evaluated_features = logger.fetch_values("feature_level_evaluated_features")
            if hasattr(evaluated_features, "__len__"):
                return sum(evaluated_features)
            else:
                return 0
        else:
            raise ValueError(
                f"Invalid requested feature-level information: {information}"
            )

    @staticmethod
    def get_fast_stops(logger: FSLogger):
        fast_stop_iterations = logger.fetch_values("fast_stop_iteration")
        if fast_stop_iterations is None:
            number_of_fast_stops = 0
        elif hasattr(fast_stop_iterations, "__len__"):
            number_of_fast_stops = len(fast_stop_iterations)
        else:
            number_of_fast_stops = 1

        return number_of_fast_stops

    @staticmethod
    def get_total_iterations(logger: FSLogger):
        selected_clusters = logger.fetch_values("selected_cluster_id")
        if selected_clusters is None:
            return None
        elif not hasattr(selected_clusters, "__len__"):
            selected_clusters = [selected_clusters]
        return len(selected_clusters)


class DataExtractor:
    ARGS_COLS = [
        "feature_selection.clustering.method",
        "dataset.domain",
        "feature_selection.method",
        "feature_selection.n_features",
        "feature_selection.direction",
        "model.name",
        "name",
        "dataset.type",
    ]
    TESTING_SERIES_COLS = [
        "amde",
        "ame",
        "mae",
        "mdae",
        "mde",
        "me",
        "n_trials",
        "neg_rmse",
        "r2",
        "rmse",
    ]
    VALUE_COLS = {"fs_runtime": "fs_runtime"}
    OBJECT_COLS = {"selected_features": "feature_names_out"}
    COMPUTATION_COLS = {
        "fitted_models": AnalysisUtils.get_fitted_models_count,
        "explanation": AnalysisUtils.get_explanation,
        "feature_level_entries": lambda logger: AnalysisUtils.get_feature_level_info(
            logger, "total_iterations"
        ),
        "fallbacks": lambda logger: AnalysisUtils.get_feature_level_info(
            logger, "fallbacks"
        ),
        "feature_level_total_evaluated": lambda logger: (
            AnalysisUtils.get_feature_level_info(logger, "total_features_evaluated")
        ),
        "fast_stops": AnalysisUtils.get_fast_stops,
        "total_iterations": AnalysisUtils.get_total_iterations,
    }

    def __init__(self, matrix_filename: str, debug: bool = False):
        self.matrix_generator = MatrixGenerator()
        self.experiments = self.matrix_generator.generate_experiments(
            self.matrix_generator.load_matrix(matrix_filename)
        )
        if debug:
            self.experiments = self.experiments[:20]
        self.exp_names = list(map(lambda exp: exp["name"], self.experiments))

    def _load(self, column: str, logger: FSLogger, aggregation_method: str = None):
        if column in self.TESTING_SERIES_COLS:
            values = logger.fetch_values(f"testing/{column}")
            return aggregate(aggregation_mode=aggregation_method, arr=values)
        elif column in self.VALUE_COLS:
            return logger.fetch_values(self.VALUE_COLS[column])
            if len(values) != 1:
                raise ValueError(f"Value column {column} has {len(values)} values")
            return values[0]
        elif column in self.OBJECT_COLS:
            return logger.fetch_object(self.OBJECT_COLS[column])
        elif column in self.COMPUTATION_COLS:
            return self.COMPUTATION_COLS[column](logger)
        else:
            raise NotImplementedError(f"Invalid column {column}")

    def get_dataframe(
        self, columns: list[str], aggregation_method: str
    ) -> pd.DataFrame:
        res = defaultdict(list)
        for exp_name in tqdm(self.exp_names):
            # model = parse_model(exp_name)
            fslogger = FSLogger(exp_name)
            if fslogger.fetch_string("status") != "success":
                warnings.warn(f"Could not fetch {exp_name}")
                continue
            exp_args: dict = OmegaConf.to_container(
                fslogger.fetch_object("exp_args"), resolve=True
            )
            # print(exp_args)
            for column in columns:
                if column in self.ARGS_COLS:
                    res[column].append(get_nested_value(exp_args, column, key_sep="."))
                else:
                    res[column].append(self._load(column, fslogger, aggregation_method))
        return pd.DataFrame.from_dict(res)
