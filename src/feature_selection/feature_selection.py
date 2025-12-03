import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Literal, Any, Union
import logging
import shutil

import scipy
from lightning import seed_everything
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import numpy as np
import pandas as pd
import random
from flaml import AutoML
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from data.data_processing import DataFrameProcessor
from models.models import get_automl_with_registered_models, get_sklearn_estim
from training.logging import FSLogger
from utils.eval import evaluate_on_test_set
from data.data_processing import DataUtils

from utils.misc import unflatten_dict, flatten_dict


def get_feature_importance(X: pd.DataFrame, y, automl_settings: dict) -> np.ndarray:
    """
    Get feature importance using a Random Forest model trained via AutoML.
    :param X: Pandas DataFrame with shape (num_samples, num_features)
    :param y: Numpy array with ground truth
    :param automl_settings: Dictionary with AutoML settings. estimator_list will be set to ["rf"] on a copy of this dict.
    :return: Numpy array with feature importances
    """
    automl = get_automl_with_registered_models(models_to_register=["rf"])
    automl_settings = automl_settings.copy()
    automl_settings['estimator_list'] = ["rf"]
    automl_settings['retrain_full'] = True
    automl.fit(X,
               y,
               **automl_settings)
    rf_fi_model = automl.model.estimator
    if 'feature_importances_' not in dir(rf_fi_model):
        raise ValueError("The selected estimator does not have feature_importances_ attribute.")
    feature_importances = rf_fi_model.feature_importances_
    return feature_importances


def get_feature_selector(logger: FSLogger,
                         config: dict) -> BaseEstimator:
    """
    Instantiates a scikit-learn compatible feature selector based on the provided configuration.
    :param logger: A logger instance for collecting logs, metrics and artifacts during feature selection.
    :param config: Configuration dictionary containing feature selection parameters. Must include 'fs_method' key.
    :return: Scikit-learn compatible feature selector instance.
    """
    if config['fs_method'] == 'CSFS':
        return ClusterSequentialFeatureSelector(
            estimator_name=config['estimator_name'],
            fixed_features=config['fixed_features'],
            direction=config['direction'],
            metrics=config['metrics'],
            aggregation_mode=config['aggregation_mode'],
            task=config['task'],
            # cv=cv,
            val_ratio=config['val_ratio'],
            gap=config['gap'],
            datetimes=config['datetimes'],
            bootstrap_sample_size=config["bootstrap_sample_size"],
            n_features_to_select=config['n_features_to_select'],
            verbose=config['verbose'],
            hpo_mode=config['hpo_mode'],
            automl_settings=config['automl_settings'],
            automl_ws_settings=config['automl_ws_settings'],
            automl_log_dir=config['automl_log_dir'],
            logger=logger,
            warm_starts=config['warm_starts'],
            random_seed=config['random_seed'],
            scoring=config['scoring'],
            fast_mode=config['fast_mode'],
            early_stopping=config['early_stopping'],
            clustering_config=config['clustering_config'],
            resume_at_iteration=config['resume_at_iteration'],
        )
    elif config['fs_method'] in ['f_value', 'mutual_info', 'RF_FI']:
        if config['fs_method'] in ['f_value', 'mutual_info']:
            score_func = f_regression if config['fs_method'] == 'f_value' else mutual_info_regression
        else:
            score_func = lambda X, y: get_feature_importance(X, y, automl_settings=config['automl_settings'])
        return SelectKBest(score_func=score_func, k=config['n_features_to_select'])
    elif config['fs_method'] == 'SFS':
        raise NotImplementedError("SFS not yet implemented.")
    else:
        raise NotImplementedError(f"fs_method {config['fs_method']} not implemented")


def dict_k_fold_lists_to_dict(d: dict, k: int) -> dict:
    flat_d = flatten_dict(d)
    for _key in flat_d.keys():
        if isinstance(flat_d[_key], list) and len(flat_d[_key]) == k:
            flat_d[_key] = dict(enumerate(flat_d[_key]))
    return unflatten_dict(flat_d)


def create_feature_clusters(X: pd.DataFrame,
                            clustering_method: Literal['singletons', 'random', 'correlation', 'feature_importance'],
                            corr_threshold: float = 0.9,
                            y: Optional[np.ndarray] = None,
                            group_size: Optional[int] = None,
                            automl_settings: Optional[dict] = None) -> dict[int, list[str]]:
    """
    Performs feature clustering on the columns ℱ in the provided dataframe.
    :param X: Pandas DataFrame with shape (num_samples, num_features)
    :param clustering_method: Clustering method to use. One of 'singletons', 'random', 'correlation', 'feature_importance'
    :param corr_threshold: Correlation threshold for 'correlation' clustering method.
    :param y: Ground truth values. Required if clustering_method == 'feature_importance'.
    :param group_size: Group size for 'random' and 'feature_importance' clustering methods.
    :param automl_settings: AutoML settings for 'feature_importance' clustering method.
    :return: Dictionary mapping cluster IDs to lists of feature names.
    """
    if clustering_method in ['random', 'feature_importance'] and group_size is None:
        raise ValueError(f"random_group_size must be provided if clustering_method == '{clustering_method}'.")
    if clustering_method in ['singletons', 'correlation'] and group_size is not None:
        raise ValueError(f"random_group_size must NOT be provided if clustering_method == '{clustering_method}'.")
    if clustering_method == 'feature_importance':
        if automl_settings is None:
            raise ValueError("automl_settings must be provided if clustering_method == 'feature_importance'.")
        if y is None:
            raise ValueError("y must be provided if clustering_method == 'feature_importance'.")

    features = X.columns.tolist()

    if clustering_method == 'singletons':
        clusters = [[feature] for feature in features]
        clusters = dict(enumerate(clusters))
    elif clustering_method == 'correlation':
        # Compute correlation matrix and distance
        corr = (_df := DataFrameProcessor.filter_columns(features)(X)).corr().abs()
        distance = 1 - corr

        # Manually set distance of sin/cos of cyclical_encoding features to 0, so that they will belong to the same cluster
        sin_feature_names = list(filter(lambda x: 'sin' in x, features))
        for sin_feature_name in sin_feature_names:
            sin_feature_id = features.index(sin_feature_name)
            cos_feature_id = features.index(sin_feature_name.replace("sin", "cos"))
            # print(sin_feature_id, cos_feature_id)
            distance.iloc[sin_feature_id, cos_feature_id] = 0
            distance.iloc[cos_feature_id, sin_feature_id] = 0

        # remove numerical noise (very small negative values)
        distance = np.maximum(distance, 0)

        linkage_matrix = linkage(squareform(distance), method='median')

        # Cut tree at desired threshold (e.g., 0.9)
        cluster_ids: np.ndarray = fcluster(linkage_matrix, t=1 - corr_threshold, criterion='distance')  # 0.1 = 1 - 0.9
        feature_clusters = {}
        for feature, cluster_id in zip(corr.columns, cluster_ids):
            feature_clusters.setdefault(cluster_id, []).append(feature)

        clusters = list(feature_clusters.values())
        clusters = dict(enumerate(clusters))
    elif clustering_method in ['random', 'feature_importance']:
        _features = features.copy()
        if clustering_method == 'feature_importance':
            logging.info("Computing feature importances for clustering:")
            feature_importances = get_feature_importance(X, y, automl_settings=automl_settings)
            # order features by importance
            feature_importance_tuples = list(zip(_features, feature_importances))
            feature_importance_tuples.sort(key=lambda x: x[1], reverse=True)  # descending, most important first
            logging.info("\n".join(f"{t[0]}: {t[1]}" for t in feature_importance_tuples))
            _features = [t[0] for t in feature_importance_tuples]  # extract ordered feature names
        elif clustering_method == 'random':
            np.random.shuffle(_features)
        else:
            raise ValueError(
                f"create_feature_clusters not correctly implemented for clustering_method = '{clustering_method}'")
        clusters = defaultdict(list)
        space_left = np.full(len(_features), fill_value=group_size)
        sin_feature_names = list(filter(lambda x: 'sin' in x, _features))
        cos_feature_names = list(filter(lambda x: 'cos' in x, _features))
        while len(_features) > 0:
            curr_feature = _features.pop()
            if (curr_feature in sin_feature_names) or (curr_feature in cos_feature_names):
                if curr_feature in sin_feature_names:
                    sin_feature = curr_feature
                    cos_feature = sin_feature.replace("sin", "cos")
                    _features.remove(cos_feature)
                else:
                    cos_feature = curr_feature
                    sin_feature = cos_feature.replace("cos", "sin")
                    _features.remove(sin_feature)
                cluster_id = (space_left >= 2).argmax().item()
                clusters[cluster_id].extend([sin_feature, cos_feature])
                space_left[cluster_id] -= 2
            else:
                cluster_id = (space_left >= 1).argmax().item()
                clusters[cluster_id].append(curr_feature)
                space_left[cluster_id] -= 1
    else:
        raise ValueError(f"Unknown clustering_method '{clustering_method}'")

    # Make sure that all features are included in clusters
    clustered_features = [f for group in clusters.values() for f in group]
    if set(clustered_features) != set(features):
        missing_features = set(features) - set(clustered_features)
        raise RuntimeError(f"The following features are missing in the clusters: {missing_features}")
    logging.info(f"Clustered {len(features)} features into {len(clusters)} clusters")
    return clusters


def mean_confidence_interval(data, confidence=0.95):
    """
    Credits to: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    :param data:
    :param confidence:
    :return:
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


class ClusterSequentialFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Cluster-based Sequential Feature Selector (CSFS) implementation.
    """

    def __init__(self,
                 estimator_name: str,
                 fixed_features: list,
                 metrics: dict,
                 scoring: str,
                 val_ratio: float,
                 gap: pd.Timedelta,
                 datetimes: pd.Series,
                 clustering_config: dict,
                 logger: FSLogger,
                 task: Literal['classification', 'regression'],
                 direction: str = 'backward',
                 cv: Optional[list[tuple] | Any] = None,
                 bootstrap_sample_size: Optional[int] = None,
                 n_features_to_select: int = 1,
                 verbose=1,
                 aggregation_mode: Literal['median', 'mean', 'min', 'max'] = 'median',
                 hpo_mode: Literal['off', 'per_feature_set', 'per_iteration'] = 'off',
                 automl_settings: dict = None,
                 automl_ws_settings: dict = None,
                 warm_starts: bool = True,
                 fast_mode: bool = True,
                 early_stopping: bool = False,
                 tolerance_margin: float = 0.0,
                 beta: float = 0.05,
                 correlation_threshold: float = 0.9,
                 random_seed: Optional[int] = None,
                 automl_log_dir: Optional[Path] = None,
                 resume_at_iteration: Optional[int] = None,
                 ):
        """
        Extension of Sequential Feature Selection based on feature clusters.

        CONFIDENCE INTERVALS - CV or Test set bootstrapping is supported for confidence intervals. Currently, you cannot use both simultaneously.
        TEST SET BOOTSTRAPPING
        - You need to define the training set by its indices `train_idx`
        - The number of iterations must be given as `bootstrap_n_iterations`
        - Instead of just one value per metric, you will receive a list of <bootstrap_n_iterations> values
        CV
        - The folds must be defined by an SKLearn CV object, or a list of (train_idx, val_idx) tuples
        - Instead of just one value per metric, you will receive a list of <n_folds> values
        LOGGING - The following things are logged
        - ordered list cluster_ids being discarded or added, depending if it's forward or backward
        - ALL scores for all iterations and feature sets
        - List with length equal to number of iterations with just the best result of each iteration that leaded to the discarding/added decision
        - Best hyperparameters
        - The logs will be stored in the File System via FSLogger

        :param estimator_name: Name of the estimator to use (registered in models.py)
        :param fixed_features: List of feature names that should always be included. Not subject to selection und not counted in n_features_to_select.
        :param direction: 'backward' or 'forward'
        :param metrics: Dictionary of metric functions to evaluate performance. Format: {'log_metric_name': ('metric_name', Lit
        :param scoring: Key of the metric to use for ranking feature sets
        :param val_ratio: Ratio of validation set size to the whole dataset size for temporal split
        :param gap: Gap between training and validation set for time series split
        :param datetimes: Pandas Series with datetime values for time series split
        :param clustering_config: Configuration dictionary for feature clustering
        :param logger: Logger instance for logging metrics and artifacts
        :param task: 'classification' or 'regression'
        :param cv: Cross-validation splitting strategy
        :param bootstrap_sample_size: Number of bootstrap samples for test set bootstrapping
        :param n_features_to_select: Desired number of features to select. Default: 1
        :param verbose: Verbosity level. Default: 1
        :param aggregation_mode: 'median', 'mean', 'min', or 'max' for aggregating CV or bootstrap results. Default: 'median'
        :param hpo_mode: 'off', 'per_feature_set', or 'per_iteration' for hyperparameter optimization. Default: 'off'
        :param automl_settings: AutoML settings for hyperparameter optimization
        :param automl_ws_settings: AutoML settings for warm-start hyperparameter optimization
        :param warm_starts: Whether to use warm-start hyperparameter optimization. Default: True
        :param fast_mode: Whether to enable fast mode (stop evaluating clusters after first non-inferior cluster). Default: True
        :param early_stopping: Whether to enable early stopping if no feature is non-inferior at feature-level testing. Default: False
        :param tolerance_margin: Tolerance margin for non-inferiority testing. Default: 0.0
        :param beta: Significance level for non-inferiority testing. Default: 0.05
        :param correlation_threshold: Correlation threshold for feature clustering. Default: 0.9
        :param random_seed: Random seed for reproducibility. Default: None
        :param automl_log_dir: Directory to store AutoML logs. Default: None
        :param resume_at_iteration: Iteration number to resume from. Logs until this iteration must be present in the logger. Default: None
        """
        self.estimator_name = estimator_name
        self.fixed_features = fixed_features
        self.direction = direction
        self.metrics = metrics
        self.scoring = scoring
        self.aggregation_mode = aggregation_mode
        self.cv = cv
        self.bootstrap_sample_size = bootstrap_sample_size
        self.val_ratio = val_ratio
        self.gap = gap
        self.datetimes = datetimes
        self.n_features_to_select = n_features_to_select
        self.verbose = verbose
        self.support_ = None
        self.history_ = []
        self.hpo_mode = hpo_mode
        self.fast_stop = fast_mode
        self.early_stopping = early_stopping
        self.tolerance_margin = tolerance_margin
        self.beta = beta
        self.warm_starts = warm_starts
        self.automl_settings = automl_settings
        self.automl_ws_settings = automl_ws_settings
        self.automl_log_dir = automl_log_dir
        self.logger = logger
        self.task = task
        self.clustering_config = clustering_config
        self.correlation_threshold = correlation_threshold
        self.resume_at_iteration = resume_at_iteration
        # self.logging_prefix = logging_prefix if logging_prefix == '' or logging_prefix[
        #     -1] == '/' else logging_prefix + '/'

        if direction not in ['backward', 'forward']:
            raise ValueError("direction must be 'backward' or 'forward'")

        if aggregation_mode not in ['median', 'avg', 'min', 'max']:
            raise ValueError("ranking must be 'median', 'avg', 'min', or 'max'")

        if self.hpo_mode != 'off' and self.automl_settings is None:
            raise ValueError("automl_config must be provided if hpo is turned on.")

        if warm_starts and self.automl_ws_settings is None:
            raise ValueError("automl_ws_settings must be provided if warm_starts is turned on.")
            self.automl_ws_settings = self.automl_settings.copy()
            self.automl_ws_settings.update(automl_warmstart_settings if automl_warmstart_settings is not None else {})

        if self.cv is not None and self.val_ratio is not None:
            raise ValueError("Either cv or val_ratio can be provided, not both.")

        if self.cv is None:
            train_val_indices = DataUtils.get_time_split_indices(
                self.datetimes,
                gap=self.gap,
                split_ratio=(1 - self.val_ratio, self.val_ratio),
            )
            self.cv = [(train_val_indices[0], train_val_indices[1])]

        if self.scoring not in self.metrics.keys():
            raise ValueError("scoring must be one of the provided metrics' keys.")

        seed_everything(random_seed)

        self.curr_iteration = -1
        self._selected_cluster_ids: list[int] = []
        self._cluster_priorities = {}
        self._fit_successful = False

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]):
        if self._fit_successful:
            raise RuntimeError("Already fitted. Re-instantiate to refit.")

        X, y = self._validate_fit_input(X, y)
        self.cv = self._prepare_cv(X, y)
        self._console_log("Creating feature clusters for CSFS.")
        _t0_create_feature_clusters = time.perf_counter()
        self.clusters = create_feature_clusters(X=X,
                                                y=y,
                                                automl_settings=self.automl_settings,
                                                corr_threshold=self.correlation_threshold,
                                                **self.clustering_config)
        _t1_create_feature_clusters = time.perf_counter()
        feature_clustering_duration = _t1_create_feature_clusters - _t0_create_feature_clusters

        self._console_log(
            f"Finished feature clustering in {feature_clustering_duration} s. Using the following {len(self.clusters)} clusters for CSFS:\n"
            + "\n".join(
                f"Cluster {i}: {cluster}" for i, cluster in self.clusters.items()))

        X, y = self._validate_clusters(X, y)

        if self.bootstrap_sample_size is not None:
            logging.info("Generating bootstrap indices for test set bootstrapping.")
            self.bootstrap_indices = {fold_id: np.random.randint(low=0, high=len(train_val_idx[1]),
                                                                 size=(self.bootstrap_sample_size,
                                                                       len(train_val_idx[1]))) for
                                      fold_id, train_val_idx in enumerate(self.cv)}
        else:
            self.bootstrap_indices = None

        self._console_log(self.describe_cv(), method_name="fit")

        if self.resume_at_iteration is not None:
            self._restore_state()
        else:
            self.logger.save_object("initial_clusters", self.clusters)
            self.logger.log_metrics({
                "feature_clustering_duration": feature_clustering_duration
            })

        while (not self._has_finished()):
            self.curr_iteration += 1
            logging.info(f"Starting iteration {self.curr_iteration + 1}")
            logging.info(f"Current selected feature count: {self._currently_selected_features()}")
            logging.info(f"Desired feature count: {self.n_features_to_select}")

            current_features = self._get_current_features(clusters_under_test=self._get_cluster_queue(),
                                                          curr_selected_cluster_ids=self._selected_cluster_ids,
                                                          curr_tested_cluster_id=None,
                                                          curr_tested_feature=None)

            if self.warm_starts:
                self._run_hpo_warmup(X, y, current_features, self.cv)

            if self.hpo_mode == 'per_iteration':
                self._run_hpo(X, y, current_features, self.cv)

            self._create_baseline(X, y, current_features, self.cv)

            selected_cluster_id = None
            for j, (cluster_id, cluster) in enumerate(self._get_cluster_queue().items()):
                self._console_log(
                    f"Evaluating cluster {j + 1}/{len(self._get_cluster_queue())}: {cluster}")
                # cluster: Feature cluster to be tested in this loop.
                # cluster_id: ID of the cluster being tested.

                current_features = self._get_current_features(clusters_under_test=self._get_cluster_queue(),
                                                              curr_selected_cluster_ids=self._selected_cluster_ids,
                                                              curr_tested_cluster_id=cluster_id,
                                                              curr_tested_feature=None)
                self._console_log(
                    f"Current features: {current_features}"
                )

                if len(current_features) == 0:
                    if j > 0:
                        raise RuntimeError(
                            "No features would remain after removing cluster, but not the first (and only, as expected) cluster.")
                    self._console_log(
                        f"Skipping evaluation of cluster {cluster_id} because no features would remain.")
                    continue

                t0 = time.perf_counter()
                self._evaluate_cluster(X, y, current_features, self.cv, cluster_id)
                t1 = time.perf_counter()
                self._console_log(f"Cluster {cluster_id} evaluated in {t1 - t0:.2f} seconds.")

                if self.fast_stop and not self._would_violate_selection_size(cluster_id) and self._cluster_non_inferior(
                        cluster_id):
                    selected_cluster_id = cluster_id
                    self._console_log(
                        f"Fast mode: Cluster {cluster_id} passed non-inferiority test. Skipping evaluation of other clusters.")
                    self.logger.log_metrics({"fast_stop_cluster_id": cluster_id})
                    self.logger.log_metrics({"fast_stop_iteration": self.curr_iteration})
                    self.logger.log_metrics({"fast_stop_evaluated_clusters": j + 1})
                    break

            # First cluster that is non-inferior and respects selection size is chosen
            if selected_cluster_id is None:
                for cluster_id, cluster in self._get_cluster_queue().items():
                    if self._would_violate_selection_size(cluster_id):
                        continue
                    if self._cluster_non_inferior(cluster_id):
                        self._console_log(f"Cluster {cluster_id} passed non-inferiority test.")
                        selected_cluster_id = cluster_id
                        break

            if selected_cluster_id is None:
                self._console_log(f"No cluster was non-inferior, entering feature-level testing.")
                self.logger.log_metrics({"feature_level_iteration": self.curr_iteration})
                # No cluster was non-inferior, enter feature-level testing
                selected_feature = None
                all_features = dict()
                evaluated_features = 0
                for cluster_id, cluster in self._get_cluster_queue().items():
                    if selected_feature is not None:
                        break
                    if len(cluster) == 1:
                        # skip single-feature clusters because they were already tested at cluster level above
                        continue
                    for feature in cluster:
                        all_features[feature] = cluster_id
                        current_features = self._get_current_features(clusters_under_test=self._get_cluster_queue(),
                                                                      curr_selected_cluster_ids=self._selected_cluster_ids,
                                                                      curr_tested_feature=feature,
                                                                      curr_tested_cluster_id=None)
                        self._evaluate_feature(X, y, current_features, self.cv, feature_name=feature)
                        evaluated_features += 1
                        if self._feature_non_inferior(feature):
                            # First feature that is non-inferior is selected
                            self._console_log(f"Feature '{feature}' passed non-inferiority test.")
                            selected_feature = feature
                            break
                self.logger.log_metrics({"feature_level_evaluated_features": evaluated_features})
                if selected_feature is None and self.early_stopping:
                    # No feature was non-inferior, stop if early stopping is enabled
                    self._console_log("CSFS procedure stopped due to early stopping.")
                    break
                elif selected_feature is not None:
                    selected_feature_or_cluster = selected_feature
                else:
                    # Otherwise, select the best feature or cluster
                    self._console_log("No feature was non-inferior, selecting best feature or cluster.")
                    self.logger.log_metrics({"feature_level_none_non-inferior_iteration": self.curr_iteration})
                    selected_feature_or_cluster = self._get_best_cluster(
                        list(all_features.keys()) + list(self._get_cluster_queue().keys()))

                # selected_feature_or_cluster is either a cluster_id (int) or a feature_name (str)
                if selected_feature_or_cluster in self._get_cluster_queue().keys():
                    # It's a cluster_id and thus can be directly selected
                    selected_cluster_id = selected_feature_or_cluster
                else:
                    # It's a feature name, create a new cluster with just this feature
                    selected_cluster_id = self._add_cluster(
                        origin_cluster_id=all_features[selected_feature_or_cluster],
                        feature_name=selected_feature_or_cluster
                    )

                    self._console_log(
                        f"Added new cluster (id={selected_cluster_id}): [{selected_feature_or_cluster}].")

            if selected_cluster_id is None:
                raise RuntimeError("No cluster selected for removal/addition, but procedure not stopped.")
            self._selected_cluster_ids.append(selected_cluster_id)

            self._console_log(f"Added/removed cluster with id={selected_cluster_id}")
            self.logger.log_metrics({'selected_cluster_id': selected_cluster_id}, step=self.curr_iteration)

        logging.info(f"CSFS procedure done!")
        self.logger.save_object('final_clusters', self.clusters)

        self._fit_successful = True

        return self

    def _restore_state(self) -> None:
        """
        Restores the state of the feature selector from logs in the logger.
        Starts from self.resume_at_iteration, if provided.
        :return: None
        """
        if self.resume_at_iteration is None:
            return
        # Step 1: Delete all "future" logs
        directories_to_delete = [self.logger.log_dir / "testing"]
        for typ in ["training", "validation", "warmup", "automl"]:
            directories_to_delete.extend(
                [(self.logger.log_dir / d) for d in (self.logger.log_dir / typ).iterdir() if
                 (self.logger.log_dir / d).is_dir() and int(d.name) >= self.resume_at_iteration]
            )

        # Step 2: Validate clustering
        original_clusters = self.logger.fetch_object("initial_clusters")
        if original_clusters != self.clusters:
            raise RuntimeError("The clusters have changed since the initial fit. Cannot resume.")

        # Step 3: Restore iteration and selected clusters
        self.curr_iteration = self.resume_at_iteration - 1
        self._selected_cluster_ids = list(self.logger.fetch_values("selected_cluster_id"))
        directories_to_delete.append(self.logger.log_dir / "selected_cluster_id.npy")
        self._selected_cluster_ids = self._selected_cluster_ids[:self.resume_at_iteration]
        for iteration, cluster_id in enumerate(self._selected_cluster_ids):
            self.logger.log_metrics({'selected_cluster_id': cluster_id}, step=iteration)

        # Step 4: Get remaining information for logger
        for k in ["fast_stop_iteration", "fast_stop_cluster_id", "feature_level_iteration",
                  "feature_level_none_non-inferior_iteration", "feature_level_evaluated_features",
                  "fast_stop_evaluated_clusters"]:
            values = self.logger.fetch_values(k)
            if values is None:
                continue
            if not hasattr(values, "__len__"):
                values = [values]
            if k == "fast_stop_cluster_id":
                for cluster_id in [v for v in values if v in self._selected_cluster_ids]:
                    self.logger.log_metrics({k: cluster_id})
            elif k in ["fast_stop_iteration", "feature_level_iteration", "feature_level_none_non-inferior_iteration"]:
                for iteration in [v for v in values if v < self.resume_at_iteration]:
                    self.logger.log_metrics({k: iteration})
            elif k == "fast_stop_evaluated_clusters":
                fast_stop_iterations = self.logger.fetch_values("fast_stop_iteration")
                number_of_fast_stop_iterations = 0 if fast_stop_iterations is None else (
                    1 if not hasattr(fast_stop_iterations, "__len__") else len(fast_stop_iterations))
                for n_clusters in values[:number_of_fast_stop_iterations]:
                    self.logger.log_metrics({k: n_clusters})
            elif k == "feature_level_evaluated_features":
                feature_level_iterations = self.logger.fetch_values("feature_level_iteration")
                number_of_feature_level_iterations = 0 if feature_level_iterations is None else (
                    1 if not hasattr(feature_level_iterations, "__len__") else len(feature_level_iterations))
                for n_features in values[:number_of_feature_level_iterations]:
                    self.logger.log_metrics({k: n_features})
            else:
                raise ValueError(f"Unknown key '{k}' in restore_state.")

            directories_to_delete.append(self.logger.log_dir / f"{k}.npy")

        # Step 5: Restore current clusters
        final_clusters = self.logger.fetch_object("final_clusters")
        missing_clusters = set(final_clusters.keys()) - set(self.clusters.keys())
        clusters_to_add = missing_clusters.intersection(self._selected_cluster_ids)
        for cluster_id in clusters_to_add:
            new_cluster = final_clusters[cluster_id].copy()
            assert len(new_cluster) == 1
            feature = new_cluster[0]
            for k in self.clusters.keys():
                if feature in self.clusters[k]:
                    self.clusters[k].remove(feature)
                    break
            self.clusters[cluster_id] = new_cluster

        # Step 6: Remaining data to delete
        for d in ["test_idx.npy", "train_val_idx.npy", "test_bootstrap_indices.npy", "status.txt",
                  "feature_names_out.pkl", "fs_runtime.npy", "final_clusters.pkl", ]:
            directories_to_delete.append(self.logger.log_dir / d)

        # Delete directories/files
        for d in directories_to_delete:
            if d.exists():
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()

        # Note: Cluster priorities cannot be restored, but this should not be a big issue.

    def _validate_fit_input(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Validates the input data for fitting.
        :param X: Pandas DataFrame with shape (num_samples, num_features). Fixed features must be present in columns.
        :param y: Pandas Series or numpy array with shape (num_samples,)
        :return:
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if isinstance(y, pd.Series):
            y = y.values
        y = y.ravel()

        features = X.columns.tolist()
        if not set(self.fixed_features).issubset(set(features)):
            raise ValueError("Some fixed_features are not present in X columns.")

        if len(features) - len(self.fixed_features) < self.n_features_to_select:
            raise ValueError(
                "n_features_to_select is larger than the number of available features after accounting for fixed_features.")

        return X, y

    def _validate_clusters(self, X: pd.DataFrame, y: np.ndarray):
        """
        Validates that the clusters and fixed features are compatible with the input data.
        :param X: Pandas DataFrame with shape (num_samples, num_features)
        :param y: Numpy array with shape (num_samples,)
        :return: X, y
        """
        all_cluster_features = [f for group in self.clusters.values() for f in group]
        if not set(self.fixed_features).isdisjoint(all_cluster_features):
            raise ValueError("fixed_features and features in clusters must be disjoint")

        expected_features = []
        expected_features.extend(self.fixed_features)
        expected_features.extend(all_cluster_features)

        if set(X.columns).issuperset(expected_features):
            X = X.loc[:, expected_features]
        else:
            raise ValueError(
                "The columns in X must be a superset of the expected features. The following are missing: {}".format(
                    set(expected_features) - set(X.columns)))

        return X, y

    def _prepare_cv(self, X: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Prepares the data splits for cross-validation.
        :param X:
        :param y:
        :return:
        """
        if self.cv is not None:
            if isinstance(self.cv, list):
                cv = self.cv  # list(check_cv(self.cv, y, classifier=is_classifier(self.estimator_name)).split(X, y))
            else:
                cv = list(self.cv.split(X, y))
        else:
            raise ValueError("cv must be provided.")
        return cv

    def _has_finished(self) -> bool:
        """
        Checks if the desired number of features has been selected.
        :return: True iff the selection is complete.
        """
        return self._currently_selected_features() == self.n_features_to_select

    def _currently_selected_features(self) -> int:
        """
        Computes the number of currently selected features (excluding fixed features).
        :return: int: Number of currently selected features.
        """
        return len(self.get_support()) - len(self.fixed_features)

    def _run_hpo_warmup(self, X: pd.DataFrame, y: np.ndarray, current_features: list[str],
                        cv: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Runs warm-up hyperparameter optimization to gather starting points for the current iteration.
        :param X: Pandas DataFrame with shape (num_samples, num_features)
        :param y: Numpy array with shape (num_samples,)
        :param current_features: List of feature names to use for HPO warm-up
        :param cv: List of (train_idx, val_idx) tuples for cross-validation
        :return: None. Everything is logged via self.logger.
        """
        for fold_id, (train_idx, _) in enumerate(cv):
            # print(f"[Iteration {iteration}] Fold {fold_id}")
            # print(
            #     f"Memory usage before fold {fold_id}: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.2f} MB")
            self._console_log(
                "Starting WarmUp", fold_id=fold_id, method_name="_run_hpo_warmup")
            automl_ws = get_automl_with_registered_models(self.automl_ws_settings['estimator_list'])
            warmup_start_time = time.perf_counter()
            automl_ws.fit(X.iloc[train_idx].loc[:, current_features],
                          y[train_idx],
                          **self.automl_ws_settings,
                          log_file_name=self._get_automl_logdir(fold_id, filename="automl_warmup"), )
            warmup_end_time = time.perf_counter()
            warmup_duration = warmup_end_time - warmup_start_time
            n_trials = automl_ws._search_states[self.estimator_name].total_iter
            curr_metrics = {
                "n_trials": n_trials,
                "duration": warmup_end_time - warmup_start_time
            }

            logs_parent_key = f"warmup/{self.curr_iteration}/{fold_id}"
            self.logger.save_object(f"{logs_parent_key}/starting_points", automl_ws.best_config_per_estimator)
            self.logger.save_object(f"{logs_parent_key}/best_config", automl_ws.best_config)
            self.logger.log_metrics(flatten_dict(curr_metrics, parent_key=logs_parent_key))
            self._console_log(
                f"WarmUp done - {n_trials} Trials in {warmup_duration} seconds.", fold_id=fold_id,
                method_name="_run_hpo_warmup")

    def _console_log(self, msg: str,
                     level: int = logging.INFO,
                     fold_id: Optional[int] = None,
                     method_name: Optional[str] = None):
        prefix = ""
        prefix += f"[{method_name}]" if method_name is not None else ""
        prefix += f"[Iteration {self.curr_iteration + 1}]"
        prefix += f"[Fold {fold_id + 1}]" if fold_id is not None else ""
        logging.log(level, f"{prefix} {msg}")

    def _run_hpo(self, X: pd.DataFrame, y: np.ndarray, current_features: list[str],
                 cv: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Runs hyperparameter optimization for the current iteration. The optimal hyperparameters are saved via self.logger.
        :param X: Pandas DataFrame with shape (num_samples, num_features)
        :param y: Numpy array with shape (num_samples,)
        :param current_features: List of feature names to use for HPO.
        :param cv: List of (train_idx, val_idx) tuples for cross-validation.
        :return: None. Everything is logged via self.logger.
        """
        for fold_id, (train_idx, _) in enumerate(cv):
            self._console_log("Starting HPO", fold_id=fold_id, method_name="_run_hpo")
            automl = get_automl_with_registered_models(self.automl_settings['estimator_list'])
            starting_points = self.logger.fetch_object(
                f"warmup/{self.curr_iteration}/{fold_id}/starting_points")
            if starting_points is not None:
                self._console_log("Using starting_points", fold_id=fold_id, method_name="_run_hpo")
            elif starting_points is None and self.warm_starts:
                raise RuntimeError(
                    f"Warm starts are enabled, but no starting points found under 'warmup/{self.curr_iteration}/{fold_id}/starting_points'")
            hpo_start_time = time.perf_counter()
            automl.fit(X.iloc[train_idx].loc[:, current_features], y[train_idx],
                       **self.automl_settings,
                       starting_points=starting_points,
                       log_file_name=self._get_automl_logdir(fold_idx=fold_id, filename="automl_hpo"), )
            hpo_end_time = time.perf_counter()
            hpo_duration = hpo_end_time - hpo_start_time
            n_trials = automl._search_states[self.estimator_name].total_iter

            curr_metrics = {
                "n_trials": n_trials,
                "duration": hpo_duration
            }
            self._console_log(f"HPO done - {n_trials} Trials in {hpo_duration} seconds.", fold_id=fold_id,
                              method_name="_run_hpo")
            logs_parent_key = f"hpo/{self.curr_iteration}/{fold_id}"
            self.logger.log_metrics(flatten_dict(curr_metrics, parent_key=logs_parent_key))
            self.logger.save_object(f"hpo/{self.curr_iteration}/{fold_id}/best_config",
                                    automl.best_config)
            self.logger.save_object(f"hpo/{self.curr_iteration}/{fold_id}/estimator", clone(automl.model.estimator))

    def _get_automl_logdir(self, fold_idx: int, filename: str) -> Optional[str]:
        """
        Gets the AutoML log directory for the current iteration and fold.
        :param fold_idx: Index of the fold.
        :param filename: Name of the AutoML log file. ".log" will be appended automatically.
        :return: Optional[str]: Path to the AutoML log directory, or None if not set. Example: "<automl_log_dir>/0/0/<filename>.log"
        """
        if self.automl_log_dir is not None:
            curr_log_dir = self.automl_log_dir / str(self.curr_iteration) / str(
                fold_idx) / f"{filename}.log"
            curr_log_dir.parent.mkdir(parents=True, exist_ok=True)
            curr_log_dir = str(curr_log_dir)
        else:
            curr_log_dir = None
        return curr_log_dir

    def _get_estimator_and_fit_kwargs(self, cluster_id: Optional[int | str],
                                      fold_id: int,
                                      use_baseline_est: bool = False):
        """
        Gets the estimator and fit kwargs for the current iteration, cluster, and fold.
        If hpo_mode is 'per_iteration', the pre-optimized estimator is fetched from the logger and will be used.
        If use_baseline_est is True, the baseline config will be used.
        Otherwise, a new estimator is created:
        If hpo_mode is 'per_feature_set', the automl_settings are used as fit kwargs.
        If hpo_mode is 'off', no fit kwargs are used, and a standard estimator with default config is created.
        :param cluster_id: ID of the cluster being evaluated.
        :param fold_id: ID of the fold being evaluated.
        :param use_baseline_est: Whether to use the baseline estimator.
        :return: scikit-learn-like estimator (scikit-learn or AutoML) and fit kwargs (dict)
        """
        fit_kwargs = {}
        if self.hpo_mode == 'per_iteration':
            est = self.logger.fetch_object(f"hpo/{self.curr_iteration}/{fold_id}/estimator")
        elif use_baseline_est:
            best_config = self.logger.fetch_object(f"training/{self.curr_iteration}/baseline/{fold_id}/best_config")
            if best_config is None:
                raise RuntimeError(
                    f"use_baseline_est==True, but no pre-optimized config found under 'training/{self.curr_iteration}/baseline/{fold_id}/best_config'")
            est = get_sklearn_estim(estimator_name=self.estimator_name,
                                    task=self.task,
                                    best_flaml_config=best_config)
        else:
            est = None
        if est is None:
            if self.hpo_mode == 'per_iteration' or use_baseline_est:
                raise RuntimeError(
                    f"hpo_mode is 'per_iteration' or use_baseline_est==True, but no pre-optimized estimator found under 'hpo/{self.curr_iteration}/{fold_id}/estimator' or 'training/{self.curr_iteration}/baseline/{fold_id}/best_config'")
            elif self.hpo_mode == 'off':
                est = get_sklearn_estim(self.estimator_name, task=self.task, best_flaml_config=None)
            else:  # self.hpo_mode == 'per_feature_set'
                fit_kwargs = self.automl_settings.copy()
                if self.automl_log_dir is not None:
                    if cluster_id is not None:
                        curr_log_dir = self.automl_log_dir / str(self.curr_iteration) / str(cluster_id) / str(
                            fold_id) / "automl_hpo.log"
                    else:
                        curr_log_dir = self.automl_log_dir / str(self.curr_iteration) / str(
                            fold_id) / "automl_hpo.log"
                    curr_log_dir.parent.mkdir(parents=True, exist_ok=True)
                    fit_kwargs["log_file_name"] = str(curr_log_dir)
                fit_kwargs['starting_points'] = self.logger.fetch_object(
                    f"warmup/{self.curr_iteration}/{fold_id}/starting_points")
                if self.warm_starts and fit_kwargs['starting_points'] is None:
                    raise RuntimeError(
                        f"Warm starts are enabled, but no starting points found under 'warmup/{self.curr_iteration}/{fold_id}/starting_points'")
                est = get_automl_with_registered_models(self.automl_settings['estimator_list'])
        else:
            self._console_log("Using pre-optimized estimator", fold_id=fold_id, method_name="_evaluate_cluster")

        return est, fit_kwargs

    def _evaluate_cluster(self, X: pd.DataFrame,
                          y: np.ndarray,
                          current_features: list[str],
                          cv: list[tuple[np.ndarray, np.ndarray]],
                          cluster_id: int | str,
                          use_baseline_est: bool = False) -> None:
        """
        Evaluates a feature cluster by training and validating the estimator on the given features.
        Logs training and validation metrics via self.logger.
        :param X: Pandas DataFrame with shape (num_samples, num_features)
        :param y: Numpy array with shape (num_samples,)
        :param current_features: List of feature names to use for evaluation.
        :param cv: List of (train_idx, val_idx) tuples for cross-validation.
        :param cluster_id: ID of the cluster being evaluated (only needed for logging purposes).
        :param use_baseline_est: Whether to use the baseline estimator configuration.
        :return: None. Everything is logged via self.logger.
        """
        for fold_id, (train_idx, val_idx) in enumerate(cv):
            est, fit_kwargs = self._get_estimator_and_fit_kwargs(cluster_id, fold_id, use_baseline_est=use_baseline_est)
            fit_start_time = time.perf_counter()
            est.fit(X.iloc[train_idx].loc[:, current_features], y[train_idx], **fit_kwargs)
            fit_end_time = time.perf_counter()
            training_metrics = {"duration": fit_end_time - fit_start_time}

            if isinstance(est, AutoML):
                self.logger.save_object(
                    f"training/{self.curr_iteration}/{cluster_id}/{fold_id}/best_config",
                    est.best_config)
                training_metrics['n_trials'] = est._search_states[self.estimator_name].total_iter
                est = est.model.estimator

            validation_metrics = evaluate_on_test_set(est,
                                                      X.iloc[val_idx].loc[:, current_features],
                                                      y[val_idx],
                                                      scoring=self.metrics,
                                                      bootstrap_sample_size=self.bootstrap_sample_size,
                                                      test_set_bootstrap=True,
                                                      bootstrap_indices=self.bootstrap_indices[
                                                          fold_id] if self.bootstrap_indices is not None else None, )

            self._cluster_priorities[cluster_id] = self._aggregate(validation_metrics[self.scoring])

            self.logger.log_metrics(
                flatten_dict(training_metrics, parent_key=f"training/{self.curr_iteration}/{cluster_id}/{fold_id}")
            )

            self.logger.log_metrics_array(
                flatten_dict(validation_metrics, parent_key=f"validation/{self.curr_iteration}/{cluster_id}"))

    def _aggregate(self, arr: np.ndarray) -> np.floating[Any]:
        if self.aggregation_mode == 'min':
            return np.min(arr)
        elif self.aggregation_mode == 'max':
            return np.max(arr)
        elif self.aggregation_mode == 'avg':
            return np.mean(arr)
        elif self.aggregation_mode == 'median':
            return np.median(arr)
        elif self.aggregation_mode.startswith('p='):
            p = float(self.aggregation_mode.split('=')[1])
            return np.percentile(arr, p)
        else:
            raise ValueError(
                f"Unknown ranking_type '{self.aggregation_mode}' (choose from 'min', 'max', 'avg', 'median', 'p=<float_value>')")

    def _copy_results_and_objects(self, source: str, dest: str) -> None:
        """
        Logging helper to copy all metrics and objects from source to dest in the logger.
        :param source: Source key.
        :param dest: Destination key.
        :return: None
        """
        self.logger.log_metrics_array(
            flatten_dict(
                self.logger.fetch_all_values(source),
                parent_key=dest
            )
        )
        self.logger.copy_all_objects(source, dest)

    def _create_baseline(self, X: pd.DataFrame, y: np.ndarray, current_features: list[str],
                         cv: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        For the first iteration (self.curr_iteration == 0), creates validation results with all existing features.
        For subsequent iterations, it is not necessary to recompute the results.
        Instead, the already logged results for that feature set from the previous iteration are taken.
        :param X: Pandas DataFrame with shape (num_samples, num_features). Only used if self.curr_iteration == 0.
        :param y: Numpy array with shape (num_samples,). Only used if self.curr_iteration == 0.
        :param cv: List of (train_idx, val_idx) tuples for cross-validation.
        :return: None. Everything is logged via self.logger.
        """
        _iterations = [0] + [] if self.resume_at_iteration is None else [self.resume_at_iteration]
        if self.curr_iteration in _iterations:
            self._evaluate_cluster(X, y, current_features, cv, cluster_id="baseline")
        else:
            # copy previous best results to current iteration (-> serves as baseline for current iteration)
            # no need to recompute / re-evaluate
            for s in ['training', 'validation']:
                source = f"{s}/{self.curr_iteration - 1}/{self._selected_cluster_ids[-1]}"
                dest = f"{s}/{self.curr_iteration}/baseline"
                self._copy_results_and_objects(source, dest)

    def _evaluate_feature(self, X: pd.DataFrame, y: np.ndarray, current_features: list[str],
                          cv: list[tuple[np.ndarray, np.ndarray]], feature_name: str) -> None:
        """
        Evaluates a single feature by training and validating the estimator on the given features.
        Logs training and validation metrics via self.logger.
        In this case, feature_name (str) is used as cluster_id (otherwise normally int) for logging purposes.
        :param X: Pandas DataFrame with shape (num_samples, num_features)
        :param y: Numpy array with shape (num_samples,)
        :param current_features: List of feature names to use for evaluation.
        :param cv: List of (train_idx, val_idx) tuples for cross-validation.
        :param feature_name: Name of the feature being evaluated.
        :return: None. Everything is logged via self.logger.
        """
        self._evaluate_cluster(X, y, current_features, cv, cluster_id=feature_name, use_baseline_est=True)

    def _get_cluster_queue(self):
        """
        Get the clusters that are still available for selection (deletion for 'backward', addition for 'forward' mode), ordered by priority.
        The priority is determined by the aggregated primary metric scores from previous evaluations.
        The initial priority is np.inf for all clusters, fostering exploration.
        :return: dict: Remaining clusters ordered by priority (highest priority first).
        """
        # Filter out already selected clusters
        remaining_clusters = {
            k: v for k, v in self.clusters.items() if k not in self._selected_cluster_ids
        }

        # Sort by priority (descending: higher priority first)
        sorted_items = sorted(
            remaining_clusters.items(),
            key=lambda item: self._cluster_priorities.get(item[0], np.inf),
            # default priority np.inf -> foster exploration
            reverse=True
        )

        # Preserve ordering by constructing a new dict
        return dict(sorted_items)

    def _get_best_cluster(self,
                          cluster_ids: list[int | str]) -> int | str:
        """
        Select the best cluster based on the primary metric.
        :param cluster_ids: Cluster ids to consider. Can also include feature names (str) in case of feature-level testing.

        :return: Best cluster id (int) or feature name (str)
        """

        # remove cluster_ids that would violate selection size
        cluster_ids = [cid for cid in cluster_ids if not self._would_violate_selection_size(cid)]

        score_dict = self.logger.fetch_all_values(f"validation/{self.curr_iteration}")

        # filter for primary metric
        score_dict = {cluster_id: score_dict[str(cluster_id)][self.scoring] for cluster_id in cluster_ids}

        # Aggregate scores per cluster
        score_dict = {cluster_id: self._aggregate(scores) for cluster_id, scores in score_dict.items()}

        # Get cluster with best (=highest) aggregated score
        # If multiple clusters have the same best score, choose one randomly
        best_score = max(score_dict.values())
        cluster_candidates = [cluster_id for cluster_id, score in score_dict.items() if score == best_score]
        cluster_candidate = random.choice(cluster_candidates)

        return cluster_candidate

    @staticmethod
    def _non_inferior(baseline_scores: np.ndarray, test_scores: np.ndarray, margin: float, beta: float) -> bool:
        """
        Perform a one-sided non-inferiority test to check if the test_scores are not worse than baseline_scores
        by more than the specified margin with a significance level of beta.
        :param baseline_scores: Array of baseline scores.
        :param test_scores: Array of test scores.
        :param margin: Non-inferiority margin.
        :param beta: Significance level.
        :return: True if non-inferiority is established, False otherwise.
        """
        differences = test_scores - baseline_scores
        lower_bound = np.quantile(differences, beta)
        return lower_bound >= -margin

    def _cluster_non_inferior(self, cluster_id: int) -> bool:
        """
        Perform a one-sided non-inferiority test for a cluster.
        :param cluster_id: ID of the cluster to test. Must already have been evaluated.
        :return: True if non-inferiority is established, False otherwise.
        """
        baseline_score_array = self.logger.fetch_values(
            f"validation/{self.curr_iteration}/baseline/{self.scoring}")
        cluster_score_array = self.logger.fetch_values(
            f"validation/{self.curr_iteration}/{cluster_id}/{self.scoring}")

        return self._non_inferior(baseline_score_array, cluster_score_array, self.tolerance_margin, self.beta)

    def _feature_non_inferior(self, feature_name: str) -> bool:
        """
        Perform a one-sided non-inferiority test for a single feature.
        :param feature_name: Name of the feature to test. Must already have been evaluated.
        :return: True if non-inferiority is established, False otherwise.
        """
        baseline_score_array = self.logger.fetch_values(
            f"validation/{self.curr_iteration}/baseline/{self.scoring}")
        cluster_score_array = self.logger.fetch_values(
            f"validation/{self.curr_iteration}/{feature_name}/{self.scoring}")

        return self._non_inferior(baseline_score_array, cluster_score_array, self.tolerance_margin, self.beta)

    def _would_violate_selection_size(self, cluster_id: int | str) -> bool:
        """
        Check if the selected cluster can be safely removed/added.
        If we would over-/undershoot self.n_clusters_to_select, it cannot be safely removed.
        :param cluster_id: ID of the cluster to check (int) or feature name (str).
        :return: False if can be safely removed/added, True if would violate selection size.
        """
        cluster_size = len(self.clusters[cluster_id]) if isinstance(cluster_id, int) else 1
        new_selected_feature_count = self._currently_selected_features() - cluster_size if self.direction == 'backward' else \
            self._currently_selected_features() + cluster_size

        if self.direction == 'backward' and new_selected_feature_count < self.n_features_to_select:
            # We would undershoot
            return True

        if self.direction == 'forward' and new_selected_feature_count > self.n_features_to_select:
            # We would overshoot
            return True

        return False

    def _add_cluster(self, origin_cluster_id: int, feature_name: str) -> int:
        """
        Take a feature from an existing origin cluster and create a new cluster with just this feature.
        :param origin_cluster_id: ID of the cluster from which to take the feature.
        :param feature_name: Name of the feature to create a new cluster with.
        :return: ID of the new cluster
        """
        if feature_name not in self.clusters[origin_cluster_id]:
            raise ValueError(f"Feature {feature_name} not in cluster {origin_cluster_id}")
        if len(self.clusters[origin_cluster_id]) == 1:
            raise ValueError(f"Cluster {origin_cluster_id} only contains one feature, cannot split further.")
        new_cluster_id = max(self.clusters.keys()) + 1
        self.clusters[new_cluster_id] = [feature_name]
        self.clusters[origin_cluster_id].remove(feature_name)

        # copy logged results from origin cluster to new cluster for current iteration
        for s in ['training', 'validation']:
            source = f"{s}/{self.curr_iteration}/{origin_cluster_id}"
            dest = f"{s}/{self.curr_iteration}/{new_cluster_id}"
            self._copy_results_and_objects(source, dest)

        return new_cluster_id

    def describe_cv(self) -> str:
        """
        Describes the cross-validation splits used for logging purposes.
        :return: str: Description of the cross-validation splits.
        """
        if not isinstance(self.cv, list):
            raise ValueError("cv must be a list of (train_idx, val_idx) tuples to describe.")
        if len(self.cv) > 1:
            string = f"Using {len(self.cv)}-fold cross-validation.\n"
        else:
            string = "Using a single train/validation split.\n"

        for fold in range(len(self.cv)):
            for split_idx, split_typ in zip(self.cv[fold], ["Train", "Val"]):
                if len(self.cv) > 1:
                    string += f"Fold {fold}:"
                string += f"{split_typ}: index=[{split_idx[0]}, {split_idx[-1]}]\n"
                string += f"{split_typ}: Datetime=[{self.datetimes.iloc[split_idx[0]]}, {self.datetimes.iloc[split_idx[-1]]}]\n"

        return string

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Only keeps the most important features. Fit must have been called before.
        :param X: Pandas DataFrame with the same features used for fitting.
        :return: DataFrame with unimportant features removed.
        """
        if not self._fit_successful:
            raise RuntimeError("transform() called before fit() or fit() was not successful.")
        return X.loc[:, self.get_support()]

    def _update_support(self):
        """
        Updates the support_ attribute with the currently selected features.
        :return:
        """
        included_cluster_ids = self._selected_cluster_ids if self.direction == 'forward' else set(
            self.clusters.keys()) - set(self._selected_cluster_ids)
        self.support_ = self.fixed_features + [_cluster_feature for _cluster_id in included_cluster_ids for
                                               _cluster_feature in
                                               self.clusters[_cluster_id]]

    def get_support(self) -> list[str]:
        """
        Get the features that have been selected by the procedure.
        :return: List of most important feature names.
        """
        self._update_support()
        return self.support_

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """
        Alias for get_support().
        :param input_features: Ignored.
        :return: List of most important feature names.
        """
        return self.get_support()

    def _get_current_features(self,
                              clusters_under_test: dict[int, list[str]],
                              curr_selected_cluster_ids: Optional[list[int]],
                              curr_tested_cluster_id: Optional[int],
                              curr_tested_feature: Optional[str]) -> list[str]:
        """
        Get the list of features to be used for training and evaluation.
        :param clusters_under_test: All remaining clusters that are still under test. Only needed in 'backward' mode.
        :param curr_selected_cluster_ids: List all already selected cluster ids. Only needed in 'forward' mode.
        :param curr_tested_cluster_id: ID of the cluster to be evaluated now.
        :param curr_tested_feature: Alternatively, name of the single feature to be evaluated now.
        :return: List of feature names to be used for training and evaluation. Fixed features are always included.
        """
        if curr_tested_cluster_id is not None and curr_tested_feature is not None:
            raise ValueError("Either curr_tested_cluster_id or curr_tested_feature can be given, but not both.")
        if self.direction == 'backward':  # The cluster is removed in the backward case
            current_clusters = clusters_under_test.copy()
            if curr_tested_cluster_id is not None:
                current_clusters.pop(curr_tested_cluster_id)
        else:  # forward
            # select all clusters that have already been added because they improved the score
            current_clusters = {_cluster_id: self.clusters[_cluster_id] for _cluster_id in
                                curr_selected_cluster_ids}
            if curr_tested_cluster_id is not None:
                # add the current cluster that should be checked for score improvement
                current_clusters[curr_tested_cluster_id] = self.clusters[curr_tested_cluster_id]

        # current_clusters now contains the feature clusters being used for training in the remainder of this loop
        # Get a flat list of all features contained in these clusters:
        current_features = [_cluster_feature for _cluster in current_clusters.values() for _cluster_feature in
                            _cluster]

        if curr_tested_feature is not None:
            if self.direction == 'forward':
                if curr_tested_feature not in current_features:
                    current_features.append(curr_tested_feature)
                else:
                    raise ValueError(
                        f"Feature {curr_tested_feature} already in current features, cannot add again.")
            else:  # backward
                if curr_tested_feature in current_features:
                    current_features.remove(curr_tested_feature)
                else:
                    raise ValueError(
                        f"Feature {curr_tested_feature} not in current features, cannot remove.")

        # If there are any fixed features excluded from the feature selection process,
        # but used anyway, these are added here:
        current_features.extend(self.fixed_features)

        return current_features
