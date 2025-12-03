import logging
import time
import warnings
from typing import Optional, Callable, Literal
import numpy as np
import pandas as pd


def data_leakage(test_data: pd.DataFrame,
                 train_data: pd.DataFrame,
                 group_col: Optional[str],
                 datetime_col: Optional[str],
                 alpha: float = .0) -> float:
    """
    Implementation of the data leakage measure as defined in "Quantifying Data Leakage in Failure Prediction Tasks".
    As explained in the paper, the measure is solely based on the group and time information. Further features will
    not be considered in the calculation.
    :param test_data: Pandas Dataframe with the test_data. Contains a group-identifying column <group_col> and a datetime column <datetime_col>
    :param train_data: Pandas Dataframe with the train_data. Contains a group-identifying column <group_col> and a datetime column <datetime_col>
    :param group_col: The column name of the group-identifying column, if groups are present in the data. If None, all observations are considered to belong to the same group.
    :param datetime_col: The column name of the datetime column.
    :param alpha: The weight of the leakage contribution for t_2 < t_1. Default is 0.
    :return: The normalized leakage value (float).
    """

    # Get the distances t_2 - t_1 for all pairs of observations (t_1, t_2) in the test and train data within the same group.
    distances: np.ndarray = get_distances_to_train_observations(test_data, train_data, group_col, datetime_col)

    # Get sum of all leakage contributions for case 1: t_2 >= t_1
    leakage = np.exp(-distances[distances >= 0]).sum()
    # Add sum of all leakage contributions for case 2: t_2 < t_1
    leakage += alpha * np.exp(distances[distances < 0]).sum()  # => unnormalized data leakage value L

    # Get normalization factor, that is, the maximum leakage value for the extreme case train_data = test_data
    distances: np.ndarray = get_distances_to_train_observations(test_data, test_data, group_col, datetime_col)
    leakage_worst_case = np.exp(-distances[distances >= 0]).sum()
    leakage_worst_case += alpha * np.exp(distances[distances < 0]).sum()

    return leakage / leakage_worst_case  # => normalized data leakage value


def get_distances_to_train_observations(test_data: pd.DataFrame, train_data: pd.DataFrame, group_col: Optional[str],
                                        datetime_col: str) -> np.ndarray:
    """
    For each observation in the test_data, finds the distance (in days) to the closest observation with the same group value in the train_data.
    If there is no such observation, the distance is np.inf.
    :param test_data:  The test data with a date column, group column and optionally further feature columns.
    :param train_data: The training data with a date column, group column and optionally further feature columns.
    :param group_col: A column defining different groups of observations in the dataset. If None, all observations are considered to belong to the same group.
    :param datetime_col: The datetime column.
    :return: test_data with a new column day_diff.
    """
    test_data = test_data.reset_index(drop=False if test_data.index.name == datetime_col else True)
    train_data = train_data.reset_index(drop=False if train_data.index.name == datetime_col else True)
    test_data[datetime_col] = pd.to_datetime(test_data[datetime_col])
    train_data[datetime_col] = pd.to_datetime(train_data[datetime_col])

    merge_suffixes = ("_test", "_train")
    if group_col is None:
        data_merged = pd.merge(test_data.loc[:, [datetime_col]], train_data.loc[:, [datetime_col]], how="cross",
                               suffixes=merge_suffixes)
    else:
        data_merged = pd.merge(left=test_data.loc[:, [group_col, datetime_col]],
                               right=train_data.loc[:, [group_col, datetime_col]], on=group_col, how="inner",
                               suffixes=merge_suffixes)
    day_diffs = (data_merged[datetime_col + "_train"] - data_merged[datetime_col + "_test"]).dt.days
    distances = day_diffs.values

    return distances


def is_better_than(x: Optional[float], y: Optional[float], target: str | float | int) -> bool:
    if x is None:
        return False
    elif y is None:
        return True
    elif isinstance(target, str):
        return x > y if target == "max" else x < y
    else:
        return True if np.abs(x - target) < np.abs(y - target) else False


class CustomScorers:

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.mean((y_true - y_pred) ** 2, axis=axis)

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.mean(np.abs(y_true - y_pred), axis=axis)

    @staticmethod
    def absolute_mean_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.abs(np.mean(y_true - y_pred, axis=axis))

    @staticmethod
    def absolute_median_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.abs(np.median(y_true - y_pred, axis=axis))

    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.median(np.abs(y_true - y_pred), axis=axis)

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        eps = np.finfo(float).eps
        denom = np.sum((y_true - np.mean(y_true, axis=axis, keepdims=True)) ** 2, axis=axis)
        r2 = 1 - np.sum((y_true - y_pred) ** 2, axis=axis) / (denom + eps)
        return np.maximum(0, r2)

    @staticmethod
    def mean_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.mean(y_true - y_pred, axis=axis)

    @staticmethod
    def median_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
        return np.median(y_true - y_pred, axis=axis)

    @staticmethod
    def get_scorer(scorer_name: str, sign: Literal['+', '-']) -> Callable[
        [np.ndarray, np.ndarray, Optional[tuple | int]], np.ndarray | float]:
        if scorer_name == "rmse":
            f = CustomScorers.root_mean_squared_error
        elif scorer_name == "mse":
            f = CustomScorers.mean_squared_error
        elif scorer_name == "mae":
            f = CustomScorers.mean_absolute_error
        elif scorer_name == "ame":
            f = CustomScorers.absolute_mean_error
        elif scorer_name == "amde":
            f = CustomScorers.absolute_median_error
        elif scorer_name == "mdae":
            f = CustomScorers.median_absolute_error
        elif scorer_name == "r2":
            f = CustomScorers.r2_score
        elif scorer_name == "me":
            f = CustomScorers.mean_error
        elif scorer_name == "mde":
            f = CustomScorers.median_error
        else:
            raise ValueError(f"Unknown scorer: {scorer_name}")

        if sign == "-":
            return lambda y_true, y_pred, axis: - f(y_true, y_pred, axis)
        else:
            return f


def evaluate_on_test_set(estimator,
                         X_test: pd.DataFrame | pd.Series,
                         y_test: np.ndarray,
                         scoring: dict,
                         bootstrap_sample_size: Optional[int] = None,
                         test_set_bootstrap: bool = False,
                         bootstrap_indices: Optional[np.ndarray] = None) -> dict[str, list | np.ndarray]:
    """
    Evaluates a trained estimator on the given test set using the provided scoring metrics.
    Optionally, test set bootstrapping can be applied to obtain confidence intervals for the metrics.

    :param estimator: SKLearn-like estimator object. Predict function must accept DataFrames of Shape (n_samples, n_features).
    :param X_test: Input df of shape (n_samples, n_features)
    :param y_test: Target array of shape (n_samples,)
    :param scoring: Dict of scoring metrics to evaluate. Keys are the names of the metrics, values are tuples of (metric_function_name, sign).
    :param bootstrap_sample_size: Number of bootstrap samples to draw from the test set if test_set_bootstrap is True.
    :param test_set_bootstrap: Whether to apply test set bootstrapping.
    :param bootstrap_indices: Optional precomputed bootstrap indices to use. Helpful for comparability.
    :return: Results dict with keys defined by the scoring dict + total_inference_time, and lists of scoring metrics.
    """
    if test_set_bootstrap and bootstrap_sample_size is None:
        raise ValueError("bootstrap_samples cannot be None if test_set_bootstrap is True")
    bootstrap_sample_size = 1 if not test_set_bootstrap else bootstrap_sample_size

    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()
    X_test_values = X_test.values
    n_samples, n_features = X_test_values.shape

    if test_set_bootstrap:
        if bootstrap_indices is None:
            logging.info(f"[evaluate_on_test_set] No bootstrap indices provided -> generating bootstrap indices.")
            bootstrap_indices = np.random.randint(low=0, high=n_samples, size=(bootstrap_sample_size, n_samples))
        else:
            logging.info("[evaluate_on_test_set] Using provided bootstrap indices")
        if (bootstrap_indices.shape[0] != bootstrap_sample_size) or (bootstrap_indices.shape[1] != n_samples) or len(
                bootstrap_indices.shape) != 2:
            raise ValueError(
                f"[evaluate_on_test_set] Provided bootstrap_indices have the wrong shape. Required: ({bootstrap_sample_size}, {n_samples}) | Provided: {bootstrap_indices.shape}")

        logging.debug(f"[evaluate_on_test_set] Start generating bootstrap samples.")

        # Gather bootstrap samples using advanced indexing
        X_test_samples = X_test_values[bootstrap_indices]  # => shape (bootstrap_samples, n_samples, n_features)
        y_test_samples = y_test[bootstrap_indices]  # => shape (bootstrap_samples, n_samples)
        logging.debug(
            f"[evaluate_on_test_set] Generating bootstrap samples done! Shapes: X.shape={X_test_samples.shape}, y.shape={y_test_samples.shape}")
    else:
        # Just reshape the data
        X_test_samples = X_test_values.reshape(1, n_samples, n_features)
        y_test_samples = y_test.reshape(1, -1)
    X_test_samples_reshaped2D = X_test_samples.reshape((bootstrap_sample_size * n_samples, n_features))

    with warnings.catch_warnings():
        # Ignore warning that we predict on ndarray instead of df here
        warnings.simplefilter("ignore", category=UserWarning)
        logging.debug(
            f"[evaluate_on_test_set] Start predicting.")
        start_time = time.perf_counter()
        y_pred_samples = estimator.predict(X_test_samples_reshaped2D)  # => shape (bootstrap_samples * n_samples, )
        end_time = time.perf_counter()
    inference_time = end_time - start_time
    logging.info(
        f"[evaluate_on_test_set] Prediction finished in {inference_time} seconds.")
    y_pred_samples = y_pred_samples.reshape(y_test_samples.shape)  # => shape (bootstrap_samples, n_samples)

    results = {'duration': [inference_time]}
    logging.debug(
        f"[evaluate_on_test_set] Start calculating scores.")
    for scorer_name, (metric_func_name, sign) in scoring.items():
        results[scorer_name] = CustomScorers.get_scorer(metric_func_name, sign)(y_test_samples, y_pred_samples,
                                                                                1)  # => score arrays of shape (bootstrap_samples, )
    logging.debug(
        f"[evaluate_on_test_set] Calculation of scores done!")

    return results
