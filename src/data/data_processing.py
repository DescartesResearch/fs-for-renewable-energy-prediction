import logging

from sklearn.base import BaseEstimator, TransformerMixin
import fnmatch
import hashlib
import io
import json
import pickle
import subprocess
import zipfile
from typing import Optional, Any
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataUtils:
    """
    A collection of utility functions (static methods) for data processing.
    """

    @staticmethod
    def remove_overlaps(arrs):
        seen = set()
        out = []
        for a in arrs:
            mask = ~np.isin(a, list(seen))
            b = a[mask]
            out.append(b)
            seen.update(b.tolist())
        return out

    @staticmethod
    def get_time_split_indices(
            datetimes: pd.Series,
            gap: Optional[pd.Timedelta],  # e.g. "1D", "12H", pd.Timedelta, or 0/None
            split_ratio: tuple[float, float] | tuple[float, float, float],
    ) -> list[np.ndarray]:
        """
        Split a Series of timestamps into train/test by a split point with a symmetric gap.
        Returns integer positions compatible with .iloc.

        Rules:
          - The gap is centered on `split_at`. Anything in [split_at - gap/2, split_at + gap/2]
            is excluded from both sets.
          - If gap is 0/None: train = t < split_at, test = t >= split_at.
          - Timestamps equal to the split fall into test when gap is 0.
        """
        if not isinstance(datetimes, pd.Series):
            raise TypeError("`datetimes` must be a pandas Series.")

        if sum(split_ratio) != 1.0:
            raise ValueError("`split_ratio` must sum up to 1.0.")

        if gap is None:
            gap = 0
        gap = pd.to_timedelta(gap)

        min_datetime = datetimes.min()
        max_datetime = datetimes.max()
        total_timedelta = max_datetime - min_datetime
        split_timedeltas = pd.Series([p * total_timedelta for p in split_ratio]).cumsum()
        assert np.abs(split_timedeltas.iloc[-1] - total_timedelta) < pd.Timedelta(seconds=1)
        split_datetimes = [min_datetime] + [min_datetime + t for t in split_timedeltas]
        assert np.abs(split_datetimes[-1] - max_datetime) < pd.Timedelta(seconds=1)
        # split_datetimes = split_datetimes[:len(split_ratio)-1]

        indices = []
        for i in range(len(split_datetimes) - 1):
            left = split_datetimes[i] + (gap / 2 if i > 0 else pd.to_timedelta(0))
            right = split_datetimes[i + 1] - (gap / 2 if i < len(split_datetimes) - 2 else pd.to_timedelta(0))
            mask = (left <= datetimes) & (datetimes <= right)
            indices.append(np.flatnonzero(mask.to_numpy()))

        return DataUtils.remove_overlaps(indices)

    @staticmethod
    def set_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the column types of the dataframe to the correct types.
        :param df: Pandas DataFrame with HDD data. Considers the columns "date", "failure", "serial_number", "model", "datacenter" and all SMART columns. Each of them is optional.
        :return: Pandas DataFrame with the correct column types.
        """
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "failure" in df.columns:
            df["failure"] = df["failure"].astype("bool")
        if "serial_number" in df.columns:
            df["serial_number"] = df["serial_number"].astype("category")
        if "model" in df.columns:
            df["model"] = df["model"].astype("category")
        if "datacenter" in df.columns:
            df["datacenter"] = df["datacenter"].astype("category")
        for smart_col in DataUtils.get_smart_cols(df):
            df[smart_col] = df[smart_col].astype("float64")
        return df

    @staticmethod
    def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast float columns to float32.
        :param df: Pandas DataFrame with HDD data.
        :return: Pandas DataFrame with float columns downcasted to float32.
        """
        for col in df.select_dtypes(include="float64").columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        return df

    @staticmethod
    def get_smart_cols(data: pd.DataFrame, raw_only: bool = False) -> list[str]:
        """
        Get all SMART columns present in the given Pandas DataFrame.
        :param data: Pandas DataFrame
        :return: List of SMART columns
        """
        return [col for col in data.columns if col.startswith("smart") & (not raw_only or col.endswith("raw"))]

    @staticmethod
    def get_num_smart_cols(data: pd.DataFrame) -> int:
        """
        Get number of SMART columns present in the given Pandas DataFrame.
        :param data: Pandas DataFrame
        :return: Number of SMART columns
        """
        return len(DataUtils.get_smart_cols(data))

    @staticmethod
    def get_labels(data: pd.DataFrame, num_classes: int, binary_threshold: int = 7) -> np.ndarray:
        """
        :param data: Pandas Dataframe with a RUL column named "rul".
        :param num_classes: How many classes?
        :param binary_threshold: Only relevant for num_classes==2. RUL values <= binary_threshold will be assigned
        the positive label 1, otherwise 0.
        :return: The labels only as a numpy array.
        """
        if num_classes == 2:
            labels = np.zeros(data.shape[0])
            labels[data["rul"].values <= binary_threshold] = 1
        else:
            labels = pd.qcut(data["rul"], q=num_classes, labels=False)
        return labels

    @staticmethod
    def get_disks_with_failures(data: pd.DataFrame) -> list[str]:
        """
        Get the serial numbers of disks that show failure in the given data.
        :param data: Pandas DataFrame with HDD data. Must contain a column "failure" with boolean values and a column "serial_number".
        :return: List of serial numbers of disks that show failure.
        """
        serial_numbers_with_failures = data[data.failure == True]["serial_number"].unique()
        return list(serial_numbers_with_failures)

    @staticmethod
    def get_disks_without_failures(data: pd.DataFrame) -> list[str]:
        """
        Get the serial numbers of disks that show no failure in the given data.
        :param data: Pandas DataFrame with HDD data. Must contain a column "failure" with boolean values and a column "serial_number".
        :return: List of serial numbers of disks that show no failure.
        """
        result = set(data["serial_number"].unique()) - set(DataUtils.get_disks_with_failures(data))
        return list(result)

    @staticmethod
    def df_to_series(data: pd.DataFrame, lags: list[int] | tuple[int]) -> np.ndarray:
        """
        Add lags to the SMART columns of the given DataFrame and convert it to a series, that is, a 3D numpy array of shape (n_samples, seq_len, n_features).
        :param data: Pandas DataFrame with HDD data. Must contain SMART columns and "serial_number" column.
        :param lags: List (or tuple) of lags to be added.
        :return: 3D numpy array of shape (n_samples, seq_len, n_features) with lags for the SMART columns added.
        """
        if isinstance(lags, tuple):
            lags = list(lags)
        features = DataUtils.get_smart_cols(data)
        data_lagged = DataFrameProcessor.add_lags(lags=lags, cols=features)(data)
        data_lagged = data_lagged.loc[:, DataUtils.get_smart_cols(data_lagged)]
        seq_len = len(data_lagged.columns) // len(features)

        # Check if the number of columns in the lagged data is as expected
        if seq_len != len(lags) + 1:
            raise ValueError(
                f"The number of columns in the lagged data is not as expected. Expected: {len(lags) + 1}, Actual: {seq_len}. The number of features is {len(features)}. The number of specified lags is {len(lags)}.")
        if len(data_lagged.columns) % len(features) != 0:
            raise ValueError(
                f"The number of columns in the lagged data is not a multiple of the number of features. Number of columns: {len(data_lagged.columns)}, Number of features: {len(features)}")

        # Make sure the columns are in the right order
        col_idx = 0
        for lag in [0] + lags:
            for feature in features:
                expected_colname = feature + (f"_lag_{lag}" if lag > 0 else "")
                if expected_colname != data_lagged.columns[col_idx]:
                    raise ValueError(
                        f"Column {col_idx} in the lagged data is not as expected. Expected: {expected_colname}, Actual: {data_lagged.columns[col_idx]}.")
                col_idx += 1

        result = data_lagged.values.reshape(data_lagged.shape[0], seq_len, len(features))
        return result

    @staticmethod
    def get_config_hash(config: dict) -> str:
        """
        Get a hash value for the given configuration dictionary.
        :param config: Configuration dictionary.
        :return: Hash value as a string.
        """
        config_str = json.dumps({k: v for k, v in config.items() if k != "loaded_objects"}, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()

    @staticmethod
    def get_git_hash() -> Optional[str]:
        """
        Get the hash of the current git commit.
        :return: Hash of the current git commit as a string or None if the hash could not be retrieved.
        """
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            # return subprocess.check_output(['git', 'log', '-1', '--pretty=format:%H', '--', file_path]).strip().decode('utf-8') # Alternative for specific file
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def save_dataset(dataset_dict: dict, config_hash: str, git_hash: str) -> None:
        """
        Cache the dataset dictionary to a pickle file with name "<config_hash>_<git_hash>.pkl".
        :param dataset_dict: Dataset dictionary to be cached.
        :param config_hash: Hash value of the configuration.
        :param git_hash: Hash value of the current git commit.
        :return: None
        """
        filename = f"{config_hash}_{git_hash}.pkl"
        filepath = Paths.DATA_CACHE / filename
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_dict, f)

    @staticmethod
    def load_dataset(config_hash: str, git_hash: str):
        """
        Load the dataset dictionary from a pickle file with name "<config_hash>_<git_hash>.pkl".
        :param config_hash: Hash value of the configuration.
        :param git_hash: Hash value of the current git commit.
        :return: Dataset dictionary or None if the file does not exist.
        """
        filename = f"{config_hash}_{git_hash}.pkl"
        filepath = Paths.DATA_CACHE / filename
        if filepath.is_file():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

    @staticmethod
    def get_smart_features_list(features: str | list[str]):
        """
        Get a list of SMART features based on the specified feature set. There are a couple of predefined feature sets
        available, for example features used in specific papers or specific vendors. "all" includes all features.
        Single features can be specified as well. The features are returned in sorted order.
        It is made sure that no duplicates are present in the list.
        :param features: Feature set name or list of feature names. For example "all" or ["smart_1_raw", "smart_5_raw"].
        :return: Sorted list of SMART features without duplicates.
        """
        features = [features] if isinstance(features, str) else features
        result = set()

        for f in features:
            if f == "microsoft":
                result.update(["smart_{}_raw".format(i) for i in [1, 4, 5, 7, 9, 12, 187, 193, 194, 197, 199]])
            elif f == "tencent":
                result.update(["smart_{}_raw".format(i) for i in [1, 3, 5, 7, 9, 10, 12, 187, 192, 194, 196, 197, 198]])
            elif f == "backblaze":
                result.update(["smart_{}_raw".format(i) for i in [5, 187, 188, 197, 198]])
            elif f == "all_raw":
                result.update(["smart_{}_raw".format(i) for i in range(1, 256)])
            elif f == "all_normalized":
                result.update(["smart_{}_normalized".format(i) for i in range(1, 256)])
            elif f == "all":
                result.update(["smart_{}_normalized".format(i) for i in range(1, 256)])
                result.update(["smart_{}_raw".format(i) for i in range(1, 256)])
            elif f == "custom0":
                result.update(["smart_{}_raw".format(i) for i in [5, 7, 184, 187, 188, 9, 4, 12, 193, 194]])
            elif f == "paper":
                smart_ids = list({1, 9, 10, 3, 12, 4, 5, 7, 192, 193, 194, 197, 198, 199,  # Lu et al.
                                  1, 3, 5, 7, 187, 189, 194, 195, 197})  # De Santo et al.
                result.update(["smart_{}_raw".format(i) for i in smart_ids] +
                              ["smart_{}_normalized".format(i) for i in smart_ids])
            else:
                result.add(f)

        return sorted(list(result))


class DataFrameProcessor:
    """
    Collection of functions for DataFrame processing.
    Each function returns a callable that takes a Pandas DataFrame as input and returns a Pandas DataFrame as output
    (except for get_values, which returns a numpy array).
    """

    @staticmethod
    def _add_lags(data: pd.DataFrame, lags: list[int], cols: list[str] | str) -> pd.DataFrame:
        if isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols)
        group = data.loc[:, ["serial_number"] + cols].groupby("serial_number", observed=True)
        lagged_series = dict()
        for lag in lags:
            shifted = group.shift(lag)
            for col in cols:
                lagged_series[f"{col}_lag_{lag}"] = shifted[col]
        return pd.concat([data, pd.DataFrame(lagged_series)], axis=1)

    @staticmethod
    def add_lags(lags: list[int], cols: list[str] | str) -> callable:
        """
        Add lagged features to a Pandas DataFrame.
        The lags are added for groups defined by "serial_number".
        :param lags: List of lags to be added.
        :param cols: List of columns for which the lags are added.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the specified lags added as columns with the name "<colname>_lag_<lag>". The DataFrame must be sorted by "serial_number" and "date".
        """
        return lambda data: DataFrameProcessor._add_lags(data, lags, cols)

    @staticmethod
    def add_lags_to_smart_cols(lags: list[int]) -> callable:
        """
        Add lagged SMART features to a Pandas DataFrame. The lags are added for groups defined by "serial_number".
        :param lags: List of lags to be added.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the specified lags added as columns with the name "<colname>_lag_<lag>". The DataFrame must be sorted by "serial_number" and "date".
        """
        return lambda data: DataFrameProcessor._add_lags(data, lags, DataUtils.get_smart_cols(data))

    @staticmethod
    def _scale(data: pd.DataFrame, cols: list[str], scaler: TransformerMixin | BaseEstimator) -> pd.DataFrame:
        # If BaseEstimator is already fitted, use transform, else fit_transform
        if hasattr(scaler, 'n_features_in_'):
            # print("Calling transform")
            data.loc[:, cols] = scaler.transform(data.loc[:, cols])
        else:
            # print("Calling fit_transform")
            data.loc[:, cols] = scaler.fit_transform(data.loc[:, cols])
        return data

    @staticmethod
    def scale(cols: list[str] | str, scaler: TransformerMixin | BaseEstimator) -> callable:
        """
        Normalize columns of a Pandas DataFrame
        :param cols: List of columns to normalize
        :param scaler: Scikit-learn transformer object
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the specified columns normalized
        """
        return lambda data: DataFrameProcessor._scale(data, cols, scaler)

    @staticmethod
    def drop_na_columns(p_thresh: float) -> callable:
        """
        Only keep columns where at least p_thresh * len(data) values aren't NaN (axis=1)
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with NaN columns removed
        """
        return lambda data: DataFrameProcessor._drop_na_columns(data, p_thresh)

    @staticmethod
    def _drop_na_columns(data: pd.DataFrame, p_thresh: float) -> pd.DataFrame:
        cleaned_data = data.dropna(axis=1, thresh=int(p_thresh * len(data)))
        col_diff = data.columns.difference(cleaned_data.columns)
        logging.info(f"Dropped {len(col_diff)} NA columns at p_thresh={p_thresh}")
        if not col_diff.empty:
            logging.info(f"Dropped NA cols: {col_diff}")
        return cleaned_data

    @staticmethod
    def drop_na_rows() -> callable:
        """
        Drop rows where any value is NaN
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with NaN rows removed
        """
        return lambda data: DataFrameProcessor._drop_na_rows(data)

    @staticmethod
    def _drop_na_rows(data: pd.DataFrame) -> pd.DataFrame:
        cleaned_data = data.dropna(axis=0, how="any")
        na_row_count = len(data) - len(cleaned_data)
        logging.info(f"Dropped {na_row_count} NA rows")
        return cleaned_data

    @staticmethod
    def drop_inf_rows(cols: Optional[list[str]] = None) -> callable:
        """
        Drop rows where any value is +inf or -inf
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with NaN rows removed
        """
        return lambda data: DataFrameProcessor._drop_inf_rows(data, cols)

    @staticmethod
    def _drop_inf_rows(data: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
        if cols is not None:
            data.loc[:, cols] = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
        else:
            data = data.replace([np.inf, -np.inf], np.nan)
        return DataFrameProcessor._drop_na_rows(data)

    @staticmethod
    def drop_columns(columns: list[str] | str) -> callable:
        """
        Drop columns of a Pandas DataFrame
        :param columns: List of columns to drop
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the specified columns dropped
        """
        if isinstance(columns, str):
            columns = [columns]
        return lambda data: data.drop(columns=columns)

    @staticmethod
    def filter_columns(columns: list[str]) -> callable:
        """
       Filter columns of a Pandas DataFrame
       :param columns: List of columns to keep. Columns not present in the dataframe are ignored
       :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the specified columns
       """
        return lambda data: data.loc[:, data.columns.intersection(columns)]

    @staticmethod
    def filter_rows(condition: pd.Series | np.ndarray | list) -> callable:
        """
        Filter rows of a Pandas DataFrame.
        :param condition: A pandas series, numpy array or list of booleans.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the selected rows.
        """
        if isinstance(condition, pd.Series) or isinstance(condition, np.ndarray) or isinstance(condition, list):
            return lambda data: data.loc[condition, :]
        else:
            raise ValueError("condition must be a pandas series, numpy array or list of booleans.")

    @staticmethod
    def _remove_low_variance_columns(data: pd.DataFrame, variance_threshold: float,
                                     cols: Optional[list[str]] = None) -> pd.DataFrame:
        selected_data = DataFrameProcessor.filter_columns(cols)(data)
        columns_to_drop = selected_data.columns[selected_data.var() <= variance_threshold]
        cleaned_data = data.drop(columns=columns_to_drop)
        logging.info(f"Removed {len(columns_to_drop)} columns with variance threshold {variance_threshold}")
        logging.info(f"Removed columns: {columns_to_drop}")
        return cleaned_data

    @staticmethod
    def remove_low_variance_columns(variance_threshold: float, cols: Optional[list[str]] = None) -> callable:
        """
        Remove columns with zero variance
        :param variance_threshold: Columns with variance <= variance_threshold will be removed.
        :param cols: List of columns to check for zero variance. Other columns are not affected. If None, all columns are checked.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with columns of zero variance removed
        """
        return lambda data: DataFrameProcessor._remove_low_variance_columns(data, variance_threshold,
                                                                            cols if cols is not None else data.columns.tolist())

    @staticmethod
    def remove_zero_variance_columns(cols: Optional[list[str]] = None) -> callable:
        """
        Remove columns with zero variance
        :param cols: List of columns to check for zero variance. Other columns are not affected. If None, all columns are checked.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with columns of zero variance removed
        """
        return lambda data: DataFrameProcessor._remove_low_variance_columns(data, 0,
                                                                            cols if cols is not None else data.columns.tolist())

    @staticmethod
    def get_values() -> callable:
        """
        Get the values of a Pandas DataFrame
        :return: Callable: data: pd.DataFrame -> The values of the Pandas DataFrame as a numpy array
        """
        return lambda data: data.values

    @staticmethod
    def _add_rul(data: pd.DataFrame, max_rul_threshold: Optional[int] = None,
                 cap_value: Optional[int] = None, inf_sample_size: Optional[float] = None) -> pd.DataFrame:
        if max_rul_threshold is None:
            max_rul_threshold = np.inf
        if inf_sample_size is None:
            inf_sample_size = 1.
        data.loc[:, "date"] = pd.to_datetime(data.loc[:, "date"])
        # get remaining useful life (RUL) for each hard drive, set to Inf if no failure occurred
        data.loc[:, "rul"] = data.loc[:, ["serial_number", "date"]].groupby("serial_number", observed=True).transform(
            lambda x: (x.max() - x).dt.days + 1.0
        ).rename(columns={"date": "rul"}).loc[:, "rul"]

        serials_with_failures = DataUtils.get_disks_with_failures(data)
        serials_without_failures = DataUtils.get_disks_without_failures(data)
        if inf_sample_size != 1.:
            if 0 < inf_sample_size < 1:
                # get random sample of size = inf_sample_size * #serials without failures
                serials_without_failures = np.random.choice(serials_without_failures,
                                                            int(inf_sample_size * len(serials_without_failures)))
            elif inf_sample_size > 1:
                serials_without_failures = np.random.choice(serials_without_failures, int(inf_sample_size))
            elif inf_sample_size == 0:
                serials_without_failures = []
            else:
                raise ValueError("inf_sample_size must be >= 0")

            data = data.loc[
                data.serial_number.isin(serials_with_failures) | data.serial_number.isin(
                    serials_without_failures), :]

        never_failing_hdds_rows = data.loc[:, "serial_number"].isin(serials_without_failures)
        data.loc[never_failing_hdds_rows, "rul"] = np.inf
        if max_rul_threshold is not None:
            # Filter data with RUL less than or equal to max_rul_threshold
            data = data.loc[(data.rul <= max_rul_threshold) | never_failing_hdds_rows, :]
        if cap_value is not None:
            # Cap RUL values to cap_value
            data.loc[(data.rul > cap_value) & (~never_failing_hdds_rows), "rul"] = cap_value
        return data

    @staticmethod
    def add_rul(max_rul_threshold: int = None,
                cap_value: int = None, inf_sample_size: float | None = None) -> callable:
        """
        Add a column named "rul" to a Pandas DataFrame containing the remaining useful life (RUL) at each
        point in time and for each specific hard drive. For hard drives that never failed, the RUL is set to np.inf.
        :param max_rul_threshold: Maximum RUL value to keep. RUL values greater than this value are discarded (except
        for hard drives that never failed with RUL set to np.inf). None means no threshold.
        :param cap_value: RUL values larger than cap_value (except for hard drives that never failed with RUL=np.inf)
        are set to cap_value. None is equivalent to np.inf. None means no cap.
        :param inf_sample_size: Must be >= 0. For values > 1, number of hard drives that never failed to keep.
        Otherwise, fraction of hard drives that never failed to keep. None is equivalent to 1.

        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the "rul" column added.
        """
        return lambda data: DataFrameProcessor._add_rul(data, max_rul_threshold, cap_value, inf_sample_size)

    @staticmethod
    def _add_binary_labels(data: pd.DataFrame, threshold: int = 7, discard_rul: bool = True) -> pd.DataFrame:
        labels = DataUtils.get_labels(data, num_classes=2, binary_threshold=threshold)
        data.loc[:, "label"] = labels
        if discard_rul:
            data = data.drop(columns=["rul"])
        return data

    @staticmethod
    def add_binary_labels(threshold: int = 7, discard_rul: bool = True) -> callable:
        """
        Add binary labels to a Pandas DataFrame in the form of a column named "label".
        :param threshold: RUL threshold for binary classification. Observations with RUL <= threshold are labeled as 1,
        otherwise as 0.
        :param discard_rul: If True, discard the "rul" column after adding the labels.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the binary labels added.
        """
        return lambda data: DataFrameProcessor._add_binary_labels(data, threshold, discard_rul)

    @staticmethod
    def _discard_disks_wo_failure(data: pd.DataFrame) -> pd.DataFrame:
        serial_numbers_with_failures = data.loc[data.failure == True, "serial_number"].unique()
        data = data.loc[data.serial_number.isin(serial_numbers_with_failures), :]
        return data

    @staticmethod
    def discard_disks_wo_failure() -> callable:
        """
        Discard hard drives that never failed.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the hard drives that failed.
        """
        return lambda data: DataFrameProcessor._discard_disks_wo_failure(data)

    @staticmethod
    def only_keep_smart_cols(also_keep: list[str] | str = None, raw_only: bool = False) -> callable:
        """
        Only keep SMART columns (and selected other columns) in a Pandas DataFrame.
        :param also_keep: (List of) additional column(s) to keep. If not in the DataFrame, the column is ignored.
        :param raw_only: If True, only keep raw SMART columns. Otherwise, keep both raw and normalized SMART columns.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the SMART columns and the
        selected additional columns.
        """
        if also_keep is None:
            also_keep = []
        elif isinstance(also_keep, str):
            also_keep = [also_keep]
        return lambda data: DataFrameProcessor.filter_columns(DataUtils.get_smart_cols(data, raw_only=raw_only)
                                                              + also_keep)(data)

    @staticmethod
    def filter_serial_numbers(serial_numbers: list[str] | str) -> callable:
        """
        Discard all rows with serial numbers not in the provided list.
        :param serial_numbers: (List of) serial number(s) to keep.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with only the selected serial number(s).
        """
        if isinstance(serial_numbers, str):
            serial_numbers = [serial_numbers]
        return lambda data: data.loc[data.serial_number.isin(serial_numbers), :]

    @staticmethod
    def drop_duplicates() -> callable:
        """
        Drop duplicate rows in a Pandas DataFrame.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with duplicate rows removed.
        """
        return lambda data: DataFrameProcessor._drop_duplicates(data)

    @staticmethod
    def _drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        deduplicated_df = data.drop_duplicates()
        duplicate_count = len(data) - len(deduplicated_df)
        logging.info(f"Dropping {duplicate_count} duplicates.")
        return deduplicated_df

    @staticmethod
    def _equidistant_daterange_imputation(data: pd.DataFrame, date_col: str = "date", group_col: str = "serial_number",
                                          freq: str = "D") -> pd.DataFrame:
        # Ensure the DataFrame is properly sorted by 'serial_number' and 'date' without altering the original DataFrame
        if not data[group_col].is_monotonic_increasing:
            raise ValueError(f"The values in column {group_col} are not sorted")
        # for each group, check that the dates are sorted
        if not data.groupby(group_col, observed=True)[date_col].is_monotonic_increasing.all():
            raise ValueError(f"The values in column {date_col} are not sorted for each group")

        # Function to create a complete date range for each group
        def reindex_group(group):
            # if group.empty:  # only needed if observed is set to False in groupby
            #     return group
            # Create a complete date range from the min to max date
            group_identifier = group[group_col].iloc[0]  # e.g. the serial number
            date_range = pd.date_range(start=group[date_col].min(), end=group[date_col].max(), freq=freq)
            reindexed_group = group.set_index(date_col).reindex(date_range).reset_index().rename(
                columns={'index': date_col})
            reindexed_group.loc[:, group_col] = group_identifier
            return reindexed_group

        # Apply the function to each group
        data = data.groupby(group_col, observed=True).apply(reindex_group).reset_index(drop=True)
        return data

    @staticmethod
    def equidistant_daterange_imputation(date_col: str = "date", group_col: str = "serial_number",
                                         freq: str = "D") -> callable:
        """
        Impute missing dates in a Pandas DataFrame by creating an equidistant date range for each group.
        The remaining columns for imputed dates are filled with NaN.
        The Pandas DataFrame must be sorted by 'group_col' and 'date', otherwise the function will raise an error.
        :param date_col: Name of the date column.
        :param group_col: Name of the group column.
        :param freq: Frequency string of the date range. Default is 'D' for daily.
        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with the missing dates imputed.
        """
        return lambda data: DataFrameProcessor._equidistant_daterange_imputation(data,
                                                                                 date_col=date_col,
                                                                                 group_col=group_col,
                                                                                 freq=freq)

    @staticmethod
    def _add_cyclical_encodings(
            df: pd.DataFrame,
            feature_tags: dict,
            value_ranges: dict,  # dict: {feature_name: (min_val, max_val)}
    ) -> pd.DataFrame:
        df = df.copy()

        # Get circular features
        circular_features = [
            feat for feat, tags in feature_tags.items()
            if "circular" in tags
        ]

        added_features = []
        for feature in circular_features:
            values = df[feature]
            # # Use custom ranges
            # if feature not in value_ranges, raise an error
            min_val, max_val = value_ranges[feature]
            scaled = 2 * np.pi * (values - min_val) / (max_val - min_val)
            df[f"{feature}_sin"] = np.sin(scaled)
            df[f"{feature}_cos"] = np.cos(scaled)
            # feature_tags[f"{feature}_sin"] = {"cyclical_encoding"}
            # feature_tags[f"{feature}_cos"] = {"cyclical_encoding"}
            added_features.extend([
                f"{feature}_sin",
                f"{feature}_cos"
            ])

        logging.info(f"Added {len(added_features)} features: {added_features}")

        return df

    @staticmethod
    def add_cyclical_encodings(feature_tags: dict, value_ranges: dict) -> callable:
        """
        Adds sine and cosine components for all circular features, including optional time components.

        :param df: pd.DataFrame -> The input DataFrame to which circular features will be added
        :param feature_tags: dict -> A mapping from feature names to their associated tag sets
        :param timestamp_feature: str -> The name of the column containing timestamp information
        :param time_features_to_include: tuple -> The set of time-based features to extract (e.g., "hour_of_day", "day_of_year")
        :param value_ranges: dict -> A dictionary specifying custom min/max ranges for circular features, formatted as {"FeatureName": (min, max)}

        :return: Callable: data: pd.DataFrame -> The resulting Pandas DataFrame with new circular features added
        """
        return lambda data: DataFrameProcessor._add_cyclical_encodings(data,
                                                                       feature_tags=feature_tags,
                                                                       value_ranges=value_ranges)

    @staticmethod
    def sort(by_columns: list[str] | str, ascending: bool) -> callable:
        logging.info(f"Sorting by columns: {by_columns}")
        return lambda df: df.sort_values(by=by_columns, ascending=ascending).reset_index(drop=True)


class DataFrameProcessingDecorators:
    """
    Decorators for processing Pandas DataFrames.
    Example usage:
    @DataFrameProcessingDecorators.apply_to_all(lambda df: df.dropna())
    def func(df: pd.DataFrame, other_arg: int, another_arg: str, df2: pd.DataFrame):
        # Function requiring DataFrames with no NaN values
        # Do something with the DataFrames
        return df, df2
    """

    @staticmethod
    def apply_to_all(filter_func: callable):
        """
        :param filter_func: Accepts a Pandas DataFrame and returns a Pandas DataFrame.
        :return: Decorator that applies the filter_func to all Pandas DataFrames in the arguments and keyword arguments.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                filtered_args = [filter_func(arg) if isinstance(arg, pd.DataFrame) else arg for arg in args]
                filtered_kwargs = {key: filter_func(value) if isinstance(value, pd.DataFrame) else value for key, value
                                   in
                                   kwargs.items()}
                return func(*filtered_args, **filtered_kwargs)

            return wrapper

        return decorator

    @staticmethod
    def apply_to_each(*filter_funcs: callable):
        """
        Applies the filter_funcs to the pandas dataframes in args and kwargs in the order they are passed. Important:
        The number of filter functions must match the number of DataFrames in the arguments and keyword.

        :param filter_funcs: List of filter functions. Each filter function should accept a Pandas DataFrame and return
        a Pandas DataFrame.

        :return: Decorator that applies the filter_funcs to the Pandas DataFrames in the arguments and keyword
        arguments.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                filter_func_idx = 0
                filtered_args = []
                filtered_kwargs = {}
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        filtered_args.append(filter_funcs[filter_func_idx](arg))
                        filter_func_idx += 1
                    else:
                        filtered_args.append(arg)
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        filtered_kwargs[key] = filter_funcs[filter_func_idx](value)
                        filter_func_idx += 1
                    else:
                        filtered_kwargs[key] = value

                if filter_func_idx != len(filter_funcs):
                    raise ValueError(
                        "Not all filter functions were used. Make sure that the number of filter functions "
                        "matches the number of DataFrames in the arguments and keyword arguments.")

                return func(*filtered_args, **filtered_kwargs)

            return wrapper

        return decorator


class ProcessingPipeline:
    """
    Class for creating processing pipelines for Pandas DataFrames. The processing pipeline consists of a chain of
    functions that are applied to the Pandas DataFrames in the order they are added to the pipeline. Each of the
    functions must accept a Pandas DataFrame as input and return a Pandas DataFrame as output. The pipeline can be
    called with the Pandas DataFrames as arguments and keyword arguments. The pipeline can also return intermediate
    results if desired.

    Usage example:
    pipeline = ProcessingPipeline().apply_to_all(func0).apply_to_each(func1, func2)
    df1, df2 = pipeline(df1, df2)

    This is equivalent to applying func0 to both dataframes and then applying func1 to df1 and func2 to df2.
    """

    def __init__(self):
        """
        Creates an empty processing pipeline.
        """
        self.chain = []
        self.chain_func_names = []

    def apply_to_all(self, func) -> "ProcessingPipeline":
        """
        Add a function to the processing pipeline that is applied to all Pandas DataFrames.
        :param func: Function that accepts a Pandas DataFrame and returns a Pandas DataFrame.
        :return: The ProcessingPipeline object with the function added to the chain.
        """
        self.chain.append(DataFrameProcessingDecorators.apply_to_all(func)(self._identity))
        self.chain_func_names.append(func.__name__)
        return self

    def apply(self, func) -> "ProcessingPipeline":
        """
        Add a function to the processing pipeline that is applied to all Pandas DataFrames.
        :param func: Function that accepts a Pandas DataFrame and returns a Pandas DataFrame.
        :return: The ProcessingPipeline object with the function added to the chain.
        """
        return self.apply_to_all(func)

    def apply_to_each(self, *funcs) -> "ProcessingPipeline":
        """
        Add multiple functions to the processing pipeline that are applied once each to the Pandas DataFrames in the order they are passed.
        :param funcs: List of functions. Each function should accept a Pandas DataFrame and return a Pandas DataFrame.
        :return: The ProcessingPipeline object with the functions added to the chain.
        """
        self.chain.append(DataFrameProcessingDecorators.apply_to_each(*funcs)(self._identity))
        self.chain_func_names.append([f.__name__ for f in funcs])
        return self

    def __call__(self, *args, return_intermediate_results=False, **kwargs):
        result_list = []  # for intermediate results
        curr_args = (*args, *kwargs.values())
        for func in self.chain:
            curr_args = func(*curr_args)
            if return_intermediate_results:
                result_list.append(curr_args[0] if len(curr_args) == 1 else curr_args)
        if return_intermediate_results:
            result = result_list
        elif len(curr_args) == 1:
            result = curr_args[0]
        else:
            result = curr_args
        return result

    def __str__(self):
        assert len(self.chain_func_names) == len(self.chain)
        return str(self.chain_func_names)

    @staticmethod
    def _identity(*args, **kwargs):
        return *args, *kwargs.values()
