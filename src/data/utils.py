from typing import Optional

import pandas as pd
import numpy as np
import datetime
import logging
from pathlib import Path

from openoa.utils.met_data_processing import compute_turbulence_intensity

from config.constants import Paths
from data.pvodataset import PVODataset
from astral.sun import sun
from astral import Observer
from zoneinfo import ZoneInfo
from pvlib.location import lookup_altitude
from pvlib.location import Location


def dst_fix(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Subtracts one hour from the timestamps that are in DST (for 2016 and 2017 only!)
    """
    dst_mask = (df[timestamp_col] >= pd.Timestamp("2016-03-27 3:00:00", tz="UTC")) & (
            df[timestamp_col] < pd.Timestamp("2016-10-30 2:00:00", tz="UTC"))
    dst_mask |= (df[timestamp_col] >= pd.Timestamp("2017-03-26 3:00:00", tz="UTC")) & (
            df[timestamp_col] < pd.Timestamp("2017-10-29 2:00:00", tz="UTC"))
    df.loc[dst_mask, timestamp_col] -= datetime.timedelta(hours=1)
    return df


def duplicate_fix(df: pd.DataFrame, timestamp_col: str, id_col: str) -> pd.DataFrame:
    dst_idxs = []
    for turbine_id in df[id_col].unique():
        curr_turbine_df = df.loc[df.Turbine_ID == turbine_id]
        # curr_turbine_df = dst_fix(curr_turbine_df, timestamp_col)
        duplicate_datetimes = curr_turbine_df[curr_turbine_df[[timestamp_col, id_col]].duplicated()][timestamp_col]

        prev_row = curr_turbine_df.loc[
            curr_turbine_df.Timestamp == duplicate_datetimes.iloc[0] - datetime.timedelta(hours=1, minutes=10)]
        next_row = curr_turbine_df.loc[
            curr_turbine_df.Timestamp == duplicate_datetimes.iloc[-1] + datetime.timedelta(hours=1, minutes=10)]
        for i in range(len(duplicate_datetimes) // 2):
            assert prev_row.shape[0] == 1
            dt_0 = duplicate_datetimes.iloc[i]
            dup_0 = curr_turbine_df.loc[curr_turbine_df.Timestamp == dt_0]
            assert dup_0.shape[0] <= 2
            prev_row_idx = np.round(
                (np.abs(dup_0.values[:, 2:] - prev_row.values[0, 2:]).argmin(axis=0).mean())).astype(int).item()
            dst_idxs.append(dup_0.iloc[[prev_row_idx]].index)
            prev_row = dup_0.iloc[[prev_row_idx]]

            assert next_row.shape[0] == 1
            dt_1 = duplicate_datetimes.iloc[-(i + 1)]
            dup_1 = curr_turbine_df.loc[curr_turbine_df.Timestamp == dt_1]
            assert dup_1.shape[0] <= 2
            next_row_idx = np.round(
                (np.abs(dup_1.values[:, 2:] - next_row.values[0, 2:]).argmin(axis=0).mean())).astype(int).item()
            dst_idxs.append(dup_1.iloc[[1 - next_row_idx]].index)
            next_row = dup_1.iloc[[next_row_idx]]
    for dst_idx in dst_idxs:
        df.loc[dst_idx, timestamp_col] -= datetime.timedelta(hours=1)

    assert (s := df[[timestamp_col, id_col]].duplicated().sum()) == 0, f"There are still {s} duplicates!"

    return df


def load_edp_data(failure_period_length: int = 21, return_dict: bool = True, verbose: bool = False,
                  turbine_ids: Optional[str | list[str]] = None, remove_zero_var_cols: bool = False,
                  additional_feature_aggs: Optional[dict] = None,
                  feature_name_map: Optional[dict[str, str]] = None) -> tuple:
    if additional_feature_aggs is None:
        additional_feature_aggs = {}

    if feature_name_map is None:
        feature_name_map = {}

    edp_data_path = Paths.EDP_DATASET_PATH

    # Load parquet files
    _df_scada, df_failures, _df_metmast, df_merged = dict(), dict(), dict(), dict()
    for y in [2016, 2017]:
        if not Path(edp_data_path / f'{y}_scada.parquet').exists():
            curr_scada_df = pd.read_excel(edp_data_path / f'Wind-Turbine-SCADA-signals-{y}.xlsx')
            curr_scada_df["Timestamp"] = pd.to_datetime(curr_scada_df["Timestamp"], utc=True)
            curr_scada_df["Turbine_ID"] = curr_scada_df["Turbine_ID"].astype(str)
            curr_scada_df = curr_scada_df.sort_values(by=['Timestamp', 'Turbine_ID']).reset_index(drop=True)
            curr_scada_df.to_parquet(edp_data_path / f'{curr_scada_df["Timestamp"].dt.year.iloc[0]}_scada.parquet')
        _df_scada[y] = pd.read_parquet(edp_data_path / f'{y}_scada.parquet')

        if not Path(edp_data_path / f'{y}_failures.parquet').exists():
            failure_df = pd.read_excel(
                edp_data_path / 'Historical-Failure-Logbook-2016.xlsx') if y == 2016 else pd.read_excel(
                edp_data_path / 'opendata-wind-failures-2017.xlsx')
            failure_df["Timestamp"] = pd.to_datetime(failure_df["Timestamp"], utc=True)
            failure_df["Turbine_ID"] = failure_df["Turbine_ID"].astype(str)
            failure_df["Remarks"] = failure_df["Remarks"].astype(str)
            failure_df["Component"] = failure_df["Component"].astype(str)
            failure_df = failure_df.sort_values(by=['Timestamp', 'Turbine_ID']).reset_index(drop=True)
            failure_df.to_parquet(edp_data_path / f'{failure_df["Timestamp"].dt.year.iloc[0]}_failures.parquet')
        df_failures[y] = pd.read_parquet(edp_data_path / f'{y}_failures.parquet')

        if not Path(edp_data_path / f'{y}_metmast.parquet').exists():
            metmast_df = pd.read_excel(edp_data_path / f'Onsite-MetMast-SCADA-data-{y}.xlsx')
            metmast_df["Timestamp"] = pd.to_datetime(metmast_df["Timestamp"], utc=True)
            metmast_df = metmast_df.sort_values(by=['Timestamp']).reset_index(drop=True)
            metmast_df.to_parquet(edp_data_path / f'{metmast_df["Timestamp"].dt.year.iloc[0]}_metmast.parquet')
        _df_metmast[y] = pd.read_parquet(edp_data_path / f'{y}_metmast.parquet')
        _df_metmast[y] = _df_metmast[y].drop(
            columns=[f"{agg_typ}_Winddirection2" for agg_typ in ["Min", "Max", "Avg", "Var"]])

        if turbine_ids is not None:
            if isinstance(turbine_ids, str):
                turbine_ids = [turbine_ids]
            _df_scada[y] = _df_scada[y].loc[_df_scada[y]["Turbine_ID"].isin(turbine_ids)]
            df_failures[y] = df_failures[y].loc[df_failures[y]["Turbine_ID"].isin(turbine_ids)]

            if _df_scada[y].empty or df_failures[y].empty or _df_metmast[y].empty:
                raise ValueError("Your selection of Turbine IDs yields empty dataframes.")

        df_merged[y] = pd.merge(_df_scada[y], _df_metmast[y], how='inner', on='Timestamp')

        df_failures[y] = dst_fix(df_failures[y], "Timestamp")
        df_merged[y] = dst_fix(df_merged[y], "Timestamp")
        df_merged[y] = duplicate_fix(df_merged[y], "Timestamp", "Turbine_ID")

        if remove_zero_var_cols:
            # Remove zero variance columns
            zero_variance_cols = []
            for col in df_merged[y].columns:
                if "Avg" not in col:
                    continue
                if df_merged[y][col].var() == 0:
                    zero_variance_cols.append(col)
            df_merged[y] = df_merged[y].drop(columns=zero_variance_cols)

        # df_merged[y]["rul"] = df_merged[y]["Timestamp"].apply(lambda x: np.min([(failure - x).days for failure in df_failures[y]["Timestamp"] if failure > x] + [np.inf]))
        # def get_rul(x):
        #     print(x)
        #     return np.min([(failure - x).days for failure in df_failures[y]["Timestamp"] if failure > x] + [-1])
        # df_merged[y].loc[:, "rul"] = df_merged[y].loc[:, ["Turbine_ID", "Timestamp"]].groupby("Turbine_ID", observed=True).transform(
        #         lambda x: get_rul(x)
        #     ).rename(columns={"Timestamp": "rul"}).loc[:, "rul"]
        # df_merged[y]["rul"] = np.inf
        # for turbine_id in df_merged[y]["Turbine_ID"].unique():
        #     curr_failures_reversed = df_failures[y].loc[(df_failures[y]["Turbine_ID"]==turbine_id), "Timestamp"].iloc[::-1]
        #     curr_timestamps = df_merged[y].loc[(df_merged[y]["Turbine_ID"]==turbine_id), "Timestamp"]
        #     for failure_timestamp in curr_failures_reversed:
        #         # print(failure_timestamp)
        #         df_merged[y].loc[(df_merged[y]["Turbine_ID"]==turbine_id), "rul"] = failure_timestamp - df_merged[y]["Timestamp"]
        # def func(df):
        #     m_df = pd.merge(df, df_failures[y], how="left", on="Turbine_ID", suffixes=("", "_failure"))
        #     # Only upcoming failures are of interest
        #     fill_value = np.max(df["Timestamp"])
        #     m_df["Timestamp_failure"] = m_df["Timestamp_failure"]
        #     m_df = m_df.loc[(m_df["Timestamp_failure"] >= m_df["Timestamp"]), :]
        #     # Get rul
        #     rul = m_df.groupby(["Timestamp", "Turbine_ID"])[["Timestamp_failure", "Timestamp"]].apply(lambda x: np.min(x["Timestamp_failure"] - x["Timestamp"]))
        #     return rul
        #
        # rul = df_merged[y].loc[:, ["Timestamp", "Turbine_ID"]].groupby("Turbine_ID").apply(lambda df: func(df))
        # break
        if verbose:
            print("Beginning:")
            print("Total size:", len(df_merged[y]))
            print("Duplicates:", df_merged[y][["Timestamp", "Turbine_ID"]].duplicated().sum(), "\n")

        _df = df_merged[y][df_merged[y][["Timestamp", "Turbine_ID"]].duplicated()]

        # for each observation, add all failures of the same turbine
        inner_merge = pd.merge(df_merged[y][["Timestamp", "Turbine_ID"]], df_failures[y], how="inner", on="Turbine_ID",
                               suffixes=("", "_failure"))
        if verbose:
            print("After merge:")
            print("Total size:", len(inner_merge))
            print("Duplicates:", inner_merge[["Timestamp", "Turbine_ID"]].duplicated().sum(), "\n")

        # Only keep future failures
        inner_merge = inner_merge.loc[(inner_merge["Timestamp"] <= inner_merge["Timestamp_failure"]), :]

        if verbose:
            print("Only future failures:")
            print("Total size:", len(inner_merge))
            print("Duplicates:", inner_merge[["Timestamp", "Turbine_ID"]].duplicated().sum(), "\n")

        # For each pair of (Timestamp, Turbine_ID), only keep the minimum "Timestamp_failure"
        # df_merged[y] = df_merged[y].groupby(["Timestamp", "Turbine_ID"], as_index=False).agg({"Timestamp_failure": "min"})
        idx_min = inner_merge.groupby(["Timestamp", "Turbine_ID"])["Timestamp_failure"].idxmin()
        inner_merge = inner_merge.loc[idx_min, :]

        if verbose:
            print("Only keep next failure (and NaNs):")
            print("Total size:", len(inner_merge))
            print("Duplicates:", inner_merge[["Timestamp", "Turbine_ID"]].duplicated().sum(), "\n")

        # Merge back with original dataframe
        df_merged[y] = pd.merge(df_merged[y], inner_merge, how="left", on=["Timestamp", "Turbine_ID"])

        if verbose:
            print("Merge back with original dataframe:")
            print("Total size:", len(df_merged[y]))
            print("Duplicates:", df_merged[y][["Timestamp", "Turbine_ID"]].duplicated().sum(), "\n")

        df_merged[y].loc[:, "rul"] = (df_merged[y]["Timestamp_failure"] - df_merged[y]["Timestamp"]).dt.days

        df_merged[y].loc[:, "rul"] = df_merged[y].loc[:, "rul"].fillna(np.inf)

        df_merged[y]["impending_failure"] = df_merged[y]["rul"].apply(lambda x: x < failure_period_length)

        df_merged[y].loc[:, "DayOfYear"] = df_merged[y]["Timestamp"].dt.dayofyear
        hour_of_day = df_merged[y]["Timestamp"].dt.hour
        df_merged[y].loc[:, "MinuteOfDay"] = df_merged[y]["Timestamp"].dt.minute + hour_of_day * 60

        diff = set(feature_name_map.keys()) - set(df_merged[y].columns)
        logging.debug(f"Not empty: set(feature_name_map.keys()) - set(df_merged[y].columns) = {diff}")
        diff = set(df_merged[y].columns) - set(feature_name_map.keys())
        if diff:
            raise ValueError(f"Not empty: set(df_merged[y].columns) - set(feature_name_map.keys()) = {diff}")
        df_merged[y] = df_merged[y].rename(columns=feature_name_map)

        df_merged[y]["Turbulence Intensity"] = compute_turbulence_intensity(
            mean_col="Ambient Wind Speed_Avg",
            std_col="Ambient Wind Speed_Std",
            data=df_merged[y],
        )

        df_merged[y]["Turbulence Intensity 1"] = compute_turbulence_intensity(
            mean_col=df_merged[y]["Wind Speed 1_Avg"],
            std_col=np.sqrt(df_merged[y]["Wind Speed 1_Var"]),
        )

        df_merged[y]["Turbulence Intensity 2"] = compute_turbulence_intensity(
            mean_col=df_merged[y]["Wind Speed 2_Avg"],
            std_col=np.sqrt(df_merged[y]["Wind Speed 2_Var"]),
        )

        final_features = []
        rename_dict = {}
        for feature_name in df_merged[y].columns:
            if (
                    "Avg_" not in feature_name and "_Avg" not in feature_name
                    and "Std_" not in feature_name.lower() and "_Std" not in feature_name.lower()
                    and "Min_" not in feature_name.lower() and "_Min" not in feature_name.lower()
                    and "Max_" not in feature_name.lower() and "_Max" not in feature_name.lower()
                    and "Var_" not in feature_name.lower() and "_Var" not in feature_name.lower()
            ):
                final_features.append(feature_name)
            else:
                curr_feature_name = "_".join((_split := feature_name.split("_"))[:-1])
                curr_feature_typ = _split[-1].lower()
                if curr_feature_typ == "avg":
                    final_features.append(feature_name)
                    rename_dict[feature_name] = curr_feature_name
                elif curr_feature_typ in additional_feature_aggs and curr_feature_name in additional_feature_aggs[
                    curr_feature_typ]:
                    final_features.append(feature_name)
        df_merged[y] = df_merged[y].loc[:, final_features]
        df_merged[y] = df_merged[y].rename(columns=rename_dict)

    if return_dict:
        return df_failures, df_merged
    else:
        return pd.concat([df_failures[y] for y in [2016, 2017]], axis=0), pd.concat(
            [df_merged[y] for y in [2016, 2017]], axis=0)


def load_pvod_data(station_id: int, feature_name_map: dict[str, str]) -> pd.DataFrame:
    pvod = PVODataset(str(Paths.PVOD_PATH) + "/", timezone="UTC+8")
    ori_data = pvod.read_ori_data(station_id=station_id)
    ori_data["date_time"] = pd.to_datetime(ori_data["date_time"])
    metadata = pvod.read_metadata()
    station_id_str = f"station{station_id:02d}"
    station_metadata = metadata.loc[metadata.Station_ID == station_id_str, :]
    if len(station_metadata) != 1:
        raise ValueError("len(station_metadata) != 1")
    for metadata_col in station_metadata:
        value = station_metadata[metadata_col].values[0]
        ori_data[metadata_col] = value

    latitude = metadata["Latitude"].values[0]
    longitude = metadata["Longitude"].values[0]
    altitude = lookup_altitude(latitude=latitude,
                               longitude=longitude)

    ori_data["Altitude"] = altitude

    dt_index = pd.DatetimeIndex(ori_data["date_time"])
    ori_data = add_sun_columns(ori_data, elev_col="Altitude")
    ori_data.loc[:, "is_daytime"] = (ori_data["date_time"] >= ori_data["sunrise"]) & (
            ori_data["date_time"] <= ori_data["sunset"])

    ori_data.loc[:, "DayOfYear"] = ori_data["date_time"].dt.dayofyear
    hour_of_day = ori_data["date_time"].dt.hour
    ori_data.loc[:, "MinuteOfDay"] = ori_data["date_time"].dt.minute + hour_of_day * 60

    location = Location(
        latitude=latitude,
        longitude=longitude,
        tz=ori_data["date_time"].dt.tz,
        altitude=altitude,
    )
    atmospheric_data = []
    for measurement_type in ["nwp", "lmd"]:
        solarposition = location.get_solarposition(times=dt_index, pressure=ori_data[f"{measurement_type}_pressure"],
                                                   temperature=ori_data[f"{measurement_type}_temperature"])
        clearsky = location.get_clearsky(times=dt_index, solar_position=solarposition)
        airmass = location.get_airmass(times=dt_index, solar_position=solarposition)
        curr_atmospheric_data = pd.concat(
            [solarposition.reset_index(drop=True), clearsky.reset_index(drop=True), airmass.reset_index(drop=True)],
            axis=1)
        curr_atmospheric_data = curr_atmospheric_data.add_prefix(f"{measurement_type}_")
        atmospheric_data.append(curr_atmospheric_data)

    ori_data = pd.concat([ori_data] + atmospheric_data, axis=1)

    return ori_data.rename(columns=feature_name_map)


def add_sun_columns(
        df: pd.DataFrame,
        *,
        lon_col: str = "Longitude",
        lat_col: str = "Latitude",
        dt_col: str = "date_time",
        tz_default: str = "UTC",
        elev_col: str | None = None,  # set to column name if you have per-row elevation
) -> pd.DataFrame:
    """
    For each row, compute Astral sun times (dawn, sunrise, noon, sunset, dusk)
    using its coordinates and calendar date from dt_col, then add one column per key.
    """
    # Ensure datetime dtype (keep tz if present; otherwise we’ll localize to tz_default)
    dt = pd.to_datetime(df[dt_col])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(ZoneInfo(tz_default))
    df = df.copy()
    df[dt_col] = dt

    def _row_sun_series(row: pd.Series) -> pd.Series:
        latitude = float(row[lat_col])
        longitude = float(row[lon_col])
        # elevation: prefer column if provided, else lookup
        elevation = float(row[elev_col]) if elev_col and pd.notna(row.get(elev_col)) else lookup_altitude(latitude,
                                                                                                          longitude)

        dt_row = row[dt_col]
        tzinfo = dt_row.tzinfo or ZoneInfo(tz_default)  # fallback if somehow missing
        obs = Observer(latitude=latitude, longitude=longitude, elevation=elevation)

        # Use the calendar date from the timestamp; Astral returns tz-aware datetimes
        s = sun(observer=obs, date=dt_row.date(), tzinfo=tzinfo)

        return pd.Series({k: v for k, v in s.items()})

    sun_df = df.apply(_row_sun_series, axis=1)

    # Optional: ensure consistent column order if you like
    preferred_order = [c for c in ("dawn", "sunrise", "noon", "sunset", "dusk") if c in sun_df.columns]
    sun_df = sun_df.reindex(columns=preferred_order + [c for c in sun_df.columns if c not in preferred_order])

    # Merge back
    return pd.concat([df, sun_df], axis=1)


# Get theoretical power curve
class WindTurbineFormulas:
    @staticmethod
    def power(air_temp, wind_speed_kmh, pressure, swept_area, power_coefficient):
        return power_coefficient * WindTurbineFormulas.wind_power(air_temp, wind_speed_kmh, pressure, swept_area)

    @staticmethod
    def wind_power(air_temp, wind_speed, pressure, swept_area):
        gas_constant_dry_air = 287.05  # J/(kg*K) | Assuming dry air
        wind_speed_ms = wind_speed
        air_density = pressure / (gas_constant_dry_air * air_temp)
        return 0.5 * air_density * swept_area * wind_speed_ms ** 3

    @staticmethod
    def Cp(actual_power, air_temp, wind_speed, pressure, swept_area):
        return actual_power / WindTurbineFormulas.wind_power(air_temp, wind_speed, pressure, swept_area)
