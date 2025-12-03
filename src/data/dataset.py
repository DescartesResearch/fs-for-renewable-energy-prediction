import datetime
import warnings
from typing import Optional, Literal
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from openoa.utils.met_data_processing import compute_shear

from data.data_processing import ProcessingPipeline, DataFrameProcessor
from data.feature_names import get_features_by_tags
from data.utils import load_edp_data, load_pvod_data


class BaseDataset(ABC):
    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validated(self, data: pd.DataFrame) -> pd.DataFrame:
        # Errors:
        if data.empty:
            raise ValueError("Dataset is empty")

        diff = data.columns.difference(list(self.get_feature_tags().keys()))
        if len(diff) > 0:
            raise ValueError(
                f"No tags defined for the following features: {diff}. Empty tag sets are also allowed.")

        # Warnings:
        nan_count = sum(data.isnull().sum().values)
        if nan_count > 0:
            warnings.warn(f"Dataset contains {nan_count} NaN values found in it.")

        obj_cols = data.select_dtypes(include='object').columns.tolist()
        if len(obj_cols) > 0:
            warnings.warn(f"The following columns have dtype 'object'. Please check if this is intended: {obj_cols}")

        diff = data.columns.difference(list(self.get_diagram_feature_name_map().keys()))
        if len(diff) > 0:
            warnings.warn(
                f"Following features not included in diagram name map: {diff}. Please add them if you want to create the feature clusters diagram later.")

        return data

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_feature_name_map(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_feature_tags(self) -> dict:
        pass

    @abstractmethod
    def get_cyclical_value_ranges(self) -> dict[str, tuple[float, float]]:
        pass

    @abstractmethod
    def get_base_pipeline(self):
        pass

    @abstractmethod
    def get_diagram_feature_name_map(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_task(self) -> Literal['regression', 'classification']:
        pass


class WindDataset(BaseDataset):

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.failure_period_length = config.get("failure_period_length", 21)
        self.verbose = config.get("verbose", False)
        self.turbine_id = config["asset_id"]
        self.remove_zero_var_cols = config.get("remove_zero_var_cols", False)
        self.additional_feature_aggs = config.get("additional_feature_aggs", None)
        self.features = config["features"]
        if self.features not in ["forecast_available", "digital_twin"]:
            raise ValueError(
                f"Unknown feature mode for the WindDataset: {self.features}. Must be 'forecast_available' or 'digital_twin'")

    def get_dataframe(self) -> pd.DataFrame:
        _failures_df, scada_and_metmast_data = load_edp_data(
            failure_period_length=self.failure_period_length,
            return_dict=False,
            verbose=self.verbose,
            turbine_ids=self.turbine_id,
            remove_zero_var_cols=self.remove_zero_var_cols,
            additional_feature_aggs=self.additional_feature_aggs,
            feature_name_map=self.get_feature_name_map(),
        )

        pipeline = self.get_base_pipeline()

        scada_and_metmast_data = pipeline(scada_and_metmast_data)

        return self._validated(scada_and_metmast_data)

    def get_feature_name_map(self) -> dict[str, str]:
        return {
            "Turbine_ID": "Turbine ID",
            "Timestamp": "Timestamp",
            "DayOfYear": "DayOfYear",
            "MinuteOfDay": "MinuteOfDay",

            # Generator
            "Gen_RPM_Max": "Generator RPM_Max",
            "Gen_RPM_Min": "Generator RPM_Min",
            "Gen_RPM_Avg": "Generator RPM_Avg",
            "Gen_RPM_Std": "Generator RPM_Std",
            "Gen_Bear_Temp_Avg": "Generator Bearing Temperature_Avg",
            "Gen_Bear2_Temp_Avg": "Generator Bearing 2 Temperature_Avg",
            "Gen_Phase1_Temp_Avg": "Generator Phase 1 Temperature_Avg",
            "Gen_Phase2_Temp_Avg": "Generator Phase 2 Temperature_Avg",
            "Gen_Phase3_Temp_Avg": "Generator Phase 3 Temperature_Avg",
            "Gen_SlipRing_Temp_Avg": "Generator Slip Ring Temperature_Avg",

            # Rotor
            "Rtr_RPM_Max": "Rotor RPM_Max",
            "Rtr_RPM_Min": "Rotor RPM_Min",
            "Rtr_RPM_Avg": "Rotor RPM_Avg",
            "Rtr_RPM_Std": "Rotor RPM_Std",

            # Ambient / Wind
            "Amb_WindSpeed_Max": "Ambient Wind Speed_Max",
            "Amb_WindSpeed_Min": "Ambient Wind Speed_Min",
            "Amb_WindSpeed_Avg": "Ambient Wind Speed_Avg",
            "Amb_WindSpeed_Std": "Ambient Wind Speed_Std",
            "Amb_WindSpeed_Est_Avg": "Estimated Ambient Wind Speed_Avg",
            "Amb_WindDir_Relative_Avg": "Relative Wind Direction_Avg",
            "Amb_WindDir_Abs_Avg": "Absolute Wind Direction_Avg",
            "Amb_Temp_Avg": "Ambient Temperature 2_Avg",

            # Deviated Quantities - computed with OpenOA
            "Turbulence Intensity": "Turbulence Intensity",
            "Turbulence Intensity 1": "Turbulence Intensity 1",
            "Turbulence Intensity 2": "Turbulence Intensity 2",
            "Wind Shear": "Wind Shear",

            # Blade Pitch
            "Blds_PitchAngle_Min": "Blade Pitch Angle_Min",
            "Blds_PitchAngle_Max": "Blade Pitch Angle_Max",
            "Blds_PitchAngle_Avg": "Blade Pitch Angle_Avg",
            "Blds_PitchAngle_Std": "Blade Pitch Angle_Std",

            # Power Grid
            "Grd_Prod_Pwr_Avg": "Grid Produced Power_Avg",
            "Grd_Prod_Pwr_Max": "Grid Produced Power_Max",
            "Grd_Prod_Pwr_Min": "Grid Produced Power_Min",
            "Grd_Prod_Pwr_Std": "Grid Produced Power_Std",
            "Grd_Prod_ReactPwr_Avg": "Grid Reactive Power_Avg",
            "Grd_Prod_ReactPwr_Max": "Grid Reactive Power_Max",
            "Grd_Prod_ReactPwr_Min": "Grid Reactive Power_Min",
            "Grd_Prod_ReactPwr_Std": "Grid Reactive Power_Std",
            "Grd_Prod_PsblePwr_Avg": "Grid Possible Power_Avg",
            "Grd_Prod_PsblePwr_Max": "Grid Possible Power_Max",
            "Grd_Prod_PsblePwr_Min": "Grid Possible Power_Min",
            "Grd_Prod_PsblePwr_Std": "Grid Possible Power_Std",
            "Grd_Prod_PsbleInd_Avg": "Grid Possible Inductive Power_Avg",
            "Grd_Prod_PsbleInd_Max": "Grid Possible Inductive Power_Max",
            "Grd_Prod_PsbleInd_Min": "Grid Possible Inductive Power_Min",
            "Grd_Prod_PsbleInd_Std": "Grid Possible Inductive Power_Std",
            "Grd_Prod_PsbleCap_Avg": "Grid Possible Capacitive Power_Avg",
            "Grd_Prod_PsbleCap_Max": "Grid Possible Capacitive Power_Max",
            "Grd_Prod_PsbleCap_Min": "Grid Possible Capacitive Power_Min",
            "Grd_Prod_PsbleCap_Std": "Grid Possible Capacitive Power_Std",
            "Grd_Prod_CosPhi_Avg": "Grid CosPhi_Avg",
            "Grd_Prod_Freq_Avg": "Grid Frequency_Avg",
            "Grd_Prod_VoltPhse1_Avg": "Grid Voltage Phase 1_Avg",
            "Grd_Prod_VoltPhse2_Avg": "Grid Voltage Phase 2_Avg",
            "Grd_Prod_VoltPhse3_Avg": "Grid Voltage Phase 3_Avg",
            "Grd_Prod_CurPhse1_Avg": "Grid Current Phase 1_Avg",
            "Grd_Prod_CurPhse2_Avg": "Grid Current Phase 2_Avg",
            "Grd_Prod_CurPhse3_Avg": "Grid Current Phase 3_Avg",
            "Grd_InverterPhase1_Temp_Avg": "Grid Inverter Phase 1 Temperature_Avg",
            "Grd_RtrInvPhase1_Temp_Avg": "Grid Rotor Inverter Phase 1 Temperature_Avg",
            "Grd_RtrInvPhase2_Temp_Avg": "Grid Rotor Inverter Phase 2 Temperature_Avg",
            "Grd_RtrInvPhase3_Temp_Avg": "Grid Rotor Inverter Phase 3 Temperature_Avg",
            "Grd_Busbar_Temp_Avg": "Grid Busbar Temperature_Avg",

            # Transformer & Inverter
            "HVTrafo_Phase1_Temp_Avg": "High Voltage Transformer Phase 1 Temperature_Avg",
            "HVTrafo_Phase2_Temp_Avg": "High Voltage Transformer Phase 2 Temperature_Avg",
            "HVTrafo_Phase3_Temp_Avg": "High Voltage Transformer Phase 3 Temperature_Avg",

            # Control cabinet
            "Cont_Top_Temp_Avg": "Control Cabinet Top Temperature_Avg",
            "Cont_Hub_Temp_Avg": "Control Hub Temperature_Avg",
            "Cont_VCP_Temp_Avg": "VCP Temperature_Avg",
            "Cont_VCP_ChokcoilTemp_Avg": "VCP Choke Coil Temperature_Avg",
            "Cont_VCP_WtrTemp_Avg": "VCP Water Temperature_Avg",

            # Others
            "Hyd_Oil_Temp_Avg": "Hydraulic Oil Temperature_Avg",
            "Gear_Oil_Temp_Avg": "Gear Oil Temperature_Avg",
            "Gear_Bear_Temp_Avg": "Gear Bearing Temperature_Avg",
            "Nac_Temp_Avg": "Nacelle Temperature_Avg",
            "Nac_Direction_Avg": "Nacelle Direction_Avg",
            "Spin_Temp_Avg": "Spinner Temperature_Avg",

            # Production (possibly turbine-specific)
            "Prod_LatestAvg_ActPwrGen0": "Active Power Generator 0_Avg",
            "Prod_LatestAvg_ActPwrGen1": "Active Power Generator 1_Avg",
            "Prod_LatestAvg_ActPwrGen2": "Active Power Generator 2_Avg",
            "Prod_LatestAvg_TotActPwr": "Total Active Power_Avg",
            "Prod_LatestAvg_ReactPwrGen0": "Reactive Power Generator 0_Avg",
            "Prod_LatestAvg_ReactPwrGen1": "Reactive Power Generator 1_Avg",
            "Prod_LatestAvg_ReactPwrGen2": "Reactive Power Generator 2_Avg",
            "Prod_LatestAvg_TotReactPwr": "Total Reactive Power_Avg",

            # Alternative Wind Measurements
            "Min_Windspeed1": "Wind Speed 1_Min",  # @80m
            "Max_Windspeed1": "Wind Speed 1_Max",
            "Avg_Windspeed1": "Wind Speed 1_Avg",
            "Var_Windspeed1": "Wind Speed 1_Var",
            "Min_Windspeed2": "Wind Speed 2_Min",  # @77m
            "Max_Windspeed2": "Wind Speed 2_Max",
            "Avg_Windspeed2": "Wind Speed 2_Avg",
            "Var_Windspeed2": "Wind Speed 2_Var",

            # "Min_Winddirection2": "Wind Direction 2_Min",
            # "Max_Winddirection2": "Wind Direction 2_Max",
            # "Avg_Winddirection2": "Wind Direction 2_Avg",
            # "Var_Winddirection2": "Wind Direction 2_Var",

            "Min_AmbientTemp": "Ambient Temperature_Min",
            "Max_AmbientTemp": "Ambient Temperature_Max",
            "Avg_AmbientTemp": "Ambient Temperature_Avg",

            # Humidity & Precipitation
            "Min_Pressure": "Atmospheric Pressure_Min",
            "Max_Pressure": "Atmospheric Pressure_Max",
            "Avg_Pressure": "Atmospheric Pressure_Avg",
            "Min_Humidity": "Humidity_Min",
            "Max_Humidity": "Humidity_Max",
            "Avg_Humidity": "Humidity_Avg",
            "Min_Precipitation": "Precipitation_Min",
            "Max_Precipitation": "Precipitation_Max",
            "Avg_Precipitation": "Precipitation_Avg",

            # Rain Detection
            "Avg_Raindetection": "Rain Detection_Avg",
            "Min_Raindetection": "Rain Detection_Min",
            "Max_Raindetection": "Rain Detection_Max",

            # Anemometers
            "Anemometer1_Freq": "Anemometer 1 Frequency",  # Anemometer 1 height: 80m
            "Anemometer1_Offset": "Anemometer 1 Offset",
            "Anemometer1_CorrGain": "Anemometer 1 Correction Gain",
            "Anemometer1_CorrOffset": "Anemometer 1 Correction Offset",
            "Anemometer1_Avg_Freq": "Anemometer 1 Frequency_Avg",
            "Anemometer2_Freq": "Anemometer 2 Frequency",  # Anemometer 1 height: 77m
            "Anemometer2_Offset": "Anemometer 2 Offset",
            "Anemometer2_CorrGain": "Anemometer 2 Correction Gain",
            "Anemometer2_CorrOffset": "Anemometer 2 Correction Offset",
            "Anemometer2_Avg_Freq": "Anemometer 2 Frequency_Avg",

            # Pressure sensor
            "Pressure_Avg_Freq": "Pressure Sensor Frequency_Avg",
            "DistanceAirPress": "Distance to Air Pressure Sensor",
            "AirRessureSensorZeroOffset": "Air Pressure Sensor Zero Offset",

            # Labels & Meta
            "Component": "Component",
            "Timestamp_failure": "Failure Timestamp",
            "Remarks": "Remarks",
            "rul": "RUL",
            "impending_failure": "Impending Failure"
        }

    def get_feature_tags(self) -> dict:
        return {
            # Meta
            "Turbine ID": {"meta"},
            "Timestamp": {"meta"},
            "RUL": {"meta"},
            "Impending Failure": {"meta"},
            "Component": {"meta"},
            "Failure Timestamp": {"meta"},
            "Remarks": {"meta"},

            # Target
            "Grid Produced Power": {"target"},

            # Datetime
            "DayOfYear": {"circular", "forecast_available"},
            "DayOfYear_sin": {"cyclical_encoding", "forecast_available"},
            "DayOfYear_cos": {"cyclical_encoding", "forecast_available"},
            "MinuteOfDay": {"circular", "forecast_available"},
            "MinuteOfDay_sin": {"cyclical_encoding", "forecast_available"},
            "MinuteOfDay_cos": {"cyclical_encoding", "forecast_available"},

            # Power Proxies (even if not the target)
            "Total Active Power": {"power_proxy"},
            "Active Power Generator 0": {"power_proxy"},
            "Active Power Generator 1": {"power_proxy"},
            "Active Power Generator 2": {"power_proxy"},
            "Total Reactive Power": {"power_proxy"},
            "Reactive Power Generator 0": {"power_proxy"},
            "Reactive Power Generator 1": {"power_proxy"},
            "Reactive Power Generator 2": {"power_proxy"},
            "Grid CosPhi": {"power_proxy"},
            "Grid Voltage Phase 1": {"power_proxy"},
            "Grid Voltage Phase 2": {"power_proxy"},
            "Grid Voltage Phase 3": {"power_proxy"},
            "Grid Current Phase 1": {"power_proxy"},
            "Grid Current Phase 2": {"power_proxy"},
            "Grid Current Phase 3": {"power_proxy"},
            "Grid Reactive Power": {"power_proxy"},
            "Grid Possible Power": {"power_proxy"},
            "Grid Possible Inductive Power": {"power_proxy"},
            "Grid Possible Capacitive Power": {"power_proxy"},
            "Estimated Ambient Wind Speed": {"power_proxy"},

            # Forecast-available environmental features
            "Ambient Wind Speed": {"forecast_available"},
            # "Ambient Wind Speed_Std": {"forecast_available"},
            # "Estimated Ambient Wind Speed": {"forecast_available"},
            "Absolute Wind Direction": {"forecast_available", "circular"},  # from 0° to 360°
            "Absolute Wind Direction_sin": {"forecast_available", "cyclical_encoding"},
            "Absolute Wind Direction_cos": {"forecast_available", "cyclical_encoding"},
            "Ambient Temperature": {"forecast_available"},
            "Ambient Temperature 2": {"forecast_available"},
            "Wind Speed 1": {"forecast_available"},
            "Wind Speed 2": {"forecast_available"},
            # "Wind Direction 2": {"forecast_available", "circular"},  # from 0 to 360°
            "Atmospheric Pressure": {"forecast_available"},
            "Humidity": {"forecast_available"},
            "Precipitation": {"forecast_available"},
            "Rain Detection": {"forecast_available"},

            # Anemometers (forecast-relevant if derived from sensors upstream)
            "Anemometer 1 Frequency": {"forecast_available"},
            "Anemometer 1 Offset": {"forecast_available"},
            "Anemometer 1 Correction Gain": {"forecast_available"},
            "Anemometer 1 Correction Offset": {"forecast_available"},
            "Anemometer 2 Frequency": {"forecast_available"},
            "Anemometer 2 Offset": {"forecast_available"},
            "Anemometer 2 Correction Gain": {"forecast_available"},
            "Anemometer 2 Correction Offset": {"forecast_available"},
            "Distance to Air Pressure Sensor": {"forecast_available"},
            "Air Pressure Sensor Zero Offset": {"forecast_available"},

            # System-state environmental features
            "Relative Wind Direction": {"system_state", "circular"},  # from -180° to 180°
            "Relative Wind Direction_sin": {"system_state", "cyclical_encoding"},
            "Relative Wind Direction_cos": {"system_state", "cyclical_encoding"},
            "Pressure Sensor Frequency": {"system_state"},

            # Deviated Quantities - computed with OpenOA
            "Turbulence Intensity": {"forecast_available"},
            "Turbulence Intensity 1": {"forecast_available"},
            "Turbulence Intensity 2": {"forecast_available"},
            "Wind Shear": {"forecast_available"},

            # system state features (unflagged → usable in Task 1 but excluded in Task 2)
            "Generator RPM": {"system_state"},
            "Generator Bearing Temperature": {"system_state"},
            "Generator Bearing 2 Temperature": {"system_state"},
            "Generator Phase 1 Temperature": {"system_state"},
            "Generator Phase 2 Temperature": {"system_state"},
            "Generator Phase 3 Temperature": {"system_state"},
            "Hydraulic Oil Temperature": {"system_state"},
            "Gear Oil Temperature": {"system_state"},
            "Gear Bearing Temperature": {"system_state"},
            "Nacelle Temperature": {"system_state"},
            "Rotor RPM": {"system_state"},
            "High Voltage Transformer Phase 1 Temperature": {"system_state"},
            "High Voltage Transformer Phase 2 Temperature": {"system_state"},
            "High Voltage Transformer Phase 3 Temperature": {"system_state"},
            "Grid Busbar Temperature": {"system_state"},
            "Grid Inverter Phase 1 Temperature": {"system_state"},
            "Control Cabinet Top Temperature": {"system_state"},
            "Control Hub Temperature": {"system_state"},
            "VCP Temperature": {"system_state"},
            "Generator Slip Ring Temperature": {"system_state"},
            "Spinner Temperature": {"system_state"},
            "Blade Pitch Angle": {"system_state"},  # from -2.5° to 90° - not circular!
            "VCP Choke Coil Temperature": {"system_state"},
            "Grid Rotor Inverter Phase 1 Temperature": {"system_state"},
            "Grid Rotor Inverter Phase 2 Temperature": {"system_state"},
            "Grid Rotor Inverter Phase 3 Temperature": {"system_state"},
            "VCP Water Temperature": {"system_state"},
            "Grid Frequency": {"system_state"},
            "Nacelle Direction": {"system_state", "circular"},
            "Nacelle Direction_sin": {"system_state", "cyclical_encoding"},
            "Nacelle Direction_cos": {"system_state", "cyclical_encoding"}
        }

    def get_cyclical_value_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            # "Blade Pitch Angle": (-2.5, 90), # not circular!
            "Relative Wind Direction": (-180, 180),
            "Absolute Wind Direction": (0, 360),
            # "Wind Direction 2": (0, 360),
            "MinuteOfDay": (0, 24 * 60 - 1),
            "DayOfYear": (1, 366),
            "Nacelle Direction": (0, 360),
            # add others as needed
        }

    def get_diagram_feature_name_map(self) -> dict[str, str]:
        return {
            "Generator Bearing Temperature": "Bearing Temp. 1-2",
            "Generator Bearing 2 Temperature": "",
            "Generator Phase 1 Temperature": "Phase Temp. 1-3",
            "Generator Phase 2 Temperature": "",
            "Generator Phase 3 Temperature": "",
            "High Voltage Transformer Phase 1 Temperature": "HV Transformer Temp. 1-3",
            "High Voltage Transformer Phase 2 Temperature": "",
            "High Voltage Transformer Phase 3 Temperature": "",
            "Wind Speed 1": "Wind Speed Sensor 1",
            "Wind Speed 2": "Wind Speed Sensor 2",
            "Grid Rotor Inverter Phase 1 Temperature": "Rotor Inv. Temp. 1-3",
            "Grid Rotor Inverter Phase 2 Temperature": "",
            "Grid Rotor Inverter Phase 3 Temperature": "",
            "Grid Inverter Phase 1 Temperature": "Grid Inv. Temp.",
            "Control Cabinet Top Temperature": "Cabinet Temp.",
            "VCP Temperature": "VCP Temp.",
            "Generator Slip Ring Temperature": "Slip Ring Temp.",
            "Control Hub Temperature": "Hub Temp.",
            "Spinner Temperature": "Spinner Temp.",
            "Ambient Temperature": "Amb. Temp.",
            "Hydraulic Oil Temperature": "Hydraulic Oil Temp.",
            "DayOfYear": "Day of Year",
            "DayOfYear_sin": "Day of Year (Sin)",
            "DayOfYear_cos": "Day of Year (Cos)",
            "MinuteOfDay": "Minute of Day",
            "MinuteOfDay_sin": "Minute of Day (Sin)",
            "MinuteOfDay_cos": "Minute of Day (Cos)",
            "Turbulence Intensity": "TI",
            "Turbulence Intensity 1": "TI1",
            "Turbulence Intensity 2": "TI2",
            "Active Power Generator 0": "Gen. Act. Power 1-3",
            "Active Power Generator 1": "",
            "Active Power Generator 2": "",
            "Absolute Wind Direction": "Abs. Wind Dir.",
            "Absolute Wind Direction_sin": "Abs. Wind Dir. (Sin)",
            "Absolute Wind Direction_cos": "Abs. Wind Dir. (Cos)",
            "Air Pressure Sensor Zero Offset": "Air Press. Sensor Zero Off.",
            "Ambient Temperature 2": "Amb. Temp. 2",
            "Ambient Wind Speed": "Amb. Wind Speed",
            "Anemometer 1 Correction Gain": "Anem.1 Corr. Gain",
            "Anemometer 1 Correction Offset": "Anem.1 Corr. Off.",
            "Anemometer 1 Frequency": "Anem.1 Freq.",
            "Anemometer 1 Offset": "Anem.1 Offset",
            "Anemometer 2 Correction Gain": "Anem.2 Corr. Gain",
            "Anemometer 2 Correction Offset": "Anem.2 Corr. Off.",
            "Anemometer 2 Frequency": "Anem.2 Freq.",
            "Anemometer 2 Offset": "Anem.2 Offset",
            "Blade Pitch Angle": "Blade Pitch Ang.",
            "Component": "Component",
            "Distance to Air Pressure Sensor": "Dist. to Air Press. Sensor",
            "Estimated Ambient Wind Speed": "Est. Ambient Wind Speed",
            "Failure Timestamp": "Failure Time",
            "Gear Bearing Temperature": "Gear Bearing Temp.",
            "Gear Oil Temperature": "Gear Oil Temp.",
            "Generator RPM": "Gen. RPM",
            "Grid Busbar Temperature": "Busbar Temp.",
            "Grid CosPhi": "Grid CosPhi",
            "Grid Current Phase 1": "Grid Curr. 1-3",
            "Grid Current Phase 2": "",
            "Grid Current Phase 3": "",
            "Grid Frequency": "Grid Freq.",
            "Grid Possible Capacitive Power": "Grid Cap. Power",
            "Grid Possible Inductive Power": "Grid Ind. Power",
            "Grid Possible Power": "Grid Poss. Power",
            "Grid Produced Power": "Grid Prod. Power",
            "Grid Reactive Power": "Grid React. Power",
            "Grid Voltage Phase 1": "Grid Volt. 1-3",
            "Grid Voltage Phase 2": "",
            "Grid Voltage Phase 3": "",
            "Humidity": "Humidity",
            "Impending Failure": "Imp. Failure",
            "Nacelle Direction": "Nacelle Dir.",
            "Nacelle Direction_sin": "Nacelle Dir. (Sin)",
            "Nacelle Direction_cos": "Nacelle Dir. (Cos)",
            "Nacelle Temperature": "Nacelle Temp.",
            "Precipitation": "Precipitation",
            "Atmospheric Pressure": "Atm. Pressure",
            "Pressure Sensor Frequency": "Pressure Freq.",
            "RUL": "RUL",
            "Rain Detection": "Rain Detect.",
            "Reactive Power Generator 0": "Gen. React. Pwr 1-3",
            "Reactive Power Generator 1": "",
            "Reactive Power Generator 2": "",
            "Relative Wind Direction": "Rel. Wind Dir.",
            "Relative Wind Direction_sin": "Rel. Wind Dir. (Sin)",
            "Relative Wind Direction_cos": "Rel. Wind Dir. (Cos)",
            "Remarks": "Remarks",
            "Rotor RPM": "Rotor RPM",
            "Timestamp": "Timestamp",
            "Total Active Power": "Total Act. Power",
            "Total Reactive Power": "Total React. Power",
            "Turbine ID": "Turbine ID",
            "VCP Choke Coil Temperature": "VCP Choke Temp.",
            "VCP Water Temperature": "VCP Water Temp.",
            "Wind Shear": "Wind Shear",
        }

    def get_base_pipeline(self):
        pipeline = ProcessingPipeline()
        pipeline.apply(DataFrameProcessor.filter_columns(get_features_by_tags(
            self.get_feature_tags(), tags_exclude={"power_proxy"}
        )))
        pipeline.apply(DataFrameProcessor.drop_duplicates())
        pipeline.apply(DataFrameProcessor.drop_na_columns(p_thresh=0.95))
        pipeline.apply(DataFrameProcessor.drop_na_rows())
        pipeline.apply(
            lambda df: self.add_wind_shear(df, ("Wind Speed 1", "Wind Speed 2"), anemometer_heights=(80, 77)))
        pipeline.apply(DataFrameProcessor.drop_na_columns(p_thresh=0.95))
        pipeline.apply(DataFrameProcessor.drop_na_rows())
        pipeline.apply(DataFrameProcessor.drop_inf_rows(
            cols=["Turbulence Intensity 2"]))
        pipeline.apply(DataFrameProcessor.remove_low_variance_columns(
            variance_threshold=1e-10,
            cols=get_features_by_tags(self.get_feature_tags(), tags_exclude={"meta", "target", "power_proxy"})
        ))
        pipeline.apply(
            DataFrameProcessor.add_cyclical_encodings(self.get_feature_tags(), self.get_cyclical_value_ranges()))
        pipeline.apply(DataFrameProcessor.sort(by_columns="Timestamp", ascending=True))
        return pipeline

    def get_task(self) -> Literal['regression', 'classification']:
        return 'regression'

    @staticmethod
    def add_wind_shear(df: pd.DataFrame, wind_speed_colnames: tuple[str, str],
                       anemometer_heights: tuple[float, float]) -> pd.DataFrame:
        # The problem with Wind Shear computation: There are zero wind speed entries, and due to log calculations, these are problematic and numerically unstable.
        # So we replace zero values with a small epsilon
        eps = 0.1
        u0 = df[wind_speed_colnames[0]].values
        u1 = df[wind_speed_colnames[1]].values

        if eps > 0:
            u0 = np.where(u0 == 0, eps, u0)
            u1 = np.where(u1 == 0, eps, u1)

        wind_shear = compute_shear(
            data=pd.DataFrame({
                "u0": u0,
                "u1": u1,
            }),
            ws_heights={
                "u0": anemometer_heights[0],
                "u1": anemometer_heights[1],
            }
        )  # => np.ndarray

        # If the wind speed difference at the two heights is smaller than the sum of their stds, the wind shear is set to 0
        # u1_std = np.sqrt(df_merged[y]["Wind Speed 1_Var"].values)
        # u2_std = np.sqrt(df_merged[y]["Wind Speed 2_Var"].values)
        # wind_shear = np.where(np.abs(u0-u1) < (u1_std+u2_std), 0, wind_shear)

        df["Wind Shear"] = wind_shear

        return df


class SolarDataset(BaseDataset):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.station_id = config["asset_id"]

    def get_dataframe(self) -> pd.DataFrame:
        df = load_pvod_data(int(self.station_id), self.get_feature_name_map())

        pipeline = self.get_base_pipeline()

        return self._validated(pipeline(df))

    def get_feature_name_map(self) -> dict[str, str]:
        return {
            "date_time": "Timestamp",
            "DayOfYear": "DayOfYear",
            "MinuteOfDay": "MinuteOfDay",

            "dawn": "Dawn",
            "sunrise": "Sunrise",
            "noon": "Noon",
            "sunset": "Sunset",
            "dusk": "Dusk",
            "is_daytime": "is_daytime",

            "Station_ID": "Station ID",
            "Capacity": "Capacity",
            "PV_Technology": "PV Technology",
            "Panel_Size": "Panel Size",
            "Module": "Module",
            "Inverters": "Inverters",
            "Layout": "Layout",
            "Panel_Number": "Panel Number",
            "Array_Tilt": "Array Tilt",
            "Pyranometer": "Pyranometer",
            "Longitude": "Longitude",
            "Latitude": "Latitude",
            "Altitude": "Altitude",

            "power": "Power",

            "nwp_globalirrad": "NWP Global Irradiance",
            "nwp_directirrad": "NWP Direct Irradiance",
            "nwp_temperature": "NWP Temperature",
            "nwp_humidity": "NWP Humidity",
            "nwp_windspeed": "NWP Wind Speed",
            "nwp_winddirection": "NWP Wind Direction",
            "nwp_pressure": "NWP Pressure",

            "lmd_totalirrad": "LMD Global Irradiance",
            "lmd_diffuseirrad": "LMD Diffuse Irradiance",
            "lmd_temperature": "LMD Temperature",
            "lmd_pressure": "LMD Pressure",
            "lmd_winddirection": "LMD Wind Direction",
            "lmd_windspeed": "LMD Wind Speed",

            # Solar position features (see: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.location.Location.get_solarposition.html)
            "nwp_apparent_zenith": "NWP Apparent Zenith",
            "nwp_zenith": "NWP Zenith",
            "nwp_apparent_elevation": "NWP Apparent Elevation",
            "nwp_elevation": "NWP Elevation",
            "nwp_azimuth": "NWP Azimuth",
            "nwp_equation_of_time": "NWP Equation of Time",

            "lmd_apparent_zenith": "LMD Apparent Zenith",
            "lmd_zenith": "LMD Zenith",
            "lmd_apparent_elevation": "LMD Apparent Elevation",
            "lmd_elevation": "LMD Elevation",
            "lmd_azimuth": "LMD Azimuth",
            "lmd_equation_of_time": "LMD Equation of Time",

            # Clear sky estimates (see: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.location.Location.get_clearsky.html)
            "nwp_ghi": "NWP GHI",
            "nwp_dni": "NWP DNI",
            "nwp_dhi": "NWP DHI",

            "lmd_ghi": "LMD GHI",
            "lmd_dni": "LMD DNI",
            "lmd_dhi": "LMD DHI",

            # See: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.location.Location.get_airmass.html
            "nwp_airmass_relative": "NWP Relative Airmass",
            "nwp_airmass_absolute": "NWP Absolute Airmass",

            "lmd_airmass_relative": "LMD Relative Airmass",
            "lmd_airmass_absolute": "LMD Absolute Airmass",

        }

    def get_feature_tags(self) -> dict:
        return {
            # Meta
            "Station ID": {"meta"},
            "Capacity": {"meta"},
            "PV Technology": {"meta"},
            "Panel Size": {"meta"},
            "Module": {"meta"},
            "Inverters": {"meta"},
            "Layout": {"meta"},
            "Panel Number": {"meta"},
            "Array Tilt": {"meta"},
            "Pyranometer": {"meta"},
            "Longitude": {"meta", "circular"},
            "Longitude_sin": {"meta", "cyclical_encoding"},
            "Longitude_cos": {"meta", "cyclical_encoding"},
            "Latitude": {"meta"},  # explicitly not circular
            "Altitude": {"meta"},

            # Target
            "Power": {"target"},

            # Datetime
            "Timestamp": {"meta"},
            "DayOfYear": {"circular", "forecast_available"},
            "DayOfYear_sin": {"cyclical_encoding", "forecast_available"},
            "DayOfYear_cos": {"cyclical_encoding", "forecast_available"},
            "MinuteOfDay": {"circular", "forecast_available"},
            "MinuteOfDay_sin": {"cyclical_encoding", "forecast_available"},
            "MinuteOfDay_cos": {"cyclical_encoding", "forecast_available"},
            "Dawn": {"meta"},
            "Sunrise": {"meta"},
            "Noon": {"meta"},
            "Sunset": {"meta"},
            "Dusk": {"meta"},
            "is_daytime": {"meta"},

            # Forecast-available environmental features
            "NWP Global Irradiance": {"forecast_available"},
            "NWP Direct Irradiance": {"forecast_available"},
            "NWP Temperature": {"forecast_available"},
            "NWP Humidity": {"forecast_available"},
            "NWP Wind Speed": {"forecast_available"},
            "NWP Wind Direction": {"forecast_available", "circular"},  # 0 to 360°
            "NWP Wind Direction_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Wind Direction_cos": {"forecast_available", "cyclical_encoding"},
            "NWP Pressure": {"forecast_available"},

            "LMD Global Irradiance": {"system_state"},
            "LMD Diffuse Irradiance": {"system_state"},
            "LMD Temperature": {"system_state"},
            "LMD Pressure": {"system_state"},
            "LMD Wind Direction": {"system_state", "circular"},  # 0 to 360°
            "LMD Wind Direction_sin": {"system_state", "cyclical_encoding"},
            "LMD Wind Direction_cos": {"system_state", "cyclical_encoding"},
            "LMD Wind Speed": {"system_state"},

            # pvlib features (see: https://pvlib-python.readthedocs.io/en/stable/user_guide/extras/nomenclature.html)
            # based on numerical weather prediction (NWP) data
            "NWP Absolute Airmass": {"forecast_available"},
            "NWP Relative Airmass": {"forecast_available"},
            "NWP DHI": {"forecast_available"},
            "NWP DNI": {"forecast_available"},
            "NWP GHI": {"forecast_available"},
            "NWP Equation of Time": {"forecast_available"},
            "NWP Apparent Elevation": {"forecast_available", "circular"},
            "NWP Apparent Zenith": {"forecast_available", "circular"},
            "NWP Azimuth": {"forecast_available", "circular"},
            "NWP Elevation": {"forecast_available", "circular"},
            "NWP Zenith": {"forecast_available", "circular"},

            "NWP Azimuth_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Azimuth_cos": {"forecast_available", "cyclical_encoding"},
            "NWP Elevation_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Elevation_cos": {"forecast_available", "cyclical_encoding"},
            "NWP Zenith_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Zenith_cos": {"forecast_available", "cyclical_encoding"},
            "NWP Apparent Zenith_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Apparent Zenith_cos": {"forecast_available", "cyclical_encoding"},
            "NWP Apparent Elevation_sin": {"forecast_available", "cyclical_encoding"},
            "NWP Apparent Elevation_cos": {"forecast_available", "cyclical_encoding"},

            # based on local measurements (LMD) data
            "LMD Absolute Airmass": {"system_state"},
            "LMD Relative Airmass": {"system_state"},
            "LMD DHI": {"system_state"},
            "LMD DNI": {"system_state"},
            "LMD GHI": {"system_state"},
            "LMD Equation of Time": {"system_state"},
            "LMD Apparent Elevation": {"system_state", "circular"},
            "LMD Apparent Zenith": {"system_state", "circular"},
            "LMD Azimuth": {"system_state", "circular"},
            "LMD Elevation": {"system_state", "circular"},
            "LMD Zenith": {"system_state", "circular"},

            "LMD Azimuth_sin": {"system_state", "cyclical_encoding"},
            "LMD Azimuth_cos": {"system_state", "cyclical_encoding"},
            "LMD Elevation_sin": {"system_state", "cyclical_encoding"},
            "LMD Elevation_cos": {"system_state", "cyclical_encoding"},
            "LMD Zenith_sin": {"system_state", "cyclical_encoding"},
            "LMD Zenith_cos": {"system_state", "cyclical_encoding"},
            "LMD Apparent Zenith_sin": {"system_state", "cyclical_encoding"},
            "LMD Apparent Zenith_cos": {"system_state", "cyclical_encoding"},
            "LMD Apparent Elevation_sin": {"system_state", "cyclical_encoding"},
            "LMD Apparent Elevation_cos": {"system_state", "cyclical_encoding"},
        }

    def get_cyclical_value_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            "Longitude": (-180, 180),
            "NWP Wind Direction": (0, 360),
            "LMD Wind Direction": (0, 360),
            "MinuteOfDay": (0, 24 * 60 - 1),
            "DayOfYear": (1, 366),

            # pvlib features
            "LMD Azimuth": (0, 360),
            "LMD Elevation": (0, 90),
            "LMD Apparent Elevation": (0, 90),
            "LMD Zenith": (0, 90),
            "LMD Apparent Zenith": (0, 90),

            "NWP Azimuth": (0, 360),
            "NWP Elevation": (0, 90),
            "NWP Apparent Elevation": (0, 90),
            "NWP Zenith": (0, 90),
            "NWP Apparent Zenith": (0, 90),
        }

    def get_base_pipeline(self):
        pipeline = ProcessingPipeline()
        pipeline.apply(lambda df: df.loc[df['is_daytime'], :])
        pipeline.apply(DataFrameProcessor.drop_duplicates())
        pipeline.apply(DataFrameProcessor.drop_na_columns(p_thresh=0.95))
        pipeline.apply(DataFrameProcessor.drop_na_rows())
        pipeline.apply(DataFrameProcessor.remove_low_variance_columns(
            variance_threshold=1e-10,
            cols=get_features_by_tags(self.get_feature_tags(), tags_exclude={"meta", "target", "power_proxy"})
        ))
        pipeline.apply(
            DataFrameProcessor.add_cyclical_encodings(self.get_feature_tags(), self.get_cyclical_value_ranges()))
        pipeline.apply(DataFrameProcessor.sort(by_columns="Timestamp", ascending=True))
        return pipeline

    def get_diagram_feature_name_map(self) -> dict[str, str]:
        return {
            'Timestamp': 'Time',
            'NWP Global Irradiance': 'NWP GI',
            'NWP Direct Irradiance': 'NWP DI',
            'NWP Temperature': 'NWP Temp',
            'NWP Humidity': 'NWP Hum',
            'NWP Wind Speed': 'NWP WS',
            'NWP Wind Direction': 'NWP WD',
            'NWP Pressure': 'NWP Press',
            'LMD Global Irradiance': 'LMD GI',
            'LMD Diffuse Irradiance': 'LMD DFI',
            'LMD Temperature': 'LMD Temp',
            'LMD Pressure': 'LMD Press',
            'LMD Wind Direction': 'LMD WD',
            'LMD Wind Speed': 'LMD WS',
            'Power': 'Power',
            'Station ID': 'Station',
            'Capacity': 'Capacity',
            'PV Technology': 'PV Tech',
            'Panel Size': 'Panel Size',
            'Module': 'Module',
            'Inverters': 'Inverters',
            'Layout': 'Layout',
            'Panel Number': 'Panel #',
            'Array Tilt': 'Tilt',
            'Pyranometer': 'Pyranometer',
            'Longitude': 'Lon',
            'Latitude': 'Lat',
            'Altitude': 'Alt',
            'Dawn': 'Dawn',
            'Sunrise': 'Sunrise',
            'Noon': 'Noon',
            'Sunset': 'Sunset',
            'Dusk': 'Dusk',
            'is_daytime': 'Daytime',
            'DayOfYear': 'DoY',
            'MinuteOfDay': 'MoD',

            'NWP Apparent Zenith': 'NWP AppZen',
            'NWP Zenith': 'NWP Zen',
            'NWP Apparent Elevation': 'NWP AppEl',
            'NWP Elevation': 'NWP El',
            'NWP Azimuth': 'NWP Az',
            'NWP Equation of Time': 'NWP EoT',
            'NWP GHI': 'NWP GHI',
            'NWP DNI': 'NWP DNI',
            'NWP DHI': 'NWP DHI',
            'NWP Relative Airmass': 'NWP RelAM',
            'NWP Absolute Airmass': 'NWP AbsAM',

            'LMD Apparent Zenith': 'LMD AppZen',
            'LMD Zenith': 'LMD Zen',
            'LMD Apparent Elevation': 'LMD AppEl',
            'LMD Elevation': 'LMD El',
            'LMD Azimuth': 'LMD Az',
            'LMD Equation of Time': 'LMD EoT',
            'LMD GHI': 'LMD GHI',
            'LMD DNI': 'LMD DNI',
            'LMD DHI': 'LMD DHI',
            'LMD Relative Airmass': 'LMD RelAM',
            'LMD Absolute Airmass': 'LMD AbsAM',

            # Trig features (sin/cos)
            'Longitude_sin': 'Lon (Sin)',
            'Longitude_cos': 'Lon (Cos)',
            'DayOfYear_sin': 'DoY (Sin)',
            'DayOfYear_cos': 'DoY (Cos)',
            'MinuteOfDay_sin': 'MoD (Sin)',
            'MinuteOfDay_cos': 'MoD (Cos)',

            'NWP Wind Direction_sin': 'NWP WD (Sin)',
            'NWP Wind Direction_cos': 'NWP WD (Cos)',
            'LMD Wind Direction_sin': 'LMD WD (Sin)',
            'LMD Wind Direction_cos': 'LMD WD (Cos)',

            'NWP Apparent Elevation_sin': 'NWP AppEl (Sin)',
            'NWP Apparent Elevation_cos': 'NWP AppEl (Cos)',
            'NWP Apparent Zenith_sin': 'NWP AppZen (Sin)',
            'NWP Apparent Zenith_cos': 'NWP AppZen (Cos)',
            'NWP Azimuth_sin': 'NWP Az (Sin)',
            'NWP Azimuth_cos': 'NWP Az (Cos)',
            'NWP Elevation_sin': 'NWP El (Sin)',
            'NWP Elevation_cos': 'NWP El (Cos)',
            'NWP Zenith_sin': 'NWP Zen (Sin)',
            'NWP Zenith_cos': 'NWP Zen (Cos)',

            'LMD Apparent Elevation_sin': 'LMD AppEl (Sin)',
            'LMD Apparent Elevation_cos': 'LMD AppEl (Cos)',
            'LMD Apparent Zenith_sin': 'LMD AppZen (Sin)',
            'LMD Apparent Zenith_cos': 'LMD AppZen (Cos)',
            'LMD Azimuth_sin': 'LMD Az (Sin)',
            'LMD Azimuth_cos': 'LMD Az (Cos)',
            'LMD Elevation_sin': 'LMD El (Sin)',
            'LMD Elevation_cos': 'LMD El (Cos)',
            'LMD Zenith_sin': 'LMD Zen (Sin)',
            'LMD Zenith_cos': 'LMD Zen (Cos)'
        }

    def get_task(self) -> Literal['regression', 'classification']:
        return 'regression'


def get_dataset(config: dict) -> BaseDataset:
    if config["domain"] == "wind":
        return WindDataset(config)
    elif config["domain"] == "pv":
        return SolarDataset(config)
    else:
        raise NotImplementedError(f"Dataset for domain={config['domain']} not implemented.")
