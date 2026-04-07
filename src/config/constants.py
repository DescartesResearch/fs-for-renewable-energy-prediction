from pathlib import Path

import dotenv
import os


class Paths:
    """
    Clean organisation of relevant paths of the project to avoid things like "C:\\Users\\MaxMustermann\\Projects\\..."
    """

    PROJECT = Path(__file__).resolve().parents[2]

    # Load environment variables from .env file
    dotenv.load_dotenv(PROJECT / ".env", override=True)

    LOGS = Path(os.getenv("LOGS_DIR", "logs"))
    DATASETS = Path(os.getenv("DATASETS_DIR", "datasets"))
    EDP_DATASET_PATH = DATASETS / "edp"
    PVOD_PATH = DATASETS / "PVOD"
    REPORT = Path(os.getenv("REPORT_DIR", "report"))
    LITERATURE_STUDY = Path(os.getenv("LITERATURE_STUDY_DIR", "literature_study"))

    EXP_CONFIG_PATH = Path(os.getenv("EXP_CONFIG_DIR", "exp_configs"))

    MPLSTYLES = PROJECT / "src" / "mplstyles"


class Constants:
    """
    Constants used throughout the project.
    """

    CUSTOM_PARAMS = {
        "mlp": {
            "validation_fraction": 0.2,
        },
        "gp": {
            "normalize_y": True,
        },
    }
    FS_METHODS = ["SFS", "CSFS", "mutual_info", "f_value", "RF_FI"]
    DOMAINS = ["wind", "pv"]
    MODELS = ["mlp", "lgbm", "gp", "xgboost", "rf"]
    FEATURE_SET_TYPES = ["forecast_available", "digital_twin"]
    HPO_MODES = ["off", "on", "per_iteration", "per_feature_set"]
    CLUSTERING_METHODS = ["correlation", "random", "feature_importance", "singletons"]
    METRICS = {
        # metrics to be used as primary score (higher is better)
        "neg_rmse": ("rmse", "-"),
        "neg_mse": ("mse", "-"),
        # additional metrics for evaluation
        "mse": ("mse", "+"),
        "r2": ("r2", "+"),
        "mae": ("mae", "+"),
        "rmse": ("rmse", "+"),
        "mdae": ("mdae", "+"),
        "me": ("me", "+"),
        "mde": ("mde", "+"),  # median error
        "ame": ("ame", "+"),
        "amde": ("amde", "+"),  # absolute median error
    }

    # Set this to True to log to the File System
    USE_FS_LOGGER = True
    # Setting for Matplotlib figures
    MATPLOTLIB_USETEX = (
        True  # Latex must be installed and be added to Path environment variable
    )
    MATPLOTLIB_FONTFAMILY = "Libertine"  # "DejaVu Sans"
    MAPTLOTLIB_FIGURE_AUTOLAYOUT = True
    # Default seed if none provided in the specific configs of the experiments
    DEFAULT_SEED = 42
