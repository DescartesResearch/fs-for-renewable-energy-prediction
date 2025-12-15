from pathlib import Path


class Paths:
    """
    Clean organisation of relevant paths of the project to avoid things like "C:\\Users\\MaxMustermann\\Projects\\..."
    """
    PROJECT = Path(__file__).resolve().parents[2]
    DATASETS = PROJECT / 'datasets'
    EDP_DATASET_PATH = DATASETS / 'edp'
    PVOD_PATH = DATASETS / 'PVOD'

    LOGS = PROJECT / "logs"

    REPORT = PROJECT / "report"
    FIGURES = REPORT / "paper_figures"
    TABLES = REPORT / "paper_tables"

    MPLSTYLES = PROJECT / "src" / "mplstyles"


class Constants:
    """
    Constants used throughout the project.
    """
    CUSTOM_PARAMS = {
        'mlp': {
            'validation_fraction': 0.2,
        },
        'gp': {
            'normalize_y': True,
        }
    }
    FS_METHODS = ["SFS", "CSFS", "mutual_info", "f_value", "RF_FI"]
    DOMAINS = ["wind", "pv"]
    MODELS = ["mlp", "lgbm", "gp", "xgboost", "rf"]
    FEATURE_SET_TYPES = ["forecast_available", "digital_twin"]
    HPO_MODES = ['off', 'per_iteration', 'per_feature_set']
    CLUSTERING_METHODS = ["correlation", "random", "feature_importance", "singletons"]
    METRICS = {
        # metric to be used as primary score (higher is better)
        "neg_rmse": ("rmse", "-"),
        # additional metrics for evaluation
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
    MATPLOTLIB_USETEX = True  # Latex must be installed and be added to Path environment variable
    MATPLOTLIB_FONTFAMILY = "Libertine"  # "DejaVu Sans"
    MAPTLOTLIB_FIGURE_AUTOLAYOUT = True
    # Default seed if none provided in the specific configs of the experiments
    DEFAULT_SEED = 42
    # Number of workers used for the dataloaders
    NUM_WORKERS = 4
    # If the RAM limit is exceeded in the main.py script, the script will be aborted.
    # Only works on Linux. Set to 0 to disable the limit.
    MEMORY_LIMIT = 80  # in GB
