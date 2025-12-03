from typing import Optional, Any, Literal, Callable

import pandas as pd
from flaml.automl.model import SKLearnEstimator, RandomForestEstimator, LGBMEstimator, XGBoostEstimator
from flaml.automl.task.task import CLASSIFICATION
from flaml.automl.task.factory import task_factory
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from scipy.optimize import curve_fit

from flaml import tune, AutoML


def get_custom_flaml_estimator(estimator_name: str) -> tuple:
    """
    Get a custom FLAML estimator by name.
    :param estimator_name: Name of the estimator ('mlp' or 'gp').
    :return: Tuple of (estimator_name, estimator_class).
    """
    if estimator_name == 'mlp':
        return (estimator_name, MLPEstimator)
    elif estimator_name == 'gp':
        return (estimator_name, GaussianProcessEstimator)
    else:
        raise ValueError('Unknown estimator name "{}"'.format(estimator_name))


# ---------- helpers ----------

def _make_x_scaler(kind: str):
    """
    Create an x scaler based on the specified kind.
    :param kind: Can be "standard", "robust", "minmax", or None/"none".
    :return: The corresponding scaler or "passthrough".
    """
    if kind == "standard":
        return StandardScaler()
    elif kind == "robust":
        return RobustScaler()
    elif kind == "minmax":
        return MinMaxScaler(feature_range=(-0.5, 0.5))
    elif kind in (None, "none"):
        return "passthrough"
    raise ValueError(f"Unknown x_scaler: {kind}")


def _make_y_transformer(kind: str):
    """
    Create a y transformer based on the specified kind.
    :param kind: Can be "standard" or "minmax".
    :return: The corresponding transformer.
    """
    if kind == "standard":
        return StandardScaler()
    elif kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown y_transform: {kind}")


# ---------- sklearn-compatible wrappers ----------

class ScaledMLPClassifier(BaseEstimator, ClassifierMixin):
    """Pipeline(scaler -> MLPClassifier)."""

    def __init__(self, x_scaler: Optional[str] = None, **mlp_kwargs):
        self.x_scaler = x_scaler
        self.mlp_kwargs = {**mlp_kwargs}  # store raw kwargs for get/set_params
        if "n_jobs" in self.mlp_kwargs:
            self.mlp_kwargs.pop("n_jobs")
        if isinstance(self.mlp_kwargs.get("hidden_layer_sizes", None), str):
            self.mlp_kwargs["hidden_layer_sizes"] = tuple(map(int, self.mlp_kwargs["hidden_layer_sizes"].split("_")))

    def _build(self):
        return Pipeline([
            ("scale", _make_x_scaler(self.x_scaler)),
            ("clf", MLPClassifier(**self.mlp_kwargs)),
        ]) if self.x_scaler is not None else MLPClassifier(**self.mlp_kwargs)

    # sklearn API
    def fit(self, X, y, **fit_params):
        self.model_ = self._build()
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"x_scaler": self.x_scaler, **self.mlp_kwargs}

    def set_params(self, **params):
        self.mlp_kwargs.update(params)
        if "x_scaler" in self.mlp_kwargs:
            self.x_scaler = self.mlp_kwargs.pop("x_scaler")
        else:
            self.x_scaler = None
        if "n_jobs" in self.mlp_kwargs:
            self.mlp_kwargs.pop("n_jobs")
        # Remaining go to the inner MLP
        if isinstance(self.mlp_kwargs.get("hidden_layer_sizes", None), str):
            self.mlp_kwargs["hidden_layer_sizes"] = tuple(map(int, self.mlp_kwargs["hidden_layer_sizes"].split("_")))
        return self


class ScaledMLPRegressor(BaseEstimator, RegressorMixin):
    """TransformedTargetRegressor( regressor=Pipeline(scaler -> MLPRegressor), transformer=<y_transform> )."""

    def __init__(self, x_scaler: Optional[str] = None, y_transform: Optional[str] = None, **mlp_kwargs):
        self.x_scaler = x_scaler
        self.y_transform = y_transform
        self.mlp_kwargs = {**mlp_kwargs}
        if "n_jobs" in self.mlp_kwargs:
            self.mlp_kwargs.pop("n_jobs")
        if isinstance(self.mlp_kwargs.get("hidden_layer_sizes", None), str):
            self.mlp_kwargs["hidden_layer_sizes"] = tuple(map(int, self.mlp_kwargs["hidden_layer_sizes"].split("_")))

    def _build(self):
        reg_pipe = Pipeline([
            ("scale", _make_x_scaler(self.x_scaler)),
            ("reg", MLPRegressor(**self.mlp_kwargs)),
        ]) if self.x_scaler is not None else MLPRegressor(**self.mlp_kwargs)
        return TransformedTargetRegressor(regressor=reg_pipe, transformer=_make_y_transformer(
            self.y_transform)) if self.y_transform is not None else reg_pipe

    # sklearn API
    def fit(self, X, y, **fit_params):
        self.model_ = self._build()
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"x_scaler": self.x_scaler, "y_transform": self.y_transform, **self.mlp_kwargs}

    def set_params(self, **params):
        self.mlp_kwargs.update(params)
        if "x_scaler" in self.mlp_kwargs:
            self.x_scaler = self.mlp_kwargs.pop("x_scaler")
        else:
            self.x_scaler = None
        if "y_transform" in self.mlp_kwargs:
            self.y_transform = self.mlp_kwargs.pop("y_transform")
        else:
            self.y_transform = None
        if "n_jobs" in self.mlp_kwargs:
            self.mlp_kwargs.pop("n_jobs")
        if isinstance(self.mlp_kwargs.get("hidden_layer_sizes", None), str):
            self.mlp_kwargs["hidden_layer_sizes"] = tuple(map(int, self.mlp_kwargs["hidden_layer_sizes"].split("_")))
        return self


class ScaledGPClassifier(BaseEstimator, ClassifierMixin):
    """Pipeline(scaler -> GaussianProcessClassifier)."""

    def __init__(self, x_scaler: Optional[str] = None, **gp_kwargs):
        self.x_scaler = x_scaler
        self.gp_kwargs = {**gp_kwargs}
        if "n_jobs" in self.gp_kwargs:
            self.gp_kwargs.pop("n_jobs")

    def _build(self):
        return Pipeline([
            ("scale", _make_x_scaler(self.x_scaler)),
            ("clf", GaussianProcessClassifier(**self.gp_kwargs)),
        ]) if self.x_scaler is not None else GaussianProcessClassifier(**self.gp_kwargs)

    # sklearn API
    def fit(self, X, y, **fit_params):
        self.model_ = self._build()
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"x_scaler": self.x_scaler, **self.gp_kwargs}

    def set_params(self, **params):
        self.gp_kwargs.update(params)
        if "x_scaler" in self.gp_kwargs:
            self.x_scaler = self.gp_kwargs.pop("x_scaler")
        else:
            self.x_scaler = None
        if "n_jobs" in self.gp_kwargs:
            self.gp_kwargs.pop("n_jobs")
        # Remaining go to the inner GP
        return self


class ScaledGPRegressor(BaseEstimator, RegressorMixin):
    """TransformedTargetRegressor( regressor=Pipeline(scaler -> GaussianProcessRegressor), transformer=<y_transform> )."""

    def __init__(self, x_scaler: Optional[str] = None, y_transform: Optional[str] = None, **gp_kwargs):
        self.x_scaler = x_scaler
        self.y_transform = y_transform
        self.gp_kwargs = {**gp_kwargs}
        if "n_jobs" in self.gp_kwargs:
            self.gp_kwargs.pop("n_jobs")

    def _build(self):
        reg_pipe = Pipeline([
            ("scale", _make_x_scaler(self.x_scaler)),
            ("reg", GaussianProcessRegressor(**self.gp_kwargs)),
        ]) if self.x_scaler is not None else GaussianProcessRegressor(**self.gp_kwargs)
        return TransformedTargetRegressor(regressor=reg_pipe, transformer=_make_y_transformer(
            self.y_transform)) if self.y_transform is not None else reg_pipe

    # sklearn API
    def fit(self, X, y, **fit_params):
        self.model_ = self._build()
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"x_scaler": self.x_scaler, "y_transform": self.y_transform, **self.gp_kwargs}

    def set_params(self, **params):
        self.gp_kwargs.update(params)
        if "x_scaler" in self.gp_kwargs:
            self.x_scaler = self.gp_kwargs.pop("x_scaler")
        else:
            self.x_scaler = None
        if "y_transform" in self.gp_kwargs:
            self.y_transform = self.gp_kwargs.pop("y_transform")
        else:
            self.y_transform = None
        if "n_jobs" in self.gp_kwargs:
            self.gp_kwargs.pop("n_jobs")
        return self


class MLPEstimator(SKLearnEstimator):
    """
    Custom FLAML estimator for MLPs, since FLAML does not provide one by default.
    Following the FLAML Tutorial: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/#estimator-and-search-space
    """

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)

        if task in CLASSIFICATION:
            self.estimator_class = ScaledMLPClassifier
        else:
            self.estimator_class = ScaledMLPRegressor

    @classmethod
    def search_space(cls, data_size, task, **params):
        upper_iter = 512  # max number of iterations for MLP

        space = {
            # Architecture
            "hidden_layer_sizes": {
                "domain": tune.choice(["8", "16", "32", "64", "100", "128", "8_8", "16_8", "32_16", "32_8", "64_16"]),
                "low_cost_init_value": "8",
                "init_value": "100",
            },
            "activation": {
                "domain": tune.choice(["relu", "tanh", "logistic"]),
                "init_value": "relu",
            },

            # Optimization
            "solver": {
                "domain": tune.choice(["adam", "sgd", "lbfgs"]),
                "init_value": "adam",
            },
            "alpha": {  # L2 regularization
                "domain": tune.loguniform(lower=1e-4, upper=1e-2),
                "init_value": 1e-4,
            },
            # Only used by solver='sgd' (harmless to keep in config)
            "learning_rate": {
                "domain": tune.choice(["constant", "invscaling", "adaptive"]),
                "init_value": "constant",
            },
            "learning_rate_init": {
                "domain": tune.loguniform(lower=1e-4, upper=1e-2),
                "init_value": 1e-3,
            },
            "power_t": {  # for 'invscaling'
                "domain": tune.uniform(lower=0.1, upper=0.9),
                "init_value": 0.5,
            },
            # Compute budget per fit
            "max_iter": {
                "domain": tune.lograndint(lower=64, upper=upper_iter, base=2),
                "low_cost_init_value": 64,
                "init_value": 200,
            },
            "tol": {
                "domain": tune.loguniform(lower=1e-4, upper=1e-2),
                "init_value": 1e-4,
            },
            "momentum": {
                "domain": tune.uniform(lower=0.5, upper=1.0),
                "init_value": 0.9,
            },
            "nesterovs_momentum": {
                "domain": tune.choice([True, False]),
                "init_value": True,
            },
            "early_stopping": {
                "domain": tune.choice([True, False]),
                "init_value": False,
            },
            "validation_fraction": {
                "domain": tune.uniform(lower=0.05, upper=0.2),
                "init_value": 0.1,
            },
            "beta_1": {
                "domain": tune.uniform(lower=0.8, upper=0.999999),
                "init_value": 0.9,
            },
            "beta_2": {
                "domain": tune.uniform(lower=0.95, upper=0.999999),
                "init_value": 0.999,
            },
            "epsilon": {
                "domain": tune.loguniform(lower=1e-9, upper=1e-5),
                "init_value": 1e-8,
            },
            "n_iter_no_change": {
                "domain": tune.qrandint(lower=10, upper=30, q=10),
                "low_cost_init_value": 10,
            },
            "max_fun": {
                "domain": tune.randint(lower=int(1e4), upper=int(3e4)),
                "low_cost_init_value": 1e4,
                "init_value": 1.5e4,
            },
            "shuffle": {
                "domain": tune.choice([True, False]),
                "init_value": True,
            },
            "x_scaler": {
                "domain": "minmax"  # tune.choice([# "standard", "minmax"]),
            },
            "y_transform": {
                "domain": tune.choice(["standard", "minmax"]),
            }
        }

        return space


class GaussianProcessEstimator(SKLearnEstimator):
    """
    Custom FLAML estimator for Gaussian Processes, since FLAML does not provide one by default.
    Following the FLAML Tutorial: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/#estimator-and-search-space
    """

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)

        if task in CLASSIFICATION:
            self.estimator_class = ScaledGPClassifier
        else:
            self.estimator_class = ScaledGPRegressor

    @classmethod
    def search_space(cls, data_size, task, **params):
        if task in CLASSIFICATION:
            # see: https://github.com/hyperopt/hyperopt-sklearn/blob/1.1.1/hpsklearn/components/gaussian_process/_gpc.py
            space = ...
            raise NotImplementedError(f"The search space for the task {task} is not implemented.")
        else:
            # see: https://github.com/hyperopt/hyperopt-sklearn/blob/1.1.1/hpsklearn/components/gaussian_process/_gpr.py
            space = {
                "alpha": {"domain": tune.loguniform(1e-10, 1e-2),
                          "init_value": 1e-10, },
                "n_restarts_optimizer": {"domain": tune.randint(0, 6),
                                         "init_value": 0,
                                         "low_cost_init_value": 0},
                "x_scaler": {"domain": "minmax"  # tune.choice(["standard", "minmax"]),
                             }
            }
        return space


class PowerCurveWrapper:
    """
    A simple wrapper for simple baseline wind turbine power curve functions.
    Not used in the CAiSE26 paper due to space constraints.

    The wrapped function maps an array of floats (e.g., wind speeds)
    to an array of floats (e.g., power outputs).
    """

    def __init__(self, power_curve: Callable[[pd.DataFrame | pd.Series | np.ndarray], pd.Series]):
        self._power_curve = power_curve

    def predict(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """
        Predict power output for the given inputs.
        :param X: The input data (e.g., wind speeds) for the power curve callable.
        :return: Numpy array of predicted power values.
        """

        if isinstance(X, pd.DataFrame):
            if len(X.shape) > 1 and X.shape[1] > 1:
                raise ValueError()
            X = X.squeeze()
        elif isinstance(X, np.ndarray):
            if len(X.shape) > 1 and X.shape[1] > 1:
                raise ValueError()
            X = pd.Series(X.ravel())

        if hasattr(self._power_curve, "predict"):
            return self._power_curve.predict(X)
        else:
            return self._power_curve(X)


class MHTan:
    """
    Multi-Hyperbolic Tangent function fitter and predictor.
    A custom model that fits a multi-hyperbolic tangent function to data.
    Typically used as a baseline model for wind turbine power curves.
    Not used in the CAiSE26 paper due to space constraints.
    """

    def __init__(self, p0: Optional[list[float]] = None, Xscale: float = 1., yscale: float = 1.):
        self.popt = None
        self.pcov = None
        self.p0 = p0
        self.Xscale = Xscale
        self.yscale = yscale

    def fit(self, X: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
        if self.popt is not None:
            raise RuntimeError("Estimator has already been fitted.")
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.popt, self.pcov = curve_fit(self._mhtan, X / self.Xscale, y / self.yscale, self.p0,
                                         maxfev=30000,
                                         method='lm')

    def predict(self, X: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.values
        return self._mhtan(X / self.Xscale, *self.popt) * self.yscale

    @staticmethod
    def _mhtan(xdata: np.ndarray, a1: float, a2: float, a3: float, a4: float, a5: float,
               a6: float, a7: float, a8: float, a9: float) -> np.ndarray:
        return (a1 * np.exp(a2 * xdata) - a3 * np.exp(-a4 * xdata)) / (
                a5 * np.exp(a6 * xdata) + a7 * np.exp(-a8 * xdata)) + a9


def get_automl_with_registered_models(models_to_register: list[str]) -> AutoML:
    """
    Create an AutoML instance and register custom FLAML estimators.
    :param models_to_register: List of model names to register. Must be in CUSTOM_FLAML_ESTIMATORS.
    :return: AutoML instance with registered models.
    """
    automl = AutoML()
    automl_estims_to_register = [get_custom_flaml_estimator(estimator_name) for estimator_name in
                                 models_to_register if estimator_name in CUSTOM_FLAML_ESTIMATORS]
    for learner_name_class_tuple in automl_estims_to_register:
        automl.add_learner(*learner_name_class_tuple)
    return automl


def flaml_config2params(estimator_name: str, task: str, config: dict) -> dict:
    """
    Convert FLAML config to estimator parameters.
    :param estimator_name: Name of the estimator.
    :param task: Task type ('regression' or 'classification').
    :param config: FLAML hyperparameter configuration, e.g., best hyperparameters found by FLAML
    :return: Dictionary ready to use as parameters for the sklearn estimator.
    """
    if config is None:
        return {}
    elif estimator_name == "rf":
        return RandomForestEstimator(task=task_factory(task, None, None)).config2params(config)
    elif estimator_name == "mlp":
        return MLPEstimator(task=task).config2params(config)
    elif estimator_name == "gp":
        return GaussianProcessEstimator(task=task).config2params(config)
    elif estimator_name == "lgbm":
        return LGBMEstimator(task=task).config2params(config)
    elif estimator_name == "xgboost":
        return XGBoostEstimator(task=task).config2params(config)
    else:
        raise NotImplementedError(f"The estimator {estimator_name} has not been implemented.")


def get_sklearn_estim(estimator_name: str, task: Literal['regression', 'classification'],
                      best_flaml_config: Optional[dict]) -> RegressorMixin | ClassifierMixin:
    """
    Get a sklearn-compatible estimator instance based on the name and task.
    :param estimator_name: Name of the estimator.
    :param task: Task type ('regression' or 'classification').
    :param best_flaml_config: Best hyperparameter configuration found by FLAML. Optional. If not provided, default parameters are used.
    :return:
    """
    estim_class = SKLEARN_ESTIMATORS.get(f"{estimator_name}_{task[:3]}", None)
    if estim_class is None:
        raise NotImplementedError(f"The estimator {estimator_name} for task={task} has not been implemented.")
    config = flaml_config2params(estimator_name, task, best_flaml_config)
    return estim_class(**config)


def needs_cyclical_encoding(model_name: str) -> bool:
    """Check if the model requires cyclical encoding based on its name."""
    return (model_name not in TREE_BASED_MODELS) and (model_name in MULTIVARIATE_MODELS)


CUSTOM_FLAML_ESTIMATORS = ['mlp', 'gp']
SKLEARN_ESTIMATORS = {
    'mlp_reg': ScaledMLPRegressor,
    'gp_reg': ScaledGPRegressor,
    'lgbm_reg': LGBMRegressor,
    'xgboost_reg': XGBRegressor,
    'mlp_cla': ScaledMLPClassifier,
    'gp_cla': ScaledGPClassifier,
    'lgbm_cla': LGBMClassifier,
    'xgboost_cla': XGBClassifier,
    'rf_reg': RandomForestRegressor,
    'rf_cla': RandomForestClassifier,
}
TREE_BASED_MODELS = ["lgbm", "xgboost", "rf"]
MULTIVARIATE_MODELS = ["mlp", "lgbm", "xgboost", "rf"]
UNIVARIATE_BASELINES = ["svr", "iec", "logistic", "polynomial", "spline", "gp"]
