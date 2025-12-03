import pickle
import shutil
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, Dict, Any, Literal, Mapping
import atexit
from abc import ABC, abstractmethod

import pandas as pd
import lightning.pytorch as pl
import numpy as np
from numpy.typing import ArrayLike
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.fabric.utilities.logger import _convert_params, _sanitize_callable_params
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Module

from config.constants import Constants, Paths
from training.neptune_utils import NeptuneUtils
from utils.eval import is_better_than
from utils.misc import flatten_dict, unflatten_dict, parse_name, CPUUnpickler

if Constants.USE_NEPTUNE_LOGGER:
    import neptune
    from neptune.types import File
    from neptune.utils import stringify_unsupported


class ExtendedLogger(Logger, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_figure(self, key: str, fig: plt.Figure, img_format: str = "pdf") -> None:
        pass

    @abstractmethod
    def fetch_values(self, key: str) -> Optional[np.ndarray | float | int]:
        pass

    @abstractmethod
    def fetch_all_values(self, key: str) -> Optional[dict]:
        pass

    @abstractmethod
    def fetch_all_objects(self, key: str) -> Optional[dict]:
        pass

    @abstractmethod
    def copy_all_objects(self, key: str, dest_key: str) -> None:
        pass

    @abstractmethod
    def save_string(self, key: str, string: str) -> None:
        pass

    @abstractmethod
    def fetch_string(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def save_object(self, key: str, obj: Any) -> None:
        pass

    @abstractmethod
    def fetch_object(self, key: str, return_obj_path: bool = False) -> Optional[Any]:
        pass

    @property
    @abstractmethod
    def save_dir(self) -> Optional[Path]:
        pass

    @property
    @abstractmethod
    def root_dir(self) -> Optional[Path]:
        pass

    @property
    @abstractmethod
    def log_dir(self) -> Optional[Path]:
        pass

    @abstractmethod
    def log_model_summary(self, model: pl.LightningModule, max_depth: int = -1):
        pass

    @abstractmethod
    def get_hyperparams(self) -> Optional[dict]:
        pass

    @abstractmethod
    def log_best_hyperparams(self, objective: tuple[str, str], versions: list[str]) -> None:
        pass

    @abstractmethod
    def get_best_hyperparams(self) -> Optional[dict]:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def log_metrics_array(self, metrics: Dict[str, np.ndarray | float], step: Optional[int] = None) -> None:
        pass


class DummyLogger(ExtendedLogger):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> Optional[str]:
        return "DummyLogger"

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_metrics_array(self, metrics: Dict[str, np.ndarray | float], step: Optional[int] = None) -> None:
        pass

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    def log_figure(self, key: str, fig: plt.Figure, img_format: str = "pdf") -> None:
        pass

    def fetch_values(self, key: str) -> Optional[np.ndarray | float | int]:
        return None

    def fetch_all_values(self, key: str) -> Optional[dict]:
        return None

    def fetch_all_objects(self, key: str) -> Optional[dict]:
        return None

    def copy_all_objects(self, key: str, dest_key: str) -> None:
        pass

    def save_string(self, key: str, string: str) -> None:
        pass

    def fetch_string(self, key: str) -> Optional[str]:
        return None

    def save_object(self, key: str, obj: Any) -> None:
        pass

    def fetch_object(self, key: str, return_obj_path: bool = False) -> Optional[Any]:
        return None

    @property
    def save_dir(self) -> Optional[Path]:
        return None

    @property
    def root_dir(self) -> Optional[Path]:
        return None

    @property
    def log_dir(self) -> Optional[Path]:
        return None

    def log_model_summary(self, model: pl.LightningModule, max_depth: int = -1):
        pass

    def get_hyperparams(self) -> Optional[dict]:
        return None

    def log_best_hyperparams(self, objective: tuple[str, str], versions: list[str]) -> None:
        pass

    def get_best_hyperparams(self) -> Optional[dict]:
        return None


class CompositeLogger(ExtendedLogger):

    def __init__(self, loggers: list["ExtendedLogger"]):
        self.loggers = loggers
        # check that all loggers have the same name and version, then set the name and version
        name = None
        version = None
        for logger in loggers:
            name = logger.name
            if name is None:
                name = logger.name
            else:
                if name != logger.name:
                    raise ValueError("All loggers must have the same name.")
            if version is None:
                version = logger.version
            else:
                if version != logger.version:
                    raise ValueError("All loggers must have the same version.")
        super().__init__()
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    # Implementation of abstract methods of ExtendedLogger abstract class
    @rank_zero_only
    def log_figure(self, key: str, fig: plt.Figure, img_format: str = "pdf") -> None:
        for logger in self.loggers:
            logger.log_figure(key, fig, img_format)

    @rank_zero_only
    def fetch_values(self, key: str) -> Optional[np.ndarray | float | int]:
        for logger in self.loggers:
            value = logger.fetch_values(key)
            if value is not None:
                return value
        return None

    @rank_zero_only
    def fetch_all_values(self, key: str) -> Optional[dict]:
        for logger in self.loggers:
            d = logger.fetch_all_values(key)
            if d is not None:
                return d
        return None

    @rank_zero_only
    def fetch_all_objects(self, key: str) -> Optional[dict]:
        for logger in self.loggers:
            d = logger.fetch_all_objects(key)
            if d is not None:
                return d
        return None

    @rank_zero_only
    def copy_all_objects(self, key: str, dest_key: str) -> None:
        for logger in self.loggers:
            logger.copy_all_objects(key, dest_key)

    @rank_zero_only
    def save_string(self, key: str, string: str) -> None:
        for logger in self.loggers:
            logger.save_string(key, string)

    @rank_zero_only
    def fetch_string(self, key: str) -> Optional[str]:
        for logger in self.loggers:
            string = logger.fetch_string(key)
            if string is not None:
                return string
        return None

    @rank_zero_only
    def save_object(self, key: str, obj: Any) -> None:
        for logger in self.loggers:
            logger.save_object(key, obj)

    @rank_zero_only
    def fetch_object(self, key: str, return_obj_path: bool = False) -> Optional[Any]:
        for logger in self.loggers:
            obj = logger.fetch_object(key, return_obj_path)
            if obj is not None:
                return obj
        return None

    @rank_zero_only
    def get_hyperparams(self) -> Optional[dict]:
        for logger in self.loggers:
            hyperparams = logger.get_hyperparams()
            if hyperparams is not None:
                return hyperparams
        return None

    @rank_zero_only
    def log_best_hyperparams(self, objective: tuple[str, str], versions: list[str]) -> None:
        for logger in self.loggers:
            logger.log_best_hyperparams(objective, versions)

    @rank_zero_only
    def get_best_hyperparams(self) -> Optional[dict]:
        for logger in self.loggers:
            best_hyperparams = logger.get_best_hyperparams()
            if best_hyperparams is not None:
                return best_hyperparams
        return None

    # Implementation of methods from Logger abstract class:

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    @rank_zero_only
    def log_metrics_array(self, metrics: Dict[str, np.ndarray | float], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_metrics_array(metrics, step)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        for logger in self.loggers:
            logger.log_hyperparams(params)

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        for logger in self.loggers:
            logger.after_save_checkpoint(checkpoint_callback)

    @property
    def save_dir(self) -> Optional[Path]:
        for logger in self.loggers:
            if logger.save_dir is not None:
                return logger.save_dir
        return None

    @property
    def root_dir(self) -> Optional[Path]:
        for logger in self.loggers:
            if logger.root_dir is not None:
                return logger.root_dir
        return None

    @property
    def log_dir(self) -> Optional[Path]:
        for logger in self.loggers:
            if logger.log_dir is not None:
                return logger.log_dir
        return None

    @property
    def group_separator(self) -> str:
        return "/"

    @rank_zero_only
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        for logger in self.loggers:
            logger.log_graph(model, input_array)

    @rank_zero_only
    def save(self) -> None:
        for logger in self.loggers:
            logger.save()

    @rank_zero_only
    def finalize(self, status: Optional[str]) -> None:
        for logger in self.loggers:
            logger.finalize(status)

    @rank_zero_only
    def log_model_summary(self, model: pl.LightningModule, max_depth: int = -1):
        for logger in self.loggers:
            logger.log_model_summary(model, max_depth)


class FSLogger(ExtendedLogger):
    """
    Logger that logs to the file system.
    """

    def __init__(self, name: str, version: Optional[str] = None, save_directory: Optional[Path] = Paths.LOGS):
        super().__init__()
        self._name = name
        self._version = version
        self._save_dir = save_directory
        self._root_dir = save_directory / name
        self._log_dir = self.root_dir / version if version is not None else self.root_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # data structures to hold logged data before saving / after loading
        self.metrics_values = defaultdict(list)
        self.metrics_arrays_values = dict()
        self.metrics_steps = defaultdict(list)
        self.current_steps = defaultdict(lambda: 0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def save_dir(self) -> Optional[Path]:
        return self._save_dir

    @property
    def root_dir(self) -> Optional[Path]:
        return self._root_dir

    @property
    def log_dir(self) -> Optional[Path]:
        return self._log_dir

    @rank_zero_only
    def log_figure(self, key: str, fig: plt.Figure, img_format: str = "pdf") -> None:
        p = self.log_dir / key
        p = p.with_name(f"{p.name}.{img_format}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)

    @rank_zero_only
    def fetch_values(self, key: str) -> Optional[np.ndarray | float | int]:
        if key in self.metrics_values and key in self.metrics_arrays_values:
            raise ValueError(f"Key {key} exists in both metrics_values and metrics_arrays_values.")
        values = self.metrics_values.get(key, None)
        if values is None:
            values = self.metrics_arrays_values.get(key, None)
        if values is None:
            # try to load from file
            values = self._load_values_from_file(key)
        if isinstance(values, list):
            values = np.array(values)
        if values is None:
            return None
        if len(values.shape) == 0 and values.size == 1:
            return values.item()
        elif len(values.shape) == 1 and values.size == 1:
            return values[0]
        elif values.size > 1:
            return values
        else:
            raise ValueError(f"Unknown values format for key {key}: {values}")

    def _load_values_from_file(self, key: str) -> Optional[np.ndarray | float | int]:
        p = self.log_dir / key
        p = p.with_name(f"{p.name}.npy")
        if p.is_file():
            values = np.load(p, allow_pickle=True)
            return values
        return None

    @rank_zero_only
    def fetch_all_values_from_disk(self, key: str) -> Optional[dict]:
        res = dict()
        # search recursively for all .npy files in log_dir / key
        for p in (self.log_dir / key).rglob(f"*.npy"):
            curr_key = str(p.relative_to(self.log_dir).with_suffix(""))
            values = self.fetch_values(curr_key)
            if values is not None:
                res[str(p.relative_to(self.log_dir / key).with_suffix(""))] = values
        if len(res) == 0:
            return None
        return unflatten_dict(res, sep=self.group_separator)

    @rank_zero_only
    def fetch_all_values(self, key: str) -> Optional[dict]:
        res = {

            **{p.replace(key + self.group_separator, ""): v for p, v in self.metrics_values.items() if
               p.startswith(key + self.group_separator)},
            **{p.replace(key + self.group_separator, ""): v for p, v in self.metrics_arrays_values.items() if
               p.startswith(key + self.group_separator)}

        }

        return unflatten_dict(res, sep=self.group_separator)

    @rank_zero_only
    def fetch_all_objects(self, key: str) -> Optional[dict]:
        res = dict()
        for p in (self.log_dir / key).rglob(f"*.pkl"):
            # save to res with key being key/* without .npy and without log_dir
            with open(p, "rb") as f:
                res[str(p.relative_to(self.log_dir / key).with_suffix(""))] = pickle.load(f)
        if len(res) == 0:
            return None
        return unflatten_dict(res, sep=self.group_separator)

    @rank_zero_only
    def copy_all_objects(self, origin_key: str, dest_key: str) -> None:
        for p in (self.log_dir / origin_key).rglob(f"*.pkl"):
            rel_path = p.relative_to(self.log_dir / origin_key)
            dest_path = self.log_dir / dest_key / rel_path
            # if dest_path.exists():
            #     raise ValueError(f"Destination path {dest_path} already exists.")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, self.log_dir / dest_key / rel_path)

    @rank_zero_only
    def save_string(self, key: str, string: str) -> None:
        p = self.log_dir / key
        p = p.with_name(f"{p.name}.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(string)

    @rank_zero_only
    def fetch_string(self, key: str) -> Optional[str]:
        p = self.log_dir / key
        p = p.with_name(f"{p.name}.txt")
        if p.is_file():
            with open(p, "r") as f:
                return f.read()
        return None

    @rank_zero_only
    def save_object(self, key: str, obj: Any) -> None:
        p = self.log_dir / key
        p = p.with_name(f"{p.name}.pkl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(obj, f)

    @rank_zero_only
    def fetch_object(self, key: str, return_obj_path: bool = False) -> Optional[Any]:
        for suffix in [".pkl", ".ckpt"]:
            if (p := (self.log_dir / key).with_name(f"{(self.log_dir / key).name}{suffix}")).is_file():
                break
        if p.is_file():
            if return_obj_path:
                return p
            with open(p, "rb") as f:
                try:
                    return pickle.load(f)
                except RuntimeError as e:
                    print(f"Error loading object from {p}: {e}")
                    print("Trying to load with CPU_Unpickler.")
                    return CPUUnpickler(f).load()
                except Exception as e:
                    print(f"Error loading object from {p}: {e}")
                    return None
        return None

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.metrics_values[key].append(value)
            if step is not None:
                self.metrics_steps[key].append(step)
            else:
                self.metrics_steps[key].append(self.current_steps[key])
                self.current_steps[key] += 1

    @rank_zero_only
    def log_metrics_array(
            self,
            metrics: Mapping[str, ArrayLike],
            step: Optional[int] = None,
            *,
            axis: int = 0,
            on_conflict: Literal["extend", "overwrite", "error"] = "extend",
    ) -> None:
        """
        Log metric arrays, optionally extending existing entries along `axis`.

        - Scalars are converted to 1-D arrays (shape (1,)).
        - If a key already exists:
            * "extend": concatenate along `axis` (requires shape match on other axes)
            * "overwrite": replace the stored value
            * "error": raise ValueError
        """

        def _as_array(x: ArrayLike) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 0:
                arr = arr.reshape(1)  # scalar -> (1,)
            return arr

        def _compatible_for_concat(a: np.ndarray, b: np.ndarray, axis: int) -> bool:
            if a.ndim != b.ndim:
                return False
            # Normalize negative axis
            ax = axis if axis >= 0 else a.ndim + axis
            return all(
                (i == ax) or (a.shape[i] == b.shape[i])
                for i in range(a.ndim)
            )

        for key, value in metrics.items():
            new = _as_array(value)

            if key not in self.metrics_arrays_values:
                self.metrics_arrays_values[key] = new
                continue

            # Key exists
            if on_conflict == "overwrite":
                self.metrics_arrays_values[key] = new
            elif on_conflict == "error":
                raise ValueError(
                    f"Key {key} already exists. You probably already logged a value for this key."
                )
            elif on_conflict == "extend":
                old = self.metrics_arrays_values[key]
                old = _as_array(old)  # just in case something scalar slipped in
                if not _compatible_for_concat(old, new, axis):
                    raise ValueError(
                        f"Cannot extend key '{key}': shapes incompatible for concatenation "
                        f"along axis {axis}: old {old.shape} vs new {new.shape}."
                    )
                self.metrics_arrays_values[key] = np.concatenate([old, new], axis=axis)
            else:
                raise ValueError(f"Unknown on_conflict='{on_conflict}'.")

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        if isinstance(params, Namespace):
            params = vars(params)
        self.save_object("hyperparams", params)

    @rank_zero_only
    def get_hyperparams(self) -> Optional[dict]:
        return self.fetch_object("hyperparams")

    @rank_zero_only
    def get_best_hyperparams(self) -> Optional[dict]:
        if self.version is not None:
            raise ValueError("get_best_hyperparams is only supported for 'top level' (version must be None).")
        if self.fetch_object(f"best_hyperparams") is not None:
            return self.fetch_object(f"best_hyperparams")
        return None

    @rank_zero_only
    def log_best_hyperparams(self, objective: tuple[str, str], versions: list[str]) -> None:
        if self.version is not None:
            raise ValueError("get_best_hyperparams is only supported for 'top level' (version must be None).")
        if self.fetch_object(f"best_hyperparams") is not None:
            return

        best_value = None
        best_hyperparams = None
        best_version = None
        for version in versions:
            curr_prefix = version + "/"
            if (curr_hyperparams := self.fetch_object(f"{curr_prefix}hyperparams")) is None:
                return None
            if (curr_best_value := self.fetch_values(f"{curr_prefix}{objective[0]}")) is None:
                return None
            if isinstance(curr_best_value, np.ndarray):
                if objective[1] == "max":
                    curr_best_value = curr_best_value.max()
                elif objective[1] == "min":
                    curr_best_value = curr_best_value.min()
                else:
                    raise ValueError(f"Unknown objective[1] value: {objective[1]}. Must be 'max' or 'min'.")
            if is_better_than(curr_best_value, best_value, objective[1]):
                best_value = curr_best_value
                best_hyperparams = curr_hyperparams
                best_version = version

        if best_hyperparams is not None:
            self.save_object(f"best_hyperparams", best_hyperparams)
            self.save_string(f"best_version", best_version)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()
        self.save_string("status", status)

    @rank_zero_only
    def save(self):
        for kv in [self.metrics_values.items(), self.metrics_arrays_values.items()]:
            for key, values in kv:
                p = self.log_dir / key
                p = p.with_name(f"{p.name}.npy")
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if hasattr(values, "__len__"):
                        np.save(p, np.array(values) if len(values) > 1 else values[0])
                    else:
                        np.save(p, values)
                except Exception as e:
                    warnings.warn(f"Could not save {p}: {e}")

    @rank_zero_only
    def log_model_summary(self, model: pl.LightningModule, max_depth: int = -1):
        model_str = str(ModelSummary(model=model, max_depth=max_depth))
        self.save_string("model_summary", model_str)


class ExtendedNeptuneLogger(NeptuneLogger, ExtendedLogger):

    def __init__(self, name: str, version: Optional[str] = None, neptune_run: Optional["neptune.Run"] = None,
                 log_model_checkpoints: bool = False):
        ExtendedLogger.__init__(self)

        if not Constants.USE_NEPTUNE_LOGGER:
            raise RuntimeError("Neptune logger is not enabled. Set Constants.USE_NEPTUNE_LOGGER = True to use it.")

        neptune_run = neptune_run or NeptuneUtils.get_run_by_name(name)
        neptune_logger_kwargs = {}
        if neptune_run is not None:
            if not isinstance(neptune_run, neptune.Run):
                raise ValueError("neptune_run must be an instance of neptune.Run.")
            if neptune_run.get_state() != "started":
                raise ValueError("Neptune run must be in 'started' state.")
            if neptune_run[
                "sys/name"].fetch() != name:
                raise ValueError("Passed Neptune run's name must match the experiment name.")
        else:
            neptune_logger_kwargs.update({
                "project": Constants.NEPTUNE_LOGGER_PROJECT,
                "api_key": Constants.NEPTUNE_LOGGER_API_TOKEN,
                "name": name,
                "dependencies": "infer",
                "capture_stdout": False,
                "capture_stderr": False,
                "capture_hardware_metrics": False,
                # "mode": "debug" if config.get("debug", False) else Constants.NEPTUNE_LOGGER_MODE,
            })

        NeptuneLogger.__init__(self,
                               run=neptune_run,
                               log_model_checkpoints=log_model_checkpoints,
                               prefix=version,
                               **neptune_logger_kwargs)

        self.experiment.sync()

        self.prefix = version + "/" if version is not None else ""
        self._name = name
        self._version = version
        self.log_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._at_exit)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @rank_zero_only
    def log_metrics_array(self, metrics: Dict[str, np.ndarray | float], step: Optional[int] = None) -> None:
        raise NotImplementedError("Logging of metrics array not implemented yet for ExtendedNeptuneLogger.")

    @rank_zero_only
    def log_figure(self, key: str, fig: plt.Figure, img_format: str = "pdf") -> None:
        self.experiment[self.prefix + key].upload(File.as_image(fig))

    @rank_zero_only
    def fetch_values(self, key: str) -> Optional[np.ndarray | float | int]:
        try:
            return self.experiment[self.prefix + key].fetch()
        except:
            pass
        try:
            values = self.experiment[self.prefix + key].fetch_values().values[:, 0]  # [:, 1] contains the timestamps
            if len(values) == 1:
                return values[0]
            elif len(values) > 1:
                return values
        except:
            pass
        return None

    @rank_zero_only
    def fetch_all_values(self, key: str) -> Optional[dict]:
        # def resolve_neptune_paths(self, run: neptune.Run, flattened_structure: dict, return_path: bool = False,
        #                           key_prefix="") -> dict:
        result = dict()
        flattened_structure = flatten_dict(self.experiment.get_structure(), sep=self.group_separator)
        flattened_structure = {k: v for k, v in flattened_structure.items() if k.startswith(self.prefix + key)}
        for k, v in flattened_structure.items():
            k = k[len(self.prefix):]
            # if type(v).__name__ == "File":
            #     result[k] = self.download_file(run, f"{key_prefix}{k}", return_path=return_path)
            curr_values = self.fetch_values(k)
            if curr_values is not None:
                result[k[len(key) + 1:]] = curr_values
        if len(result) == 0:
            return None
        return unflatten_dict(result, sep=self.group_separator)

    @rank_zero_only
    def save_string(self, key: str, string: str) -> None:
        self.experiment[self.prefix + key] = string

    @rank_zero_only
    def fetch_string(self, key: str) -> Optional[str]:
        if not self.experiment.exists(self.prefix + key):
            return None
        return self.experiment[self.prefix + key].fetch()

    @rank_zero_only
    def save_object(self, key: str, obj: Any) -> None:
        self.experiment[self.prefix + key].upload(File.as_pickle(obj))

    @rank_zero_only
    def fetch_object(self, key: str, return_obj_path: bool = False) -> Optional[Any]:
        obj_path = self._download_file(key)
        if obj_path is None:
            return None
        if return_obj_path:
            return obj_path
        with open(obj_path, "rb") as f:
            if obj_path.suffix == ".npy" or obj_path.suffix == ".npz" or obj_path.suffix == ".np" or obj_path.suffix == ".bin":
                obj = np.load(f)
            elif obj_path.suffix == ".pkl":
                obj = pickle.load(f)
            else:
                raise ValueError(f"Unknown file extension: {obj_path.suffix}")
        return obj

    @rank_zero_only
    def fetch_all_objects(self, key: str, return_obj_path: bool = False) -> Optional[dict]:
        result = dict()
        exp_structure = self.experiment.get_structure()
        exp_structure = flatten_dict(exp_structure, sep=self.group_separator)
        exp_structure = {k.replace(self.prefix, ""): v for k, v in exp_structure.items() if
                         k.startswith(self.prefix + key)}
        for k, v in exp_structure.items():
            if type(v).__name__ == "File":
                if (obj := self.fetch_object(k, return_obj_path=return_obj_path)) is not None:
                    result[k[len(key) + 1:]] = obj
            # elif "Series" in type(v).__name__:
            #     result[k] = run[f"{key_prefix}{k}"].fetch_values()['value'].values
            # else:
            #     result[k] = run[f"{key_prefix}{k}"].fetch()
        if len(result) == 0:
            return None
        return unflatten_dict(result, sep=self.group_separator)

    @rank_zero_only
    def copy_all_objects(self, key: str, dest_key: str) -> None:
        raise NotImplementedError()

    @rank_zero_only
    def save(self) -> None:
        super(NeptuneLogger, self).save()
        self.experiment.sync()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super(NeptuneLogger, self).finalize(status)
        self.save_string("status", status)
        self.save()

    @property
    def save_dir(self) -> Optional[Path]:
        return Paths.TMP / "neptune"

    @property
    def root_dir(self) -> Optional[Path]:
        return self.save_dir / self.name

    @property
    def log_dir(self) -> Optional[Path]:
        # Call the NeptuneLogger's log_dir property
        return self.root_dir if self.version is None else self.root_dir / self.version

    @rank_zero_only
    def get_hyperparams(self) -> Optional[dict]:
        if not self.experiment.exists(self.prefix + "hyperparams"):
            return None
        config = self.experiment[self.prefix + "hyperparams"].fetch()

        return self._eval_dict(config)

    @rank_zero_only
    def get_best_hyperparams(self) -> Optional[dict]:
        if self.version is not None:
            raise ValueError("get_best_hyperparams is only supported for 'top level' (version must be None).")
        if self.experiment.exists(f"best_hyperparams"):
            return self._eval_dict(self.experiment[f"best_hyperparams"].fetch())
        return None

    @rank_zero_only
    def log_best_hyperparams(self, objective: tuple[str, str], versions: list[str]) -> None:
        if self.version is not None:
            raise ValueError("get_best_hyperparams is only supported for 'top level' (version must be None).")
        if self.experiment.exists(f"best_hyperparams"):
            return

        best_value = None
        best_version = None
        for version in versions:
            curr_prefix = version + "/"
            if not self.experiment.exists(f"{curr_prefix}hyperparams"):
                return None
            if not self.experiment.exists(f"{curr_prefix}{objective[0]}"):
                return None
            curr_best_value = self.fetch_values(f"{curr_prefix}{objective[0]}")
            if isinstance(curr_best_value, np.ndarray):
                if objective[1] == "max":
                    curr_best_value = curr_best_value.max()
                elif objective[1] == "min":
                    curr_best_value = curr_best_value.min()
                else:
                    raise ValueError(f"Unknown objective[1] value: {objective[1]}. Must be 'max' or 'min'.")
            if is_better_than(curr_best_value, best_value, objective[1]):
                best_value = curr_best_value
                best_version = version

        if best_version is None:
            return None

        best_hyperparams = self._eval_dict(self.experiment[best_version + "/hyperparams"].fetch())

        best_hyperparams = _convert_params(best_hyperparams)
        best_hyperparams = _sanitize_callable_params(best_hyperparams)

        parameters_key = "best_hyperparams"
        # parameters_key = self._construct_path_with_prefix(parameters_key)

        self.experiment[parameters_key + self.prefix] = stringify_unsupported(best_hyperparams)

        # self.experiment[f"best_hyperparams"] = best_hyperparams
        self.save_string(f"best_version", best_version)

    def _eval_dict(self, d: dict) -> dict:
        d = flatten_dict(d)
        for k, v in d.items():
            if isinstance(v, str) and (v.startswith("[") or v.startswith("(") or v == "None"):
                d[k] = eval(v)
        d = unflatten_dict(d)
        return d

    def _download_file(self, key: str) -> Optional[Path]:
        if not self.experiment.exists(self.prefix + key):
            return None
        file_extension = self.experiment[self.prefix + key].fetch_extension()
        dest: Path = self.log_dir / f"{key}.{file_extension}"
        dest.parent.mkdir(exist_ok=True, parents=True)
        self.experiment[self.prefix + key].download(str(dest))
        self.experiment.wait()
        return dest

    @rank_zero_only
    def _at_exit(self) -> None:
        # shutil.rmtree(self.log_dir)
        self.experiment.stop()


def get_logger(name: str, version: Optional[str] = None):
    loggers = []
    if Constants.USE_FS_LOGGER:
        loggers.append(FSLogger(name=name, version=version))
    if Constants.USE_NEPTUNE_LOGGER:
        loggers.append(ExtendedNeptuneLogger(name=name, version=version))
    logger = CompositeLogger(
        loggers
    )
    return logger


def get_results_df(names: list[str], ds_typs: Optional[list[str]] = None) -> pd.DataFrame:
    if ds_typs is None:
        ds_typs = ["train", "val", "test"]
    results = defaultdict(list)
    for name in names:
        results["name"].append(name)
        logger = get_logger(name, None)
        for agg_type in ["mean", "std"]:
            for ds_typ in ds_typs:
                for key, value in flatten_dict(logger.fetch_all_values(f"metrics/{agg_type}/{ds_typ}"),
                                               sep=logger.group_separator).items():
                    results[f"{agg_type}/{ds_typ}/" + key].append(value)
    df = pd.DataFrame(results)
    # apply the parse_name function to the name column
    df = df.join(df["name"].apply(parse_name).apply(pd.Series))
    return df
