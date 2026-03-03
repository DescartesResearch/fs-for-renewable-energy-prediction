"""
Matrix-based experiment generator.

Loads experiment parameter matrices from YAML and generates individual
experiment configurations with constraint validation.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from itertools import product
import logging

from .constraint_validator import ConstraintValidator
from .condition_utils import (
    get_nested_value,
    set_nested_value,
    matches_condition,
    merge_dicts,
)
from .naming_engine import NamingEngine

logger = logging.getLogger(__name__)


class MatrixGenerator:
    """Generates experiment configurations from YAML matrix definitions."""

    def __init__(self, matrix_dir: Optional[Path] = None, raise_on_error: bool = True):
        """
        Initialize matrix generator.

        Args:
            matrix_dir: Directory containing matrix YAML files (defaults to src/config/experiments/matrices/)
            raise_on_error: If True, raise on validation errors during generation
        """
        if matrix_dir is None:
            matrix_dir = (
                Path(__file__).parent.parent / "config" / "experiments" / "matrices"
            )
        self.matrix_dir = Path(matrix_dir)
        self.raise_on_error = raise_on_error
        self.naming_config: Optional[Dict[str, Any]] = None
        self.naming_engine: Optional[NamingEngine] = None

    def load_matrix(self, matrix_file: str) -> Dict[str, Any]:
        """
        Load a matrix definition from YAML file.

        Args:
            matrix_file: Filename (without path) or relative path in matrix_dir

        Returns:
            Parsed matrix dictionary
        """
        matrix_path = self.matrix_dir / matrix_file
        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        with open(matrix_path, "r") as f:
            matrix = yaml.safe_load(f)

        if matrix is None:
            matrix = {}

        # Handle defaults (inheritance)
        if "defaults" in matrix:
            defaults_list = matrix.pop("defaults")
            if isinstance(defaults_list, str):
                defaults_list = [defaults_list]

            merged = {}
            for default_file in defaults_list:
                # Support paths like /base.yaml for absolute or relative
                if default_file.startswith("/"):
                    default_file = default_file.lstrip("/")
                default_matrix = self.load_matrix(default_file)
                merged = merge_dicts(merged, default_matrix)

            merged = merge_dicts(merged, matrix["dimensions"] if "dimensions" in matrix else matrix)
            matrix["dimensions"] = merged

        # Extract naming configuration if present
        if "naming" in matrix:
            naming_cfg = matrix["naming"]
            if isinstance(naming_cfg, dict):
                self.naming_config = naming_cfg
                try:
                    self.naming_engine = NamingEngine(naming_cfg)
                    logger.debug("Loaded naming configuration from matrix")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize NamingEngine: {e}. Will use fallback."
                    )
                    self.naming_engine = None

        return matrix

    def generate_experiments(
        self,
        matrix: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate experiment parameter combinations from a matrix definition.

        Args:
            matrix: Matrix dictionary with 'dimensions' and optional 'constraints'
            constraints: Optional additional constraints to apply

        Returns:
            List of valid parameter dictionaries
        """
        dimensions = matrix.get("dimensions", {})
        if not dimensions:
            raise ValueError("Matrix must contain 'dimensions'")

        # Apply constraint rules from matrix and arguments
        all_constraints = matrix.get("constraints", [])
        if constraints:
            all_constraints = list(all_constraints) + constraints

        # Flatten nested dimensions to dotted paths
        flat_dimensions = self._flatten_dimensions(dimensions)

        # Generate all combinations
        keys = list(flat_dimensions.keys())
        values = [flat_dimensions[k] for k in keys]

        all_combinations = list(product(*values))

        # Convert flat combinations back to nested structure
        params_list = []
        for combo in all_combinations:
            flat_params = {keys[i]: combo[i] for i in range(len(keys))}
            nested_params = self._unflatten_params(flat_params)
            params_list.append(nested_params)

        # Apply constraints to filter invalid combinations
        valid_params_list = []
        skipped_count = 0

        for params in params_list:
            # Apply constraint rules
            filtered_params = self._apply_constraints(params, all_constraints)
            if filtered_params is None:
                skipped_count += 1
                continue

            # Auto-generate name if not provided
            generated_name = filtered_params.get("name")
            if not isinstance(generated_name, str) or not generated_name.strip() or (isinstance(generated_name, str) and generated_name.strip().lower() in ["null", "none", "???"]):
                filtered_params["name"] = self.generate_experiment_name(filtered_params, exp_nr=len(valid_params_list))

            valid_params_list.append(filtered_params)

        if skipped_count > 0:
            logger.info(
                f"Skipped {skipped_count} invalid combinations out of {len(params_list)} total"
            )

        self._assert_unique_experiment_names(valid_params_list)
        logger.info(
            f"Generated {len(valid_params_list)} valid experiment configurations"
        )

        return valid_params_list

    @staticmethod
    def _assert_unique_experiment_names(experiments: List[Dict[str, Any]]) -> None:
        """Raise ValueError when duplicate experiment names are found."""
        name_counts: Dict[str, list[int]] = {}

        for idx, experiment in enumerate(experiments):
            name = experiment.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid experiment name generated: {name!r}")
            clean_name = name.strip()
            name_counts[clean_name] = name_counts.get(clean_name, []) + [idx]

        duplicates = {name: idxs for name, idxs in name_counts.items() if len(idxs) > 1}
        if duplicates:
            raise ValueError(
                "Duplicate experiment names are not allowed. "
                f"Found duplicates: {len(duplicates)}\n"
                "Examples:"
                "Configuration 1:\n"
                f"{experiments[duplicates[next(iter(duplicates))][0]]}\n"
                "================\n"
                "Configuration 2:"
                f"{experiments[duplicates[next(iter(duplicates))][1]]}"
            )

    def _flatten_dimensions(
        self, dimensions: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, List[Any]]:
        """
        Flatten nested dimension dictionaries into dotted paths.

        Args:
            dimensions: Nested dimension dictionary
            prefix: Current path prefix

        Returns:
            Flat dictionary with dotted keys and list values

        Example:
            Input: {"model": {"name": ["mlp", "lgbm"]}, "domain": ["pv", "wind"]}
            Output: {"model.name": ["mlp", "lgbm"], "domain": ["pv", "wind"]}
        """
        flat = {}

        for key, value in dimensions.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flat.update(self._flatten_dimensions(value, full_key))
            elif isinstance(value, list):
                # This is a dimension with multiple values
                flat[full_key] = value
            else:
                # Single value - wrap in list
                flat[full_key] = [value]

        return flat

    def _unflatten_params(self, flat_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat dotted-path parameters back to nested structure.

        Args:
            flat_params: Flat dictionary with dotted keys

        Returns:
            Nested parameter dictionary

        Example:
            Input: {"model.name": "mlp", "domain": "pv"}
            Output: {"model": {"name": "mlp"}, "domain": "pv"}
        """
        nested = {}

        for key, value in flat_params.items():
            if "." in key:
                # Split path and create nested structure
                parts = key.split(".")
                current = nested
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                nested[key] = value

        return nested

    def _apply_constraints(
        self, params: Dict[str, Any], constraints: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate parameter set against constraint rules (validation-only mode).

        Constraints are purely validation rules; no updates are applied.

        Constraint format:
        {
            "when": {"param": value or [values]},  # Condition to check
            "then": {"param": value or [values], ...},  # If condition met, params must satisfy these
            "else": {"param": value or [values], ...},  # If condition NOT met, params must satisfy these
        }

        If a parameter fails validation, the entire parameter set is skipped (return None).

        Args:
            params: Current parameter set (unmodified)
            constraints: List of constraint rules

        Returns:
            params if all constraints satisfied, None if skip
        """
        for constraint in constraints:
            when = constraint.get("when", {})
            then = constraint.get("then", {})
            else_clause = constraint.get("else", {})

            matches = self._matches_condition(params, when)

            if matches:
                # Condition satisfied: validate `then` requirements
                if then and not self._validate_constraint_requirements(params, then):
                    return None
            else:
                # Condition NOT satisfied: validate `else` requirements
                if else_clause and not self._validate_constraint_requirements(
                    params, else_clause
                ):
                    return None

        return params

    @staticmethod
    def _validate_constraint_requirements(
        params: Dict[str, Any], requirements: Dict[str, Any]
    ) -> bool:
        """
        Check if params satisfy all constraint requirements.

        Args:
            params: Parameter dictionary
            requirements: {key: expected_value or [expected_values]}

        Returns:
            True if all requirements satisfied, False otherwise
        """
        for key, expected_value in requirements.items():
            actual_value = get_nested_value(params, key)

            if isinstance(expected_value, list):
                # Value must be in the list
                if actual_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict) and isinstance(actual_value, dict):
                if not MatrixGenerator._validate_constraint_requirements(
                    actual_value, expected_value
                ):
                    return False
            else:
                # Value must equal exactly
                if actual_value != expected_value:
                    return False

        return True

    @staticmethod
    def _matches_condition(params: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if parameters match a condition. Delegates to shared utility."""
        return matches_condition(params, condition)

    @staticmethod
    def _get_nested_value(params: Dict[str, Any], key: str) -> Any:
        """
        Get value for key that may use dotted notation (e.g., "model.name").
        Delegates to shared utility.
        """
        return get_nested_value(params, key)

    @staticmethod
    def _set_nested_value(params: Dict[str, Any], key: str, value: Any) -> None:
        """Set value for key that may use dotted notation. Delegates to shared utility."""
        set_nested_value(params, key, value)

    def generate_experiment_name(self, params: Dict[str, Any], exp_nr: int) -> str:
        """
        Generate a deterministic experiment name from parameters.

        If naming configuration is available (from matrix YAML), uses NamingEngine.
        Otherwise falls back to hard-coded logic for backward compatibility.

        Args:
            params: Parameter dictionary
            exp_nr: Experiment number

        Returns:
            Experiment name string
        """
        # Use NamingEngine if configured
        if self.naming_engine is not None:
            try:
                return self.naming_engine.generate_name(params)
            except Exception as e:
                logger.warning(f"NamingEngine failed: {e}. Using fallback logic.")
        return f"exp{exp_nr:04d}"


__all__ = ["MatrixGenerator"]
