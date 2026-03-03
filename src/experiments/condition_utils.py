"""
Shared utilities for working with nested parameter dictionaries and conditions.

Used by MatrixGenerator, ConstraintValidator, and NamingEngine to avoid code duplication.
"""

from typing import Any, Dict


def get_nested_value(params: Dict[str, Any], path: str) -> Any:
    """
    Get value from nested dict using dotted path notation.
    
    Args:
        params: Dictionary to traverse
        path: Dotted path (e.g., "feature_selection.method" or "model.name")
    
    Returns:
        Value at path, or None if path doesn't exist
    
    Examples:
        >>> params = {"model": {"name": "lgbm"}, "domain": "pv"}
        >>> get_nested_value(params, "model.name")
        "lgbm"
        >>> get_nested_value(params, "domain")
        "pv"
        >>> get_nested_value(params, "nonexistent.path")
        None
    """
    # First try direct lookup (common case)
    if path in params:
        return params[path]
    
    # If no dot, it's a simple key lookup
    if "." not in path:
        return params.get(path)
    
    # Navigate nested structure
    parts = path.split(".")
    current = params
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_value(params: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set value in nested dict using dotted path notation.
    
    Args:
        params: Dictionary to modify
        path: Dotted path (e.g., "feature_selection.method")
        value: Value to set
    
    Examples:
        >>> params = {}
        >>> set_nested_value(params, "model.name", "lgbm")
        >>> params
        {"model": {"name": "lgbm"}}
    """
    if "." not in path:
        params[path] = value
        return
    
    parts = path.split(".")
    current = params
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def matches_condition(params: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    """
    Check if parameter set matches a condition (when clause).
    
    Supports:
    - Dotted path keys (e.g., "feature_selection.method")
    - List values (matches if actual value is in the list)
    - Scalar values (matches if actual value equals expected)
    
    Args:
        params: Parameter dictionary to check
        condition: Condition dictionary with key-value pairs
    
    Returns:
        True if all conditions match, False otherwise
    
    Examples:
        >>> params = {"model": {"name": "lgbm"}, "domain": "pv"}
        >>> matches_condition(params, {"model.name": "lgbm"})
        True
        >>> matches_condition(params, {"model.name": ["lgbm", "mlp"]})
        True
        >>> matches_condition(params, {"model.name": "mlp"})
        False
    """
    for key, expected_value in condition.items():
        actual_value = get_nested_value(params, key)
        
        # Handle list of possible values
        if isinstance(expected_value, list):
            if actual_value not in expected_value:
                return False
        else:
            if actual_value != expected_value:
                return False
    
    return True


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base dictionary.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
    
    Returns:
        New merged dictionary (does not modify inputs)
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "get_nested_value",
    "set_nested_value",
    "matches_condition",
    "merge_dicts",
]
