"""
Experiment name generation engine with configurable naming rules.

Supports rule-based name generation from YAML configuration rather than hard-coded logic.
"""

import logging
from typing import Any, Dict, List, Optional

from .condition_utils import get_nested_value, matches_condition

logger = logging.getLogger(__name__)


class NamingEngine:
    """
    Generates experiment names based on YAML configuration rules.
    
    Configuration structure:
        naming:
          separator: "_"  # Join parts with underscore
          base_parts:
            - field: domain
            - field: asset_id
              join_with: "-"  # Join with previous part
            - field: model.name
            - field: features.n_features
              template: "n-{value}"
            - field: features.type
          conditional_parts:
            - when: {feature_selection.method: CSFS}
              parts:
                - field: feature_selection.clustering.method
                  template: "csfs-{value}"
                - field: hpo.mode
            - when: {feature_selection.method: [SFS, mutual_info, f_value, RF_FI]}
              parts:
                - field: feature_selection.method
                  transform: lowercase
    """
    
    def __init__(self, naming_config: Dict[str, Any]):
        """
        Initialize naming engine with configuration.
        
        Args:
            naming_config: Naming configuration dict from YAML
        
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = naming_config
        self.separator = naming_config.get("separator", "_")
        self.base_parts = naming_config.get("base_parts", [])
        self.conditional_parts = naming_config.get("conditional_parts", [])
        
        # Validate configuration
        if not self.base_parts:
            raise ValueError("naming configuration must include 'base_parts'")
        
        self._validate_part_specs(self.base_parts)
        for conditional_block in self.conditional_parts:
            if "when" not in conditional_block or "parts" not in conditional_block:
                raise ValueError(
                    "Each conditional_parts block must have 'when' and 'parts' keys"
                )
            self._validate_part_specs(conditional_block["parts"])
    
    def _validate_part_specs(self, parts: List[Dict[str, Any]]) -> None:
        """Validate part specifications."""
        valid_keys = {"field", "template", "transform", "join_with"}
        valid_transforms = {"lowercase", "uppercase"}
        
        for part in parts:
            if "field" not in part:
                raise ValueError(f"Each part must have a 'field' key: {part}")
            
            # Check for invalid keys
            invalid_keys = set(part.keys()) - valid_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid keys in part specification: {invalid_keys}. "
                    f"Valid keys are: {valid_keys}"
                )
            
            # Validate transform
            if "transform" in part and part["transform"] not in valid_transforms:
                raise ValueError(
                    f"Invalid transform '{part['transform']}'. "
                    f"Valid transforms are: {valid_transforms}"
                )
    
    def generate_name(self, params: Dict[str, Any]) -> str:
        """
        Generate experiment name from parameters using configured rules.
        
        Args:
            params: Parameter dictionary (may be nested)
        
        Returns:
            Generated experiment name string
        
        Examples:
            >>> config = {
            ...     "separator": "_",
            ...     "base_parts": [
            ...         {"field": "domain"},
            ...         {"field": "asset_id", "join_with": "-"},
            ...         {"field": "model.name"},
            ...     ]
            ... }
            >>> engine = NamingEngine(config)
            >>> params = {"domain": "pv", "asset_id": "01", "model": {"name": "lgbm"}}
            >>> engine.generate_name(params)
            "pv-01_lgbm"
        """
        name_parts: List[str] = []
        
        # Process base parts
        for part_spec in self.base_parts:
            part_value = self._format_part(params, part_spec)
            if part_value is not None:
                # Handle special join_with directive
                if "join_with" in part_spec and name_parts:
                    # Join with previous part instead of appending
                    name_parts[-1] = f"{name_parts[-1]}{part_spec['join_with']}{part_value}"
                else:
                    name_parts.append(part_value)
        
        # Process conditional parts
        for conditional_block in self.conditional_parts:
            when_clause = conditional_block["when"]
            if matches_condition(params, when_clause):
                for part_spec in conditional_block["parts"]:
                    part_value = self._format_part(params, part_spec)
                    if part_value is not None:
                        name_parts.append(part_value)
        
        return self.separator.join(name_parts)
    
    def _format_part(self, params: Dict[str, Any], part_spec: Dict[str, Any]) -> Optional[str]:
        """
        Format a single name part according to its specification.
        
        Args:
            params: Parameter dictionary
            part_spec: Part specification dict
        
        Returns:
            Formatted string or None if field value is None/missing
        """
        field = part_spec["field"]
        value = get_nested_value(params, field)
        
        # Skip if field is None or missing
        if value is None:
            return None
        
        # Convert to string
        value_str = str(value)
        
        # Apply template if specified
        if "template" in part_spec:
            value_str = part_spec["template"].format(value=value_str)
        
        # Apply transform if specified
        if "transform" in part_spec:
            transform = part_spec["transform"]
            if transform == "lowercase":
                value_str = value_str.lower()
            elif transform == "uppercase":
                value_str = value_str.upper()
        
        return value_str


__all__ = ["NamingEngine"]
