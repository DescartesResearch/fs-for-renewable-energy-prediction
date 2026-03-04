"""
Utility functions for experiment management.

Provides listing, filtering, and analysis utilities for generated experiments.
"""

import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from .condition_utils import get_nested_value


@dataclass
class ExperimentStats:
    """Statistics about a set of experiments."""

    total_count: int
    domains: Dict[str, int]
    models: Dict[str, int]
    fs_methods: Dict[str, int]
    n_features_values: Dict[int, int]
    feature_types: Dict[str, int]


class ExperimentUtils:
    """Utility functions for experiment management."""

    @staticmethod
    def filter_experiments(
        experiments: List[Dict[str, Any]],
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Filter experiments by name pattern and/or parameter values.

        Args:
            experiments: List of experiment parameter dictionaries
            include_pattern: Regex pattern to match experiment names
            exclude_pattern: Regex pattern to exclude from results
            **kwargs: Parameter key-value pairs to filter by (e.g., model="mlp", domain="wind")

        Returns:
            Filtered list of experiments
        """
        results = experiments

        # Apply parameter filters
        for key, value in kwargs.items():
            results = [e for e in results if e.get(key) == value]

        # Apply name patterns if needed
        if include_pattern or exclude_pattern:
            names = [e["name"] for e in results]

            filtered_results = []
            for name, exp in zip(names, results):
                if include_pattern and not re.search(include_pattern, name):
                    continue
                if exclude_pattern and re.search(exclude_pattern, name):
                    continue
                filtered_results.append(exp)

            results = filtered_results

        return results

    @staticmethod
    def get_experiment_stats(experiments: List[Dict[str, Any]]) -> ExperimentStats:
        """
        Generate statistics about a set of experiments.

        Args:
            experiments: List of experiment parameter dictionaries

        Returns:
            ExperimentStats object with aggregated information
        """
        stats = ExperimentStats(
            total_count=len(experiments),
            domains=defaultdict(int),
            models=defaultdict(int),
            fs_methods=defaultdict(int),
            n_features_values=defaultdict(int),
            feature_types=defaultdict(int),
        )

        for exp in experiments:
            stats.domains[exp.get("domain", "unknown")] += 1
            
            # Handle nested model structure
            model = get_nested_value(exp, "model.name")
            if model is None:
                model = exp.get("model", "unknown")
            stats.models[model] += 1
            
            # Handle nested feature selection structure
            fs_method = get_nested_value(exp, "feature_selection.method")
            if fs_method is None:
                fs_method = exp.get("fs_method", "unknown")
            stats.fs_methods[fs_method] += 1
            
            # Handle nested features structure
            n_features = get_nested_value(exp, "features.n_features")
            if n_features is None:
                n_features = exp.get("n_features", 0)
            stats.n_features_values[n_features] += 1
            
            feature_type = get_nested_value(exp, "features.type")
            if feature_type is None:
                feature_type = exp.get("features", "unknown")
            stats.feature_types[feature_type] += 1

        # Convert defaultdicts to regular dicts
        stats.domains = dict(stats.domains)
        stats.models = dict(stats.models)
        stats.fs_methods = dict(stats.fs_methods)
        stats.n_features_values = dict(stats.n_features_values)
        stats.feature_types = dict(stats.feature_types)

        return stats

    @staticmethod
    def print_experiment_stats(stats: ExperimentStats) -> None:
        """Pretty-print experiment statistics."""
        print(f"\nExperiment Statistics")
        print(f"{'=' * 50}")
        print(f"Total experiments: {stats.total_count}")
        print(f"\nBy Domain:")
        for domain, count in sorted(stats.domains.items()):
            print(f"  {domain}: {count}")
        print(f"\nBy Model:")
        for model, count in sorted(stats.models.items()):
            print(f"  {model}: {count}")
        print(f"\nBy Feature Selection Method:")
        for method, count in sorted(stats.fs_methods.items()):
            print(f"  {method}: {count}")
        print(f"\nBy Number of Features:")
        for n_features in sorted(stats.n_features_values.keys()):
            print(f"  n_features={n_features}: {stats.n_features_values[n_features]}")
        print(f"\nBy Feature Type:")
        for feat_type, count in sorted(stats.feature_types.items()):
            print(f"  {feat_type}: {count}")
        print(f"{'=' * 50}\n")

    @staticmethod
    def list_experiments(
        experiments: List[Dict[str, Any]],
        limit: Optional[int] = None,
        show_params: bool = False,
    ) -> None:
        """
        Print a list of experiments.

        Args:
            experiments: List of experiment parameter dictionaries
            limit: Maximum number to display (None for all)
            show_params: If True, show all parameters for each experiment
        """
        for i, exp in enumerate(experiments):
            if limit is not None and i >= limit:
                print(f"... and {len(experiments) - limit} more")
                break

            name = exp["name"]
            if show_params:
                print(f"{i:4d}. {name}")
                for key, value in sorted(exp.items()):
                    if key != "name":
                        print(f"       {key}: {value}")
            else:
                print(f"{i:4d}. {name}")

    @staticmethod
    def export_experiment_list(
        experiments: List[Dict[str, Any]],
        output_format: str = "txt",
        filepath: Optional[str] = None,
    ) -> str:
        """
        Export experiment list to a file.

        Args:
            experiments: List of experiment parameter dictionaries
            output_format: Format type ('txt', 'csv', 'yaml', 'json')
            filepath: Output file path (if None, returns as string)

        Returns:
            Exported content as string (if filepath is None)
        """
        import json


        if output_format == "txt":
            lines = []
            for i, exp in enumerate(experiments):
                name = exp["name"]
                lines.append(f"{i:04d} {name}")
            content = "\n".join(lines)

        elif output_format == "csv":
            # Get all unique keys
            all_keys = set()
            for exp in experiments:
                all_keys.update(exp.keys())
            all_keys = sorted(list(all_keys))

            lines = [",".join(all_keys)]
            for exp in experiments:
                values = [str(exp.get(key, "")) for key in all_keys]
                lines.append(",".join(values))
            content = "\n".join(lines)

        elif output_format == "json":
            content = json.dumps(experiments, indent=2)

        elif output_format == "yaml":
            import yaml

            content = yaml.dump(experiments, default_flow_style=False)

        else:
            raise ValueError(f"Unknown format: {output_format}")

        if filepath:
            with open(filepath, "w") as f:
                f.write(content)
            return f"Exported to {filepath}"
        else:
            return content


# Import here to avoid circular dependencies
from .matrix_generator import MatrixGenerator  # noqa: E402


__all__ = ["ExperimentUtils", "ExperimentStats"]
