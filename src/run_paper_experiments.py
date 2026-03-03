#!/usr/bin/env python3
"""
Paper experiments command-line tool.

Generates and manages paper experiment matrices from YAML definitions.

Usage:
    python run_paper_experiments.py list [matrix_file] [--limit N] [--stats]
    python run_paper_experiments.py run [matrix_file] [--dry-run] [--start N] [--end N]
    python run_paper_experiments.py filter [matrix_file] [--domain DOMAIN] [--model MODEL] [--limit N]
    python run_paper_experiments.py export [matrix_file] [--format FORMAT] [--output FILE]
    python run_paper_experiments.py validate [matrix_file]
"""

import sys
import os
import argparse
import logging
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from experiments import MatrixGenerator, ExperimentLauncher, ExperimentUtils

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_list(args):
    """List experiments from a matrix."""
    generator = MatrixGenerator()
    matrix = generator.load_matrix(args.matrix_file)
    experiments = generator.generate_experiments(matrix)

    if args.stats:
        stats = ExperimentUtils.get_experiment_stats(experiments)
        ExperimentUtils.print_experiment_stats(stats)
    else:
        ExperimentUtils.list_experiments(
            experiments,
            limit=args.limit,
            show_params=args.show_params,
        )

    print(f"\nTotal: {len(experiments)} experiments")


def cmd_run(args):
    """Run experiments from a matrix."""
    generator = MatrixGenerator()
    matrix = generator.load_matrix(args.matrix_file)
    experiments = generator.generate_experiments(matrix)

    launcher = ExperimentLauncher()

    results = launcher.launch_experiments(
        experiments,
        mode=args.mode,
        dry_run=args.dry_run,
        start_index=args.start,
        end_index=args.end,
    )

    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    print(f"\n{'=' * 60}")
    print("Execution Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(results, f)


def cmd_filter(args):
    """Filter experiments by parameters."""
    generator = MatrixGenerator()
    matrix = generator.load_matrix(args.matrix_file)
    experiments = generator.generate_experiments(matrix)

    # Build filter kwargs
    filter_kwargs = {}
    if args.domain:
        filter_kwargs["domain"] = args.domain
    if args.model:
        filter_kwargs["model"] = args.model
    if args.fs_method:
        filter_kwargs["fs_method"] = args.fs_method

    filtered = ExperimentUtils.filter_experiments(experiments, **filter_kwargs)

    if args.stats:
        stats = ExperimentUtils.get_experiment_stats(filtered)
        ExperimentUtils.print_experiment_stats(stats)
    else:
        ExperimentUtils.list_experiments(filtered, limit=args.limit)

    print(f"\nFiltered: {len(filtered)} / {len(experiments)} experiments")


def cmd_export(args):
    """Export experiments to a file."""
    generator = MatrixGenerator()
    matrix = generator.load_matrix(args.matrix_file)
    experiments = generator.generate_experiments(matrix)

    result = ExperimentUtils.export_experiment_list(
        experiments,
        output_format=args.format,
        filepath=args.output,
    )

    if args.output:
        print(result)
    else:
        print(result)


def cmd_validate(args):
    """Validate a matrix file."""
    generator = MatrixGenerator()

    try:
        matrix = generator.load_matrix(args.matrix_file)
        logger.info(f"Matrix loaded successfully: {args.matrix_file}")

        experiments = generator.generate_experiments(matrix)
        logger.info(f"Generated {len(experiments)} valid experiments")

        stats = ExperimentUtils.get_experiment_stats(experiments)
        ExperimentUtils.print_experiment_stats(stats)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paper experiments command-line tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments from a matrix")
    list_parser.add_argument(
        "matrix_file", help="Matrix YAML file (e.g., wind_grid.yaml)"
    )
    list_parser.add_argument(
        "--limit", type=int, help="Limit number of experiments to display"
    )
    list_parser.add_argument(
        "--stats", action="store_true", help="Show statistics instead of listing"
    )
    list_parser.add_argument(
        "--show-params", action="store_true", help="Show all parameters"
    )
    list_parser.set_defaults(func=cmd_list)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments from a matrix")
    run_parser.add_argument("matrix_file", help="Matrix YAML file")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    run_parser.add_argument(
        "--mode", default="sequential", choices=["sequential", "parallel"]
    )
    run_parser.add_argument("--start", type=int, default=0, help="Start index")
    run_parser.add_argument("--end", type=int, help="End index (exclusive)")
    run_parser.add_argument("--output", type=Path, help="Output file for results summary")
    run_parser.set_defaults(func=cmd_run)

    # Filter command
    filter_parser = subparsers.add_parser(
        "filter", help="Filter experiments by parameters"
    )
    filter_parser.add_argument("matrix_file", help="Matrix YAML file")
    filter_parser.add_argument("--domain", help="Filter by domain (wind, pv)")
    filter_parser.add_argument("--model", help="Filter by model")
    filter_parser.add_argument(
        "--fs-method", dest="fs_method", help="Filter by feature selection method"
    )
    filter_parser.add_argument("--limit", type=int, help="Limit number to display")
    filter_parser.add_argument("--stats", action="store_true", help="Show statistics")
    filter_parser.set_defaults(func=cmd_filter)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export experiments to a file")
    export_parser.add_argument("matrix_file", help="Matrix YAML file")
    export_parser.add_argument(
        "--format", default="txt", choices=["txt", "csv", "json", "yaml"]
    )
    export_parser.add_argument("--output", type=Path, help="Output file path")
    export_parser.set_defaults(func=cmd_export)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a matrix file")
    validate_parser.add_argument("matrix_file", help="Matrix YAML file")
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
