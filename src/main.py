import sys
from pathlib import Path
import argparse

# Add the src directory to the path to be able to import the modules
sys.path.append(str(Path(__file__).parent.resolve()))

from config.constants import Constants

parser = argparse.ArgumentParser(description='main.py argument parser')
group = parser.add_argument_group('Run mode')
group.add_argument('--run_single_experiment', action='store_true')
group.add_argument('--generate_report', action='store_true')
group.add_argument('--run_multiple_experiments', action='store_true')

group = parser.add_argument_group('Single experiment basic settings')
group.add_argument('--name', type=str,
                   help='Experiment name used for logging')
group.add_argument('--domain', type=str, help='The domain of the experiment. Possible values: wind, pv')
group.add_argument('--asset_id', type=str,
                   help='Turbine ID / PV site ID / ... (depending on domain) to be used for the experiment')
group.add_argument('--model', type=str,
                   help=f'The model to run the experiments for. Possible values: {Constants.MODELS}')
group.add_argument('--features', type=str,
                   help=f'One of {Constants.FEATURE_SET_TYPES}')
group.add_argument('--fs_method', type=str,
                   help=f'Feature selection method. Must be in {Constants.FS_METHODS}')
group.add_argument('--n_features', type=int,
                   help='Number of features to select')
group.add_argument('--test_ratio', type=float, default=0.25, help="How much of all data to use for testing.")
group.add_argument('--gap', type=str, default='1D', help='Gap between train and val/test sets, e.g., 1D, 12H, 30min')
group.add_argument('--overwrite', action='store_true', help='Whether to overwrite logs of existing runs.')

group = parser.add_argument_group('Single experiment CSFS and SFS settings')
group.add_argument('--fast_mode', action='store_true')
group.add_argument('--direction', type=str)
group.add_argument('--clustering_method', type=str)
group.add_argument('--group_size', type=int)
group.add_argument('--val_ratio', type=float, default=1 / 3,
                   help='How much of the training data to use for validation during feature selection.')
group.add_argument('--feature_level_hpo_mode', type=str, default='per_iteration',
                   help='HPO mode at feature selection level. Must be in ["off", "per_iteration", "per_feature_set"].')
group.add_argument('--hpo_mode', type=str,
                   help='Mode for hyperparam optimization. Must be in ["off", "per_iteration", "per_feature_set"]')
group.add_argument('--resume_at_iteration', type=int, )

group = parser.add_argument_group('Single experiment hyperparameter optimization settings')
group.add_argument('--hpo_time_budget', type=int, help='Time budget for each HPO (in s).')
group.add_argument('--hpo_max_iter', type=int, help='Maximum number of iterations for each HPO.')
group.add_argument('--hpo_early_stop', action='store_true')
group.add_argument('--hpo_train_time_limit', type=int, help='Training time limit for HPO.')

group = parser.add_argument_group('Single experiment HPO warm starts settings')
group.add_argument('--warm_starts', action='store_true',
                   help='Whether to use warm starts for HPO or not')
group.add_argument('--warmup_time_budget', type=int, help='Time budget for each warm-up HPO (in s).')
group.add_argument('--warmup_max_iter', type=int, help='Maximum number of iterations for each warm-up HPO.')
group.add_argument('--warmup_early_stop', action='store_true')

group = parser.add_argument_group('Single experiment cross-validation settings')
group.add_argument('--cv', action='store_true', help='Whether to use cross-validation or not')
group.add_argument('--n_folds', type=int, help='The number of folds for the cross-validation.')

group = parser.add_argument_group('Single experiment Evaluation/Prediction settings')
group.add_argument('--bootstrapping', action='store_true',
                   help='Whether to use test set bootstrapping or not')
group.add_argument('--n_bootstrap_samples', type=int, help='The number of bootstrap samples.')

group = parser.add_argument_group('Parallelism and randomness settings')
group.add_argument('--n_jobs', type=int, help='Number of jobs running in parallel.', default=1)
group.add_argument('--random_seed', type=int, help='The random seed set prior to the experiments.')

group = parser.add_argument_group('Report settings')
group.add_argument('--exclude_figures', action='store_true', help='Whether to overwrite logs of existing runs.')
group.add_argument('--exclude_latex', action='store_true', help='Whether to overwrite logs of existing runs.')
group.add_argument('--exclude_csv', action='store_true', help='Whether to overwrite logs of existing runs.')

group = parser.add_argument_group('Multiple experiments settings')
group.add_argument('--first_experiment_idx', type=int, help='Index of the first experiment to run.')
group.add_argument('--last_experiment_idx', type=int, help='Index of the last experiment to run.')
group.add_argument('--first_cpu_idx', type=int, help='Index of the first CPU to use for experiments.')
group.add_argument('--last_cpu_idx', type=int, help='Index of the last CPU to use for experiments.')
group.add_argument('--script_path', type=str, help='Path to the script to be executed for each experiment.')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.generate_report:
        from evaluation.generate_report import generate_report

        generate_report(
            generate_figures=not args.exclude_figures,
            generate_latex_tables=not args.exclude_latex,
            generate_csv_tables=not args.exclude_csv
        )
    elif args.run_single_experiment:
        from run_single_experiment import run_experiment

        run_experiment(args)

    elif args.run_multiple_experiments:
        from run_multiple_experiments import run_multiple_experiments

        run_multiple_experiments(args)

    else:
        print("No valid mode selected. Use --run_single_experiment, --generate_report, or --run_multiple_experiments.")
