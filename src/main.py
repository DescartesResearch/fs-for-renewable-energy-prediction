import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add the src directory to the path to be able to import the modules
sys.path.append(str(Path(__file__).parent.resolve()))

@hydra.main(
    version_base=None,
    config_path="config/experiments",
    config_name="base"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the application.
    
    Supports three modes (single experiment mode is Hydra-enabled):
    - Single experiment: python src/main.py --config-name wind/t11_mlp_csfs
    - Multiple experiments: Not yet migrated to Hydra
    - Report generation: Not yet migrated to Hydra
    """
    from run_single_experiment import run_experiment

    run_experiment(cfg)


if __name__ == "__main__":
    main()

