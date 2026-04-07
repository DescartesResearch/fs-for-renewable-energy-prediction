import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from config.constants import Paths

# Add the src directory to the path to be able to import the modules
sys.path.append(str(Path(__file__).parent.resolve()))

@hydra.main(
    version_base=None,
    config_path=str(Paths.EXP_CONFIG_PATH),
    config_name="base"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running a single experiment.
    """
    from run_single_experiment import run_experiment

    run_experiment(cfg)


if __name__ == "__main__":
    main()

