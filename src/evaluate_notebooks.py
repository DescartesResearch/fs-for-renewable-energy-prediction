import papermill as pm

from config.constants import Paths

import logging

logging.info("Evaluating notebook 1: Literature Review")
pm.execute_notebook(
    "src/notebooks/01_literature_review.ipynb",
    Paths.REPORT / "01_literature_review.ipynb",
)

logging.info("Evaluating notebook 2: Experiment Results")
pm.execute_notebook(
    "src/notebooks/02_experiment_results.ipynb",
    Paths.REPORT / "02_experiment_results.ipynb",
)
