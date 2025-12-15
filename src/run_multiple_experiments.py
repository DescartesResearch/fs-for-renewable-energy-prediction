import logging
import os
import subprocess
import time
from pathlib import Path
from threading import Thread, Lock

import psutil


def has_enough_memory(min_free_bytes=70 * 1024 ** 3):
    mem = psutil.virtual_memory()
    return mem.available >= min_free_bytes


logging.basicConfig(level=logging.INFO)


class ExperimentRunner:
    def __init__(self, cpu_range, script_path, n_jobs=4):
        self.cpu_range = list(cpu_range)
        self.script_path = script_path
        self.n_jobs = n_jobs
        self.lock = Lock()
        self.running_experiments = {}

    def get_free_cpus(self):
        with self.lock:
            used_cpus = {cpu for cpus in self.running_experiments.values() for cpu in cpus}
            return [cpu for cpu in self.cpu_range if cpu not in used_cpus]

    def start_experiment(self, experiment_index, cpus):
        logging.info(f"Starting experiment {experiment_index} with CPUs {cpus}")

        env = os.environ.copy()
        env["N_JOBS"] = str(self.n_jobs)
        cpu_list = ",".join(map(str, cpus))
        command = f"taskset --cpu-list {cpu_list} bash {self.script_path} {experiment_index}"

        process = subprocess.Popen(command, shell=True, env=env)
        with self.lock:
            self.running_experiments[process.pid] = cpus

        def monitor_process():
            process.wait()
            with self.lock:
                del self.running_experiments[process.pid]

        Thread(target=monitor_process, daemon=True).start()

    def run(self, experiment_indices):
        experiment_indices = set(experiment_indices)
        while experiment_indices:
            free_cpus = self.get_free_cpus()
            if len(free_cpus) >= self.n_jobs and has_enough_memory():
                cpus_to_use = free_cpus[:self.n_jobs]
                experiment_index = experiment_indices.pop()
                self.start_experiment(experiment_index, cpus_to_use)
                time.sleep(10)  # Brief pause to allow system stabilization
            else:
                time.sleep(60)

        # Wait for all running experiments to finish
        while self.running_experiments:
            time.sleep(1)


def validate_args(args):
    num_cpus = os.cpu_count()
    if args.first_cpu_idx < 0 or args.last_cpu_idx >= num_cpus:
        raise ValueError(f"CPU indices must be between 0 and {num_cpus - 1}.")
    if args.first_cpu_idx is None or args.last_cpu_idx is None:
        raise ValueError("Both first_cpu_idx and last_cpu_idx must be specified.")
    if args.first_cpu_idx > args.last_cpu_idx:
        raise ValueError("First CPU index must be less than or equal to last CPU index.")

    # Same for experiment indices
    if args.first_experiment_idx is None or args.last_experiment_idx is None:
        raise ValueError("Both first_experiment_idx and last_experiment_idx must be specified.")
    if args.first_experiment_idx > args.last_experiment_idx:
        raise ValueError("First experiment index must be less than or equal to last experiment index.")
    if args.first_experiment_idx < 0 or args.last_experiment_idx < 0:
        raise ValueError("Experiment indices must be non-negative.")

    if not Path(args.script_path).is_file():
        raise ValueError(f"Script path {args.script_path} does not exist or is not a file.")

    if args.n_jobs <= 0:
        raise ValueError("Number of parallel jobs must be a positive integer.")


def run_multiple_experiments(args):
    runner = ExperimentRunner(cpu_range=range(args.first_cpu_idx, args.last_cpu_idx + 1),
                              script_path=Path(args.script_path),
                              n_jobs=args.n_jobs)
    runner.run(range(args.first_experiment_idx, args.last_experiment_idx + 1))
