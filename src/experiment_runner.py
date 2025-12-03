import logging
import os
import subprocess
import time
from pathlib import Path
from threading import Thread, Lock

from config.constants import Paths
import psutil


def has_enough_memory(min_free_bytes=70 * 1024 ** 3):
    mem = psutil.virtual_memory()
    return mem.available >= min_free_bytes


logging.basicConfig(level=logging.INFO)


class ExperimentRunner:
    def __init__(self, cpu_range, log_dir, script_path, n_jobs=4):
        self.cpu_range = list(cpu_range)
        self.log_dir = Path(log_dir)
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
        env["CUDA_VISIBLE_DEVICES"] = "MIG-a1208c4e-caad-5519-9d69-6b0998c74b9f"
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


if __name__ == "__main__":
    experiment_indices = range(560)
    cpu_range = range(0, 32)
    n_parallel_experiments = 4
    log_dir = Paths.LOGS
    script_path = Paths.PROJECT / "run_experiments.sh"
    runner = ExperimentRunner(cpu_range=cpu_range,
                              log_dir=log_dir,
                              script_path=script_path,
                              n_jobs=n_parallel_experiments)
    runner.run(experiment_indices)
