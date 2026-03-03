"""
Experiment launcher for running matrix-generated experiments.

Handles sequential and parallel execution of experiments with CPU-aware resource
management, experiment selection, and result tracking.
"""

import logging
import subprocess
import os
import threading
import time
import psutil
import queue
from typing import Dict, List, Any, Optional, Callable, Set
from pathlib import Path
from dataclasses import dataclass

from .matrix_generator import MatrixGenerator
from utils.misc import flatten_dict

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment execution."""

    index: int
    name: str
    params: Dict[str, Any]
    exit_code: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    success: bool = False


class ExperimentLauncher:
    """Launches and manages experiment execution with CPU-aware resource scheduling."""

    def __init__(self, src_root: Optional[Path] = None, min_memory_bytes: Optional[int] = None):
        """
        Initialize launcher with resource management.

        Args:
            src_root: Path to src directory (defaults to parent of experiments module)
            min_memory_bytes: Minimum available memory required before launching experiment.
                            If None, defaults to 70GB (75161927680 bytes).
        """
        if src_root is None:
            src_root = Path(__file__).parent.parent
        self.src_root = Path(src_root)
        self.generator = MatrixGenerator()
        
        # Resource management
        self.min_memory_bytes = min_memory_bytes or (10 * (1024 ** 3))  # 70GB default
        self.cpu_count = os.cpu_count() or 1
        self.cpu_range = list(range(self.cpu_count))
        self.running_experiments: Dict[int, Dict[str, Any]] = {}  # pid -> {name, cpus, params}
        self.lock = threading.RLock()  # RLock allows nested acquisition by same thread
        self.condition = threading.Condition(self.lock)
        
        # Parallel execution
        self.results_queue: queue.Queue = queue.Queue()
        self.monitoring_threads: List[threading.Thread] = []
        
        # Progress tracking
        self.total_experiments = 0
        self.launched_count = 0
        self.completed_count = 0
        self.failed_count = 0

    def _get_free_cpus(self) -> List[int]:
        """
        Get list of free CPU cores.

        Returns:
            List of available CPU core IDs
        """
        with self.lock:
            used_cpus: Set[int] = set()
            for pid, exp_data in self.running_experiments.items():
                used_cpus.update(exp_data.get('cpus', []))
            free_cpus = [cpu for cpu in self.cpu_range if cpu not in used_cpus]
        return free_cpus

    def _has_enough_memory(self, min_memory_bytes: Optional[int] = None) -> bool:
        """
        Check if system has enough available memory.

        Returns:
            True if available memory >= min_memory_bytes, False otherwise
        """
        required_memory = min_memory_bytes or self.min_memory_bytes
        memory_info = psutil.virtual_memory()
        return memory_info.available >= required_memory

    def _try_allocate_resources(
        self,
        n_cpus_needed: int = 1,
        min_memory_bytes: Optional[int] = None,
    ) -> Optional[List[int]]:
        """
        Try to allocate resources without blocking.

        Args:
            n_cpus_needed: Number of CPU cores needed
            min_memory_bytes: Minimum memory required

        Returns:
            List of allocated CPU core IDs if available, None otherwise
        """
        if n_cpus_needed > self.cpu_count:
            return None

        required_memory = min_memory_bytes or self.min_memory_bytes
        free_cpus = self._get_free_cpus()
        has_memory = self._has_enough_memory(required_memory)

        if len(free_cpus) >= n_cpus_needed and has_memory:
            allocated = free_cpus[:n_cpus_needed]
            return allocated
        
        return None

    def _wait_for_resources(
        self,
        n_cpus_needed: int = 1,
        min_memory_bytes: Optional[int] = None,
    ) -> List[int]:
        """
        Wait for sufficient resources to become available.

        Args:
            n_cpus_needed: Number of CPU cores needed
            min_memory_bytes: Minimum memory required

        Returns:
            List of allocated CPU core IDs

        Raises:
            ValueError: If n_cpus_needed > total CPU count
        """
        if n_cpus_needed > self.cpu_count:
            raise ValueError(
                f"Requested {n_cpus_needed} CPUs but only {self.cpu_count} available"
            )

        required_memory = min_memory_bytes or self.min_memory_bytes
        poll_interval = 5  # Reduced from 60 for faster responsiveness

        while True:
            with self.condition:
                free_cpus = self._get_free_cpus()
                has_memory = self._has_enough_memory(required_memory)

                if len(free_cpus) >= n_cpus_needed and has_memory:
                    # Allocate and return the first n_cpus_needed cores
                    allocated = free_cpus[:n_cpus_needed]
                    logger.info(
                        f"Resources available: allocated CPUs {allocated}, "
                        f"memory {psutil.virtual_memory().available / (1024**3):.1f}GB"
                    )
                    return allocated

                resource_status = f"need {n_cpus_needed} CPUs (have {len(free_cpus)})"
                if not has_memory:
                    resource_status += f", insufficient memory (need {required_memory / (1024**3):.1f}GB)"
                
                logger.debug(f"Resources unavailable: {resource_status}. Waiting {poll_interval}s...")
                
                # Release lock while waiting so other threads can acquire it
                self.condition.wait(timeout=poll_interval)

    def _monitor_process(
        self,
        process: subprocess.Popen,
        exp_index: int,
        name: str,
        cpus: List[int],
        params: Dict[str, Any],
        callback: Optional[Callable[[ExperimentResult], None]] = None,
    ) -> None:
        """
        Monitor a process and collect results when complete.

        Args:
            process: Popen process object to monitor
            exp_index: Experiment index in batch
            name: Experiment name
            cpus: Allocated CPU cores
            params: Experiment parameters
            callback: Optional callback to invoke on completion
        """
        try:
            # Wait for process completion
            exit_code = process.wait()
            pid = process.pid
            
            # Create result
            result = ExperimentResult(
                index=exp_index,
                name=name,
                params=params,
                exit_code=exit_code,
                stdout=None,
                stderr=None,
                success=(exit_code == 0),
            )
            
            # Log completion
            logger.log(
                logging.INFO if exit_code == 0 else logging.ERROR,
                f"Experiment {name} (PID {pid}) "
                f"{'completed' if exit_code == 0 else 'FAILED'} "
                f"(exit code: {exit_code})",
            )
            
            # Queue result for collection
            self.results_queue.put(result)
            
            # Invoke callback if provided
            if callback:
                callback(result)
                
        except Exception as e:
            logger.error(f"Error monitoring process for {name}: {e}")
            # Queue failed result
            result = ExperimentResult(
                index=exp_index,
                name=name,
                params=params,
                exit_code=-1,
                stdout=None,
                stderr=f"Monitoring error: {e}",
                success=False,
            )
            self.results_queue.put(result)
        finally:
            # Clean up: remove from running_experiments and notify waiting threads
            with self.condition:
                if process.pid in self.running_experiments:
                    del self.running_experiments[process.pid]
                    logger.info(
                        f"Cleaned up experiment {name} (PID {process.pid}), freed CPUs {cpus}"
                    )
                
                # CRITICAL: Notify all waiting threads that resources may now be available
                self.condition.notify_all()

    def launch_experiment(
        self,
        params: Dict[str, Any],
        exp_index: int = 0,
        dry_run: bool = False,
        hydra_overrides: Optional[List[str]] = None,
        min_memory_bytes: Optional[int] = None,
        callback: Optional[Callable[[ExperimentResult], None]] = None,
    ) -> Optional[threading.Thread]:
        """
        Launch a single experiment with CPU pinning and resource availability checks.

        Args:
            params: Experiment parameters (n_jobs extracted for CPU allocation)
            exp_index: Index of experiment in batch (for result tracking)
            dry_run: If True, print command without executing
            hydra_overrides: Additional Hydra overrides (e.g., ["hydra.run.dir=/tmp"])
            min_memory_bytes: Override memory requirement for this experiment
            callback: Optional callback invoked when experiment completes

        Returns:
            Monitoring thread object (for parallel mode), or None (for dry_run or sequential).
            Thread is already started; caller should join() it.
        """
        # Use per-experiment memory requirement if provided, otherwise use instance default
        memory_requirement = min_memory_bytes or params.get("min_memory_bytes") or self.min_memory_bytes
        
        name = params["name"]
        
        # Auto-detect n_jobs from params for CPU allocation
        n_jobs = params.get("n_jobs", 1)
        n_cpus_needed = max(1, min(n_jobs, self.cpu_count))
        
        # Build Hydra CLI command
        cmd = ["uv", "run", "python", str(self.src_root / "main.py")]

        # Add config name if available
        config_name = params.pop("config_name", None)
        if config_name:
            cmd.extend(["--config-name", config_name])

        # Flatten nested parameters and add as Hydra overrides
        flat_params = flatten_dict(params, sep=".")
        for key, value in flat_params.items():
            cmd.append(f"{key}={self._format_value(value)}")

        # Add additional Hydra overrides
        if hydra_overrides:
            cmd.extend(hydra_overrides)

        if dry_run:
            logger.info(f"[DRY RUN] {' '.join(cmd)}")
            return None

        # Wait for sufficient resources
        logger.debug(f"Preparing to launch {name}, waiting for {n_cpus_needed} CPUs and {memory_requirement / (1024**3):.1f}GB memory...")
        allocated_cpus = self._wait_for_resources(
            n_cpus_needed=n_cpus_needed,
            min_memory_bytes=memory_requirement,
        )
        cpu_list = ",".join(map(str, allocated_cpus))
        
        # Pin to CPUs using taskset
        cmd = ["taskset", "--cpu-list", cpu_list] + cmd
        
        logger.debug(f"Launching experiment: {name} (CPUs: {cpu_list})")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Launch non-blocking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Track in running_experiments
        with self.condition:
            self.running_experiments[process.pid] = {
                'name': name,
                'cpus': allocated_cpus,
                'params': params,
            }
        
        # Spawn monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(process, exp_index, name, allocated_cpus, params, callback),
            daemon=False,
            name=f"monitor-{name}",
        )
        monitor_thread.start()
        self.monitoring_threads.append(monitor_thread)
        
        return monitor_thread

    def _shutdown(self) -> None:
        """
        Gracefully shutdown all running experiments.
        
        Attempts to terminate processes, waits briefly, then kills if needed.
        """
        with self.condition:
            running_pids = list(self.running_experiments.keys())
        
        if not running_pids:
            return
        
        logger.warning(f"Shutting down {len(running_pids)} running experiments...")
        
        # First pass: terminate gracefully
        for pid in running_pids:
            try:
                process = psutil.Process(pid)
                process.terminate()
                logger.info(f"Sent SIGTERM to process {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Wait for graceful shutdown
        time.sleep(5)
        
        # Second pass: kill remaining processes
        for pid in running_pids:
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    process.kill()
                    logger.warning(f"Sent SIGKILL to process {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        logger.info("Shutdown complete")

    def _print_progress(self) -> None:
        """
        Log real-time progress of experiment batch.
        Thread-safe; uses lock to read counters.
        """
        with self.lock:
            remaining = self.total_experiments - self.completed_count
        logger.info(
            f"Progress: [{self.completed_count}/{self.total_experiments} ✓ | "
            f"{self.failed_count} ✗ | {remaining} remaining]"
        )

    def _launch_experiments_parallel(
        self,
        experiments: List[Dict[str, Any]],
        start_index: int = 0,
        end_index: Optional[int] = None,
        callback: Optional[Callable[[ExperimentResult], None]] = None,
    ) -> List[ExperimentResult]:
        """
        Launch multiple experiments in parallel with resource-aware scheduling.

        Args:
            experiments: List of experiment parameter dictionaries
            start_index: Start executing from this experiment index
            end_index: Stop executing at this experiment index (exclusive)
            callback: Optional callback function called after each experiment

        Returns:
            List of ExperimentResult objects sorted by submission index
        """
        if end_index is None:
            end_index = len(experiments)

        experiments_to_run = experiments[start_index:end_index]
        num_experiments = len(experiments_to_run)
        
        logger.info(f"Launching {num_experiments} experiments in parallel mode")
        
        # Clear results queue at start
        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except queue.Empty:
                break
        
        self.monitoring_threads.clear()
        
        # Initialize progress tracking
        with self.lock:
            self.total_experiments = num_experiments
            self.launched_count = 0
            self.completed_count = 0
            self.failed_count = 0
        
        try:
            # Dispatch all experiments (may block in launch_experiment if resources unavailable)
            for i, exp in enumerate(experiments_to_run):
                exp_index = start_index + i
                try:
                    # Create progress-tracking callback that wraps user callback
                    def make_progress_callback(user_cb):
                        def progress_callback(result: ExperimentResult):
                            with self.lock:
                                self.completed_count += 1
                                if not result.success:
                                    self.failed_count += 1
                            self._print_progress()
                            if user_cb:
                                user_cb(result)
                        return progress_callback
                    
                    self.launch_experiment(
                        exp,
                        exp_index=exp_index,
                        callback=make_progress_callback(callback),
                    )
                    
                    with self.lock:
                        self.launched_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to launch experiment {i}: {e}")
                    # Create failed result to maintain result count
                    result = ExperimentResult(
                        index=exp_index,
                        name=f"experiment-{i}",
                        params=exp,
                        exit_code=-1,
                        stdout=None,
                        stderr=f"Launch error: {e}",
                        success=False,
                    )
                    self.results_queue.put(result)
                    with self.lock:
                        self.completed_count += 1
                        self.failed_count += 1
                    self._print_progress()
                time.sleep(0.5)  # Stagger launches slightly to avoid resource contention spikes
            
            # Wait for all monitoring threads to complete
            logger.info(f"All {num_experiments} experiments launched, waiting for completion...")
            for thread in self.monitoring_threads:
                thread.join()
            
            logger.info("All monitoring threads completed")
            
        except KeyboardInterrupt:
            logger.warning("Interrupted! Shutting down running experiments...")
            self._shutdown()
            # Wait for threads to finish cleanup
            for thread in self.monitoring_threads:
                thread.join(timeout=10)
            raise
        
        # Collect results from queue and sort by index
        results_by_index: Dict[int, ExperimentResult] = {}
        results_count = 0
        while not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                results_by_index[result.index] = result
                results_count += 1
            except queue.Empty:
                break
        
        logger.info(f"Collected {results_count}/{num_experiments} results from queue")
        logger.info(f"Final Results: {self.completed_count - self.failed_count} succeeded, {self.failed_count} failed")
        
        # Return results sorted by submission order
        results = [results_by_index[i] for i in range(start_index, end_index) if i in results_by_index]
        
        return results

    def launch_experiments(
        self,
        experiments: List[Dict[str, Any]],
        mode: str = "sequential",
        dry_run: bool = False,
        start_index: int = 0,
        end_index: Optional[int] = None,
        callback: Optional[Callable[[ExperimentResult], None]] = None,
    ) -> List[ExperimentResult]:
        """
        Launch multiple experiments with resource-aware scheduling.

        Args:
            experiments: List of experiment parameter dictionaries
            mode: Execution mode ('sequential' or 'parallel')
            dry_run: If True, don't actually execute
            start_index: Start executing from this experiment index
            end_index: Stop executing at this experiment index (exclusive)
            callback: Optional callback function called after each experiment

        Returns:
            List of ExperimentResult objects
        """
        if end_index is None:
            end_index = len(experiments)

        experiments_to_run = experiments[start_index:end_index]

        # Handle dry-run mode for both sequential and parallel
        if dry_run:
            results = []
            for i, exp in enumerate(experiments_to_run):
                self.launch_experiment(exp, exp_index=start_index + i, dry_run=True)
                result = ExperimentResult(
                    index=start_index + i,
                    name=exp["name"],
                    params=exp,
                    exit_code=0,
                    stdout="[DRY RUN - No execution]",
                    success=True,
                )
                results.append(result)
                if callback:
                    callback(result)
            return results

        # Route to appropriate execution mode
        if mode == "parallel":
            return self._launch_experiments_parallel(
                experiments,
                start_index=start_index,
                end_index=end_index,
                callback=callback,
            )
        else:
            # Sequential mode (backward compatible)
            results = []
            for i, exp in enumerate(experiments_to_run):
                # In sequential mode, block until each experiment completes
                monitor_thread = self.launch_experiment(
                    exp,
                    exp_index=start_index + i,
                    callback=callback,
                )
                if monitor_thread:
                    monitor_thread.join()
            
            # Collect results from queue
            while not self.results_queue.empty():
                try:
                    result = self.results_queue.get_nowait()
                    results.append(result)
                except queue.Empty:
                    break
            
            # Sort by index to maintain order
            results.sort(key=lambda r: r.index)
            return results

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for Hydra CLI."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Quote strings with special characters
            if any(c in value for c in [" ", "=", ","]):
                return f'"{value}"'
            return value
        elif value is None:
            return "null"
        else:
            return str(value)


__all__ = ["ExperimentLauncher", "ExperimentResult"]
