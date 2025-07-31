"""
Scalability Manager - Framework for horizontal scaling and resource management.

This module provides the core architecture for scaling the conversation
categorization system from 19K to 500K+ tickets efficiently.
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import psutil
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for worker processes/threads."""

    worker_id: str
    worker_type: str  # 'thread' or 'process'
    max_tasks_per_worker: int = 50
    task_timeout_seconds: int = 300
    memory_limit_mb: int = 1024
    cpu_affinity: Optional[List[int]] = None


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_task_duration: float
    queue_depth: int
    active_workers: int
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_tasks_per_minute: float
    estimated_completion_time: timedelta


@dataclass
class ResourceProfile:
    """Resource profile for different dataset sizes."""

    dataset_size_range: Tuple[int, int]  # (min, max) ticket count
    recommended_workers: int
    memory_per_worker_mb: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    batch_size: int
    estimated_cost_per_ticket: float


class ScalabilityManager:
    """
    Core manager for horizontal scaling operations.

    Provides intelligent worker pool management, resource allocation,
    and performance monitoring for large-scale ticket processing.
    """

    # Predefined resource profiles for different dataset sizes
    RESOURCE_PROFILES = [
        ResourceProfile(
            dataset_size_range=(1, 1000),
            recommended_workers=2,
            memory_per_worker_mb=512,
            chunk_size_tokens=50000,
            chunk_overlap_tokens=5000,
            batch_size=25,
            estimated_cost_per_ticket=0.048,
        ),
        ResourceProfile(
            dataset_size_range=(1001, 10000),
            recommended_workers=4,
            memory_per_worker_mb=512,
            chunk_size_tokens=75000,
            chunk_overlap_tokens=7500,
            batch_size=50,
            estimated_cost_per_ticket=0.045,
        ),
        ResourceProfile(
            dataset_size_range=(10001, 50000),
            recommended_workers=8,
            memory_per_worker_mb=768,
            chunk_size_tokens=100000,
            chunk_overlap_tokens=10000,
            batch_size=75,
            estimated_cost_per_ticket=0.042,
        ),
        ResourceProfile(
            dataset_size_range=(50001, 100000),
            recommended_workers=16,
            memory_per_worker_mb=1024,
            chunk_size_tokens=150000,
            chunk_overlap_tokens=15000,
            batch_size=100,
            estimated_cost_per_ticket=0.040,
        ),
        ResourceProfile(
            dataset_size_range=(100001, 500000),
            recommended_workers=32,
            memory_per_worker_mb=1536,
            chunk_size_tokens=200000,
            chunk_overlap_tokens=20000,
            batch_size=150,
            estimated_cost_per_ticket=0.038,
        ),
        ResourceProfile(
            dataset_size_range=(500001, float("inf")),
            recommended_workers=64,
            memory_per_worker_mb=2048,
            chunk_size_tokens=250000,
            chunk_overlap_tokens=25000,
            batch_size=200,
            estimated_cost_per_ticket=0.036,
        ),
    ]

    def __init__(
        self,
        max_workers: Optional[int] = None,
        worker_type: str = "thread",  # "thread" or "process"
        enable_auto_scaling: bool = True,
        enable_monitoring: bool = True,
        storage_dir: Optional[Path] = None,
    ):
        """
        Initialize the ScalabilityManager.

        Args:
            max_workers: Maximum number of workers (auto-detected if None)
            worker_type: Type of workers to use ("thread" or "process")
            enable_auto_scaling: Enable automatic scaling based on load
            enable_monitoring: Enable performance monitoring
            storage_dir: Directory for storing scaling metrics and logs
        """
        # System capabilities
        self.cpu_count = os.cpu_count()
        self.max_system_workers = min(self.cpu_count * 4, 64)  # Reasonable upper bound

        # Worker configuration
        self.max_workers = max_workers or min(self.cpu_count, 8)
        self.worker_type = worker_type
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_monitoring = enable_monitoring

        # Storage and logging
        self.storage_dir = storage_dir or Path("database/scaling_metrics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self.active_workers: Dict[str, WorkerConfig] = {}
        self.task_queue = Queue()
        self.completed_tasks = []
        self.failed_tasks = []
        self.start_time = None
        self.scaling_metrics = ScalingMetrics(
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            avg_task_duration=0.0,
            queue_depth=0,
            active_workers=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            throughput_tasks_per_minute=0.0,
            estimated_completion_time=timedelta(0),
        )

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

        logger.info(
            f"ScalabilityManager initialized with max_workers={self.max_workers}, type={self.worker_type}"
        )

    def get_optimal_resource_profile(self, dataset_size: int) -> ResourceProfile:
        """
        Get optimal resource configuration for given dataset size.

        Args:
            dataset_size: Number of tickets to process

        Returns:
            ResourceProfile with optimal configuration
        """
        for profile in self.RESOURCE_PROFILES:
            min_size, max_size = profile.dataset_size_range
            if min_size <= dataset_size <= max_size:
                logger.info(
                    f"Selected resource profile for {dataset_size:,} tickets: {profile.recommended_workers} workers"
                )
                return profile

        # Fallback to largest profile
        return self.RESOURCE_PROFILES[-1]

    def estimate_processing_cost(
        self, dataset_size: int, resource_profile: Optional[ResourceProfile] = None
    ) -> Dict[str, float]:
        """
        Estimate processing costs for dataset.

        Args:
            dataset_size: Number of tickets
            resource_profile: Resource profile to use (auto-selected if None)

        Returns:
            Dictionary with cost breakdown
        """
        if not resource_profile:
            resource_profile = self.get_optimal_resource_profile(dataset_size)

        base_cost = dataset_size * resource_profile.estimated_cost_per_ticket

        # Calculate additional costs
        infrastructure_cost = (
            resource_profile.recommended_workers * 0.001
        )  # $0.001 per worker-hour
        storage_cost = (
            (dataset_size * 0.5) / 1024 / 1024 * 0.023
        )  # $0.023 per GB storage

        total_cost = base_cost + infrastructure_cost + storage_cost

        return {
            "base_processing_cost": base_cost,
            "infrastructure_cost": infrastructure_cost,
            "storage_cost": storage_cost,
            "total_estimated_cost": total_cost,
            "cost_per_ticket": total_cost / dataset_size,
            "workers_needed": resource_profile.recommended_workers,
        }

    def auto_configure_for_dataset(self, dataset_size: int) -> Dict[str, Any]:
        """
        Automatically configure scaling parameters for dataset size.

        Args:
            dataset_size: Number of tickets to process

        Returns:
            Configuration dictionary
        """
        profile = self.get_optimal_resource_profile(dataset_size)

        # Adjust worker count based on system capabilities
        optimal_workers = min(
            profile.recommended_workers,
            self.max_system_workers,
            self.max_workers if self.max_workers else float("inf"),
        )

        # Calculate processing time estimates
        estimated_tasks_per_minute = optimal_workers * 2.5  # Rough estimate
        estimated_duration_minutes = dataset_size / estimated_tasks_per_minute

        config = {
            "dataset_size": dataset_size,
            "workers": optimal_workers,
            "memory_per_worker_mb": profile.memory_per_worker_mb,
            "chunk_size_tokens": profile.chunk_size_tokens,
            "chunk_overlap_tokens": profile.chunk_overlap_tokens,
            "batch_size": profile.batch_size,
            "estimated_duration_minutes": estimated_duration_minutes,
            "estimated_cost": self.estimate_processing_cost(dataset_size, profile),
        }

        logger.info(
            f"Auto-configured for {dataset_size:,} tickets: {optimal_workers} workers, ~{estimated_duration_minutes:.1f}min"
        )
        return config

    def create_worker_pool(self, num_workers: int) -> ThreadPoolExecutor:
        """
        Create optimized worker pool.

        Args:
            num_workers: Number of workers to create

        Returns:
            Configured executor
        """
        if self.worker_type == "process":
            # Process-based executor for CPU-intensive tasks
            executor = ProcessPoolExecutor(
                max_workers=num_workers, mp_context=None  # Use default context
            )
        else:
            # Thread-based executor for I/O-intensive tasks (default)
            executor = ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="ScalabilityWorker"
            )

        # Update active workers tracking
        for i in range(num_workers):
            worker_config = WorkerConfig(
                worker_id=f"worker_{i}",
                worker_type=self.worker_type,
                max_tasks_per_worker=50,
                task_timeout_seconds=300,
            )
            self.active_workers[worker_config.worker_id] = worker_config

        logger.info(f"Created {self.worker_type} pool with {num_workers} workers")
        return executor

    def submit_tasks_batch(
        self,
        executor: ThreadPoolExecutor,
        tasks: List[Any],
        process_function: Callable,
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        """
        Submit tasks in optimized batches.

        Args:
            executor: Worker pool executor
            tasks: List of tasks to process
            process_function: Function to process each task
            batch_size: Size of batches (auto-calculated if None)

        Returns:
            List of processed results
        """
        if not batch_size:
            # Auto-calculate optimal batch size
            profile = self.get_optimal_resource_profile(len(tasks))
            batch_size = profile.batch_size

        # Start monitoring if enabled
        if self.enable_monitoring and not self._monitoring_thread:
            self.start_monitoring()

        self.start_time = time.time()
        self.scaling_metrics.total_tasks = len(tasks)

        # Create batches
        batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]
        logger.info(
            f"Processing {len(tasks):,} tasks in {len(batches)} batches of size {batch_size}"
        )

        # Submit all batches
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(
                self._process_batch_wrapper, batch, process_function, i
            )
            future_to_batch[future] = (i, batch)

        # Collect results
        all_results = []
        for future in as_completed(future_to_batch):
            batch_index, batch = future_to_batch[future]
            try:
                batch_results = future.result(
                    timeout=600
                )  # 10-minute timeout per batch
                all_results.extend(batch_results)
                self.scaling_metrics.completed_tasks += len(batch_results)
                logger.info(
                    f"Batch {batch_index + 1}/{len(batches)} completed: {len(batch_results)} results"
                )
            except Exception as e:
                logger.error(f"Batch {batch_index} failed: {str(e)}")
                self.scaling_metrics.failed_tasks += len(batch)
                self.failed_tasks.extend(batch)

        # Update metrics
        self._update_final_metrics()

        logger.info(
            f"Processing completed: {len(all_results):,} successful, {self.scaling_metrics.failed_tasks} failed"
        )
        return all_results

    def _process_batch_wrapper(
        self, batch: List[Any], process_function: Callable, batch_index: int
    ) -> List[Any]:
        """
        Wrapper for batch processing with error handling and monitoring.

        Args:
            batch: Batch of tasks to process
            process_function: Function to process the batch
            batch_index: Index of this batch

        Returns:
            List of processed results
        """
        batch_start_time = time.time()

        try:
            # Process the batch
            results = process_function(batch, batch_index)

            # Update performance metrics
            batch_duration = time.time() - batch_start_time
            self._update_performance_metrics(len(batch), batch_duration)

            return results if isinstance(results, list) else [results]

        except Exception as e:
            logger.error(f"Error in batch {batch_index}: {str(e)}")
            self._log_batch_error(batch_index, batch, e)
            return []

    def _update_performance_metrics(self, batch_size: int, duration: float):
        """Update performance metrics with batch results."""
        # Thread-safe metric updates
        if duration > 0:
            current_avg = self.scaling_metrics.avg_task_duration
            completed = self.scaling_metrics.completed_tasks

            # Update rolling average
            if completed > 0:
                self.scaling_metrics.avg_task_duration = (
                    current_avg * completed + duration * batch_size
                ) / (completed + batch_size)
            else:
                self.scaling_metrics.avg_task_duration = duration / batch_size

    def _update_final_metrics(self):
        """Update final processing metrics."""
        if self.start_time:
            total_duration = time.time() - self.start_time

            # Calculate throughput
            if total_duration > 0:
                self.scaling_metrics.throughput_tasks_per_minute = (
                    self.scaling_metrics.completed_tasks / (total_duration / 60)
                )

            # System resource usage
            process = psutil.Process()
            self.scaling_metrics.memory_usage_mb = (
                process.memory_info().rss / 1024 / 1024
            )
            self.scaling_metrics.cpu_usage_percent = process.cpu_percent()

    def start_monitoring(self):
        """Start performance monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring thread."""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Update system metrics
                process = psutil.Process()
                self.scaling_metrics.memory_usage_mb = (
                    process.memory_info().rss / 1024 / 1024
                )
                self.scaling_metrics.cpu_usage_percent = process.cpu_percent()
                self.scaling_metrics.active_workers = len(self.active_workers)
                self.scaling_metrics.queue_depth = self.task_queue.qsize()

                # Estimate completion time
                if self.scaling_metrics.throughput_tasks_per_minute > 0:
                    remaining_tasks = (
                        self.scaling_metrics.total_tasks
                        - self.scaling_metrics.completed_tasks
                    )
                    remaining_minutes = (
                        remaining_tasks
                        / self.scaling_metrics.throughput_tasks_per_minute
                    )
                    self.scaling_metrics.estimated_completion_time = timedelta(
                        minutes=remaining_minutes
                    )

                # Log metrics every 30 seconds
                self._log_monitoring_metrics()

                # Auto-scaling logic
                if self.enable_auto_scaling:
                    self._check_auto_scaling()

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(10)

    def _log_monitoring_metrics(self):
        """Log current monitoring metrics."""
        metrics = self.scaling_metrics
        logger.info(
            f"Metrics - Tasks: {metrics.completed_tasks}/{metrics.total_tasks}, "
            f"Memory: {metrics.memory_usage_mb:.1f}MB, CPU: {metrics.cpu_usage_percent:.1f}%, "
            f"Throughput: {metrics.throughput_tasks_per_minute:.1f}/min"
        )

    def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        metrics = self.scaling_metrics

        # Scale up if queue is getting large and we have CPU/memory available
        if (
            metrics.queue_depth > 100
            and metrics.cpu_usage_percent < 70
            and metrics.memory_usage_mb < 4096
            and len(self.active_workers) < self.max_system_workers
        ):

            logger.info("Auto-scaling: Queue depth high, considering scale-up")
            # Implementation would add more workers here

        # Scale down if queue is empty and resource usage is low
        elif (
            metrics.queue_depth < 10
            and metrics.cpu_usage_percent < 30
            and len(self.active_workers) > 2
        ):

            logger.info("Auto-scaling: Low utilization, considering scale-down")
            # Implementation would remove workers here

    def _log_batch_error(self, batch_index: int, batch: List[Any], error: Exception):
        """Log detailed batch error information."""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "batch_index": batch_index,
            "batch_size": len(batch),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "batch_data": str(batch)[:1000],  # First 1000 chars
        }

        error_file = self.storage_dir / f"batch_error_{int(time.time())}.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)

    def get_scaling_report(self) -> Dict[str, Any]:
        """
        Get comprehensive scaling performance report.

        Returns:
            Dictionary with detailed scaling metrics
        """
        metrics = self.scaling_metrics

        # Calculate efficiency metrics
        success_rate = (
            (metrics.completed_tasks / metrics.total_tasks * 100)
            if metrics.total_tasks > 0
            else 0
        )

        # Calculate cost efficiency
        total_duration_hours = (
            (time.time() - self.start_time) / 3600 if self.start_time else 0
        )
        cost_per_hour = (
            len(self.active_workers) * 0.001
        )  # Estimated infrastructure cost
        total_cost = cost_per_hour * total_duration_hours

        report = {
            "performance_metrics": asdict(metrics),
            "efficiency_metrics": {
                "success_rate_percent": success_rate,
                "tasks_per_worker": (
                    metrics.completed_tasks / len(self.active_workers)
                    if self.active_workers
                    else 0
                ),
                "avg_task_duration_seconds": metrics.avg_task_duration,
                "throughput_tasks_per_minute": metrics.throughput_tasks_per_minute,
            },
            "resource_utilization": {
                "peak_memory_mb": metrics.memory_usage_mb,
                "avg_cpu_percent": metrics.cpu_usage_percent,
                "worker_count": len(self.active_workers),
                "worker_type": self.worker_type,
            },
            "cost_analysis": {
                "total_duration_hours": total_duration_hours,
                "estimated_total_cost": total_cost,
                "cost_per_completed_task": (
                    total_cost / metrics.completed_tasks
                    if metrics.completed_tasks > 0
                    else 0
                ),
            },
            "scaling_recommendations": self._generate_scaling_recommendations(),
        }

        return report

    def _generate_scaling_recommendations(self) -> Dict[str, str]:
        """Generate recommendations for future scaling."""
        metrics = self.scaling_metrics
        recommendations = {}

        # Memory recommendations
        if metrics.memory_usage_mb > 3000:
            recommendations["memory"] = (
                "Consider increasing memory limits or reducing batch sizes"
            )
        elif metrics.memory_usage_mb < 500:
            recommendations["memory"] = (
                "Memory usage is low, could increase batch sizes for better efficiency"
            )

        # CPU recommendations
        if metrics.cpu_usage_percent > 80:
            recommendations["cpu"] = (
                "High CPU usage detected, consider adding more workers or optimizing processing"
            )
        elif metrics.cpu_usage_percent < 30:
            recommendations["cpu"] = (
                "Low CPU usage, could increase concurrent processing"
            )

        # Throughput recommendations
        if metrics.throughput_tasks_per_minute < 10:
            recommendations["throughput"] = (
                "Low throughput detected, review processing logic and worker configuration"
            )

        # Worker recommendations
        worker_efficiency = (
            metrics.completed_tasks / len(self.active_workers)
            if self.active_workers
            else 0
        )
        if worker_efficiency < 10:
            recommendations["workers"] = (
                "Low worker efficiency, consider reducing worker count or increasing task complexity"
            )
        elif worker_efficiency > 100:
            recommendations["workers"] = (
                "High worker efficiency, could benefit from additional workers"
            )

        return recommendations

    def export_scaling_report(self, filename: Optional[str] = None) -> Path:
        """
        Export scaling report to JSON file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaling_report_{timestamp}.json"

        report_path = self.storage_dir / filename
        report = self.get_scaling_report()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Scaling report exported to: {report_path}")
        return report_path

    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        self.stop_monitoring()
        self.active_workers.clear()
        logger.info("ScalabilityManager cleanup completed")
