"""
Auto Scaler - Complete auto-scaling system with intelligent resource optimization.

This module provides comprehensive auto-scaling system with real-time performance
monitoring, queue depth analysis, predictive scaling, and resource optimization.
"""

import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import statistics
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class ScalingTrigger(Enum):
    """Triggers that can cause scaling events."""

    QUEUE_DEPTH = "queue_depth"
    CPU_USAGE = "cpu_usage"
    MEMORY_PRESSURE = "memory_pressure"
    THROUGHPUT_DROP = "throughput_drop"
    PREDICTED_LOAD = "predicted_load"
    MANUAL_TRIGGER = "manual_trigger"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingMetrics:
    """Real-time metrics for scaling decisions."""

    timestamp: datetime
    queue_depth: int
    active_workers: int
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    throughput_tasks_per_minute: float
    error_rate_percent: float
    avg_task_duration_seconds: float
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    timestamp: datetime
    action: ScalingAction
    trigger: ScalingTrigger
    workers_before: int
    workers_after: int
    trigger_value: float
    threshold_value: float
    success: bool
    duration_seconds: float
    impact_description: str


@dataclass
class ScalingThresholds:
    """Configurable thresholds for auto-scaling."""

    # Queue depth thresholds
    queue_scale_up_threshold: int = 100
    queue_scale_down_threshold: int = 10

    # Resource usage thresholds
    cpu_scale_up_threshold: float = 75.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0

    # Performance thresholds
    throughput_drop_threshold: float = 20.0  # Percent drop
    error_rate_threshold: float = 5.0  # Percent

    # Timing thresholds
    scale_up_delay_seconds: int = 60
    scale_down_delay_seconds: int = 300  # More conservative
    emergency_scale_delay_seconds: int = 10

    # Worker limits
    min_workers: int = 1
    max_workers: int = 64
    scale_factor: float = 1.5  # Multiplier for scaling


class PerformanceMonitor:
    """Monitors system and application performance for scaling decisions."""

    def __init__(self, history_size: int = 1000):
        self.metrics_history: deque = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 30  # seconds

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)

    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system and application metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Application metrics (would be provided by the actual application)
        # For now, using placeholder values
        queue_depth = 0  # Would be actual queue depth
        active_workers = 0  # Would be actual worker count
        throughput = 0.0  # Would be actual throughput
        error_rate = 0.0  # Would be actual error rate
        avg_duration = 0.0  # Would be actual average task duration
        pending_tasks = 0
        completed_tasks = 0
        failed_tasks = 0

        return ScalingMetrics(
            timestamp=datetime.now(),
            queue_depth=queue_depth,
            active_workers=active_workers,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage_percent,
            memory_available_gb=memory_available_gb,
            throughput_tasks_per_minute=throughput,
            error_rate_percent=error_rate,
            avg_task_duration_seconds=avg_duration,
            pending_tasks=pending_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
        )

    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_trend(self, minutes: int = 10) -> Dict[str, float]:
        """Get trend analysis for recent metrics."""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}

        # Get metrics from last N minutes
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if len(recent_metrics) < 2:
            recent_metrics = list(self.metrics_history)[
                -min(10, len(self.metrics_history)) :
            ]

        # Calculate trends
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_percent for m in recent_metrics]
        queue_values = [m.queue_depth for m in recent_metrics]
        throughput_values = [m.throughput_tasks_per_minute for m in recent_metrics]

        return {
            "avg_cpu_usage": statistics.mean(cpu_values),
            "cpu_trend": self._calculate_trend(cpu_values),
            "avg_memory_usage": statistics.mean(memory_values),
            "memory_trend": self._calculate_trend(memory_values),
            "avg_queue_depth": statistics.mean(queue_values),
            "queue_trend": self._calculate_trend(queue_values),
            "avg_throughput": statistics.mean(throughput_values),
            "throughput_trend": self._calculate_trend(throughput_values),
            "sample_count": len(recent_metrics),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple trend calculation
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        if not first_half or not second_half:
            return "stable"

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        change_percent = ((second_avg - first_avg) / max(first_avg, 0.01)) * 100

        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"


class ScalingDecisionEngine:
    """Engine that makes intelligent scaling decisions based on metrics and thresholds."""

    def __init__(self, thresholds: Optional[ScalingThresholds] = None):
        self.thresholds = thresholds or ScalingThresholds()
        self.last_scale_action_time = {}  # Track last action time per trigger type
        self.scaling_history: List[ScalingEvent] = []

    def analyze_scaling_need(
        self,
        current_metrics: ScalingMetrics,
        trends: Dict[str, float],
        current_workers: int,
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """
        Analyze current state and determine if scaling is needed.

        Args:
            current_metrics: Current system metrics
            trends: Trend analysis
            current_workers: Current number of workers

        Returns:
            Tuple of (action, trigger, trigger_value)
        """
        # Emergency scaling checks (immediate action needed)
        emergency_action = self._check_emergency_scaling(
            current_metrics, current_workers
        )
        if emergency_action[0] != ScalingAction.MAINTAIN:
            return emergency_action

        # Standard scaling checks
        scale_up_checks = [
            self._check_queue_scaling_up(current_metrics, trends),
            self._check_cpu_scaling_up(current_metrics, trends),
            self._check_memory_scaling_up(current_metrics),
            self._check_throughput_scaling_up(current_metrics, trends),
        ]

        scale_down_checks = [
            self._check_queue_scaling_down(current_metrics, trends),
            self._check_cpu_scaling_down(current_metrics, trends),
            self._check_memory_scaling_down(current_metrics),
            self._check_throughput_scaling_down(current_metrics, trends),
        ]

        # Find highest priority scale-up trigger
        scale_up_triggers = [
            check for check in scale_up_checks if check[0] == ScalingAction.SCALE_UP
        ]
        if scale_up_triggers and self._can_scale_up(current_workers):
            # Choose trigger with highest urgency (largest trigger value)
            return max(scale_up_triggers, key=lambda x: x[2])

        # Find scale-down triggers if no scale-up needed
        scale_down_triggers = [
            check for check in scale_down_checks if check[0] == ScalingAction.SCALE_DOWN
        ]
        if scale_down_triggers and self._can_scale_down(current_workers):
            # Choose trigger with highest confidence (largest trigger value for scale down)
            return max(scale_down_triggers, key=lambda x: x[2])

        return ScalingAction.MAINTAIN, ScalingTrigger.MANUAL_TRIGGER, 0.0

    def _check_emergency_scaling(
        self, metrics: ScalingMetrics, current_workers: int
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check for emergency scaling conditions."""

        # Emergency scale up: very high error rate
        if metrics.error_rate_percent > 20:
            return (
                ScalingAction.EMERGENCY_SCALE,
                ScalingTrigger.ERROR_RATE,
                metrics.error_rate_percent,
            )

        # Emergency scale up: memory critically low
        if metrics.memory_available_gb < 0.5:
            return (
                ScalingAction.EMERGENCY_SCALE,
                ScalingTrigger.MEMORY_PRESSURE,
                metrics.memory_usage_percent,
            )

        # Emergency scale up: queue extremely deep
        if metrics.queue_depth > 1000:
            return (
                ScalingAction.EMERGENCY_SCALE,
                ScalingTrigger.QUEUE_DEPTH,
                metrics.queue_depth,
            )

        return ScalingAction.MAINTAIN, ScalingTrigger.MANUAL_TRIGGER, 0.0

    def _check_queue_scaling_up(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if queue depth requires scaling up."""
        if (
            metrics.queue_depth > self.thresholds.queue_scale_up_threshold
            and trends.get("queue_trend") in ["increasing", "stable"]
            and self._check_cooldown(
                ScalingTrigger.QUEUE_DEPTH, self.thresholds.scale_up_delay_seconds
            )
        ):
            return (
                ScalingAction.SCALE_UP,
                ScalingTrigger.QUEUE_DEPTH,
                metrics.queue_depth,
            )

        return ScalingAction.MAINTAIN, ScalingTrigger.QUEUE_DEPTH, metrics.queue_depth

    def _check_queue_scaling_down(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if queue depth allows scaling down."""
        if (
            metrics.queue_depth < self.thresholds.queue_scale_down_threshold
            and trends.get("queue_trend") in ["decreasing", "stable"]
            and self._check_cooldown(
                ScalingTrigger.QUEUE_DEPTH, self.thresholds.scale_down_delay_seconds
            )
        ):
            return (
                ScalingAction.SCALE_DOWN,
                ScalingTrigger.QUEUE_DEPTH,
                metrics.queue_depth,
            )

        return ScalingAction.MAINTAIN, ScalingTrigger.QUEUE_DEPTH, metrics.queue_depth

    def _check_cpu_scaling_up(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if CPU usage requires scaling up."""
        if (
            metrics.cpu_usage_percent > self.thresholds.cpu_scale_up_threshold
            and trends.get("cpu_trend") in ["increasing", "stable"]
        ):

            if self._check_cooldown(
                ScalingTrigger.CPU_USAGE, self.thresholds.scale_up_delay_seconds
            ):
                return (
                    ScalingAction.SCALE_UP,
                    ScalingTrigger.CPU_USAGE,
                    metrics.cpu_usage_percent,
                )

        return (
            ScalingAction.MAINTAIN,
            ScalingTrigger.CPU_USAGE,
            metrics.cpu_usage_percent,
        )

    def _check_cpu_scaling_down(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if CPU usage allows scaling down."""
        if (
            metrics.cpu_usage_percent < self.thresholds.cpu_scale_down_threshold
            and trends.get("cpu_trend") in ["decreasing", "stable"]
        ):

            if self._check_cooldown(
                ScalingTrigger.CPU_USAGE, self.thresholds.scale_down_delay_seconds
            ):
                return (
                    ScalingAction.SCALE_DOWN,
                    ScalingTrigger.CPU_USAGE,
                    metrics.cpu_usage_percent,
                )

        return (
            ScalingAction.MAINTAIN,
            ScalingTrigger.CPU_USAGE,
            metrics.cpu_usage_percent,
        )

    def _check_memory_scaling_up(
        self, metrics: ScalingMetrics
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if memory usage requires scaling up."""
        if metrics.memory_usage_percent > self.thresholds.memory_scale_up_threshold:
            if self._check_cooldown(
                ScalingTrigger.MEMORY_PRESSURE, self.thresholds.scale_up_delay_seconds
            ):
                return (
                    ScalingAction.SCALE_UP,
                    ScalingTrigger.MEMORY_PRESSURE,
                    metrics.memory_usage_percent,
                )

        return (
            ScalingAction.MAINTAIN,
            ScalingTrigger.MEMORY_PRESSURE,
            metrics.memory_usage_percent,
        )

    def _check_memory_scaling_down(
        self, metrics: ScalingMetrics
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if memory usage allows scaling down."""
        if metrics.memory_usage_percent < self.thresholds.memory_scale_down_threshold:
            if self._check_cooldown(
                ScalingTrigger.MEMORY_PRESSURE, self.thresholds.scale_down_delay_seconds
            ):
                return (
                    ScalingAction.SCALE_DOWN,
                    ScalingTrigger.MEMORY_PRESSURE,
                    metrics.memory_usage_percent,
                )

        return (
            ScalingAction.MAINTAIN,
            ScalingTrigger.MEMORY_PRESSURE,
            metrics.memory_usage_percent,
        )

    def _check_throughput_scaling_up(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if throughput drop requires scaling up."""
        if trends.get("throughput_trend") == "decreasing":
            throughput_drop = abs(
                trends.get("avg_throughput", 0) - metrics.throughput_tasks_per_minute
            )
            drop_percent = (
                throughput_drop / max(trends.get("avg_throughput", 1), 1)
            ) * 100

            if drop_percent > self.thresholds.throughput_drop_threshold:
                if self._check_cooldown(
                    ScalingTrigger.THROUGHPUT_DROP,
                    self.thresholds.scale_up_delay_seconds,
                ):
                    return (
                        ScalingAction.SCALE_UP,
                        ScalingTrigger.THROUGHPUT_DROP,
                        drop_percent,
                    )

        return ScalingAction.MAINTAIN, ScalingTrigger.THROUGHPUT_DROP, 0.0

    def _check_throughput_scaling_down(
        self, metrics: ScalingMetrics, trends: Dict[str, float]
    ) -> Tuple[ScalingAction, ScalingTrigger, float]:
        """Check if high throughput allows scaling down."""
        # Only scale down if throughput is consistently high and stable
        if (
            trends.get("throughput_trend") == "stable"
            and trends.get("avg_throughput", 0)
            > metrics.throughput_tasks_per_minute * 1.5
        ):

            if self._check_cooldown(
                ScalingTrigger.THROUGHPUT_DROP, self.thresholds.scale_down_delay_seconds
            ):
                return (
                    ScalingAction.SCALE_DOWN,
                    ScalingTrigger.THROUGHPUT_DROP,
                    trends.get("avg_throughput", 0),
                )

        return ScalingAction.MAINTAIN, ScalingTrigger.THROUGHPUT_DROP, 0.0

    def _check_cooldown(self, trigger: ScalingTrigger, cooldown_seconds: int) -> bool:
        """Check if enough time has passed since last scaling action for this trigger."""
        last_action_time = self.last_scale_action_time.get(trigger)
        if last_action_time is None:
            return True

        time_since_last = (datetime.now() - last_action_time).total_seconds()
        return time_since_last >= cooldown_seconds

    def _can_scale_up(self, current_workers: int) -> bool:
        """Check if scaling up is allowed."""
        return current_workers < self.thresholds.max_workers

    def _can_scale_down(self, current_workers: int) -> bool:
        """Check if scaling down is allowed."""
        return current_workers > self.thresholds.min_workers

    def record_scaling_event(self, event: ScalingEvent):
        """Record a scaling event for history tracking."""
        self.scaling_history.append(event)
        self.last_scale_action_time[event.trigger] = event.timestamp

        # Keep only recent history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]  # Keep last 500 events


class WorkerPoolManager:
    """Manages the actual worker pool scaling operations."""

    def __init__(self, initial_workers: int = 4, worker_type: str = "thread"):
        self.current_workers = initial_workers
        self.worker_type = worker_type
        self.executor = None
        self.scaling_lock = threading.Lock()

        self._initialize_worker_pool()

    def _initialize_worker_pool(self):
        """Initialize the worker pool."""
        if self.worker_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)

        logger.info(
            f"Initialized {self.worker_type} pool with {self.current_workers} workers"
        )

    def scale_workers(self, target_workers: int) -> bool:
        """
        Scale the worker pool to the target number of workers.

        Args:
            target_workers: Target number of workers

        Returns:
            True if scaling was successful
        """
        with self.scaling_lock:
            if target_workers == self.current_workers:
                return True

            try:
                # Shutdown current executor
                if self.executor:
                    self.executor.shutdown(wait=False)

                # Create new executor with target workers
                self.current_workers = target_workers
                if self.worker_type == "process":
                    self.executor = ProcessPoolExecutor(
                        max_workers=self.current_workers
                    )
                else:
                    self.executor = ThreadPoolExecutor(max_workers=self.current_workers)

                logger.info(f"Scaled worker pool to {self.current_workers} workers")
                return True

            except Exception as e:
                logger.error(f"Error scaling worker pool: {str(e)}")
                return False

    def get_current_worker_count(self) -> int:
        """Get current number of workers."""
        return self.current_workers

    def shutdown(self):
        """Shutdown the worker pool."""
        if self.executor:
            self.executor.shutdown(wait=True)


class AutoScaler:
    """
    Main auto-scaling orchestrator that coordinates monitoring, decision-making, and scaling.
    """

    def __init__(
        self,
        initial_workers: int = 4,
        worker_type: str = "thread",
        thresholds: Optional[ScalingThresholds] = None,
        storage_dir: Optional[Path] = None,
    ):
        self.storage_dir = storage_dir or Path("database/auto_scaling")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = PerformanceMonitor()
        self.decision_engine = ScalingDecisionEngine(thresholds)
        self.worker_manager = WorkerPoolManager(initial_workers, worker_type)

        self.auto_scaling_active = False
        self.scaling_thread = None
        self.scaling_interval = 60  # Check every minute

        # Callbacks for application integration
        self.metrics_provider: Optional[Callable] = None
        self.scaling_callback: Optional[Callable] = None

        logger.info(f"AutoScaler initialized with {initial_workers} workers")

    def set_metrics_provider(self, provider: Callable[[], ScalingMetrics]):
        """Set callback function to provide application-specific metrics."""
        self.metrics_provider = provider

    def set_scaling_callback(self, callback: Callable[[int, int], bool]):
        """Set callback function to notify application of scaling events."""
        self.scaling_callback = callback

    def start_auto_scaling(self):
        """Start the auto-scaling system."""
        if self.auto_scaling_active:
            return

        self.auto_scaling_active = True
        self.monitor.start_monitoring()

        self.scaling_thread = threading.Thread(
            target=self._auto_scaling_loop, daemon=True
        )
        self.scaling_thread.start()

        logger.info("Auto-scaling system started")

    def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self.auto_scaling_active = False
        self.monitor.stop_monitoring()

        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)

        logger.info("Auto-scaling system stopped")

    def _auto_scaling_loop(self):
        """Main auto-scaling loop."""
        while self.auto_scaling_active:
            try:
                self._perform_scaling_check()
                time.sleep(self.scaling_interval)
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {str(e)}")
                time.sleep(30)  # Wait longer on error

    def _perform_scaling_check(self):
        """Perform a single scaling check and action if needed."""
        # Get current metrics
        current_metrics = self._get_current_metrics()
        if not current_metrics:
            return

        # Get trend analysis
        trends = self.monitor.get_metrics_trend()

        # Get current worker count
        current_workers = self.worker_manager.get_current_worker_count()

        # Analyze scaling need
        action, trigger, trigger_value = self.decision_engine.analyze_scaling_need(
            current_metrics, trends, current_workers
        )

        # Execute scaling action if needed
        if action != ScalingAction.MAINTAIN:
            self._execute_scaling_action(
                action, trigger, trigger_value, current_workers
            )

    def _get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get current metrics from monitor or application callback."""
        if self.metrics_provider:
            try:
                return self.metrics_provider()
            except Exception as e:
                logger.error(f"Error getting metrics from provider: {str(e)}")

        return self.monitor.get_current_metrics()

    def _execute_scaling_action(
        self,
        action: ScalingAction,
        trigger: ScalingTrigger,
        trigger_value: float,
        current_workers: int,
    ):
        """Execute the determined scaling action."""
        start_time = datetime.now()

        # Calculate target workers
        if action in [ScalingAction.SCALE_UP, ScalingAction.EMERGENCY_SCALE]:
            if action == ScalingAction.EMERGENCY_SCALE:
                target_workers = min(
                    current_workers * 2, self.decision_engine.thresholds.max_workers
                )
            else:
                target_workers = min(
                    int(current_workers * self.decision_engine.thresholds.scale_factor),
                    self.decision_engine.thresholds.max_workers,
                )
        else:  # SCALE_DOWN
            target_workers = max(
                int(current_workers / self.decision_engine.thresholds.scale_factor),
                self.decision_engine.thresholds.min_workers,
            )

        # Perform scaling
        success = self.worker_manager.scale_workers(target_workers)

        # Notify application if callback provided
        if self.scaling_callback and success:
            try:
                self.scaling_callback(current_workers, target_workers)
            except Exception as e:
                logger.error(f"Error in scaling callback: {str(e)}")

        # Record scaling event
        duration = (datetime.now() - start_time).total_seconds()

        event = ScalingEvent(
            timestamp=start_time,
            action=action,
            trigger=trigger,
            workers_before=current_workers,
            workers_after=target_workers if success else current_workers,
            trigger_value=trigger_value,
            threshold_value=self._get_threshold_for_trigger(trigger, action),
            success=success,
            duration_seconds=duration,
            impact_description=f"Scaled from {current_workers} to {target_workers} workers due to {trigger.value}",
        )

        self.decision_engine.record_scaling_event(event)

        logger.info(
            f"Scaling event: {action.value} triggered by {trigger.value} "
            f"({trigger_value:.2f}) - {current_workers} -> {target_workers} workers"
        )

    def _get_threshold_for_trigger(
        self, trigger: ScalingTrigger, action: ScalingAction
    ) -> float:
        """Get the threshold value that triggered the action."""
        thresholds = self.decision_engine.thresholds

        threshold_map = {
            (
                ScalingTrigger.QUEUE_DEPTH,
                ScalingAction.SCALE_UP,
            ): thresholds.queue_scale_up_threshold,
            (
                ScalingTrigger.QUEUE_DEPTH,
                ScalingAction.SCALE_DOWN,
            ): thresholds.queue_scale_down_threshold,
            (
                ScalingTrigger.CPU_USAGE,
                ScalingAction.SCALE_UP,
            ): thresholds.cpu_scale_up_threshold,
            (
                ScalingTrigger.CPU_USAGE,
                ScalingAction.SCALE_DOWN,
            ): thresholds.cpu_scale_down_threshold,
            (
                ScalingTrigger.MEMORY_PRESSURE,
                ScalingAction.SCALE_UP,
            ): thresholds.memory_scale_up_threshold,
            (
                ScalingTrigger.MEMORY_PRESSURE,
                ScalingAction.SCALE_DOWN,
            ): thresholds.memory_scale_down_threshold,
            (
                ScalingTrigger.THROUGHPUT_DROP,
                ScalingAction.SCALE_UP,
            ): thresholds.throughput_drop_threshold,
        }

        return threshold_map.get((trigger, action), 0.0)

    def manual_scale(self, target_workers: int, reason: str = "Manual scaling"):
        """Manually trigger scaling to specific worker count."""
        current_workers = self.worker_manager.get_current_worker_count()

        if target_workers == current_workers:
            logger.info(f"Already at target worker count: {target_workers}")
            return True

        action = (
            ScalingAction.SCALE_UP
            if target_workers > current_workers
            else ScalingAction.SCALE_DOWN
        )

        success = self.worker_manager.scale_workers(target_workers)

        if success:
            event = ScalingEvent(
                timestamp=datetime.now(),
                action=action,
                trigger=ScalingTrigger.MANUAL_TRIGGER,
                workers_before=current_workers,
                workers_after=target_workers,
                trigger_value=target_workers,
                threshold_value=0.0,
                success=True,
                duration_seconds=1.0,
                impact_description=f"Manual scaling: {reason}",
            )

            self.decision_engine.record_scaling_event(event)
            logger.info(
                f"Manual scaling successful: {current_workers} -> {target_workers} workers"
            )

        return success

    def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive scaling dashboard data."""
        current_metrics = self._get_current_metrics()
        trends = self.monitor.get_metrics_trend()
        current_workers = self.worker_manager.get_current_worker_count()

        # Recent scaling events
        recent_events = [
            asdict(event) for event in self.decision_engine.scaling_history[-10:]
        ]

        # Scaling statistics
        total_events = len(self.decision_engine.scaling_history)
        successful_events = sum(
            1 for e in self.decision_engine.scaling_history if e.success
        )
        success_rate = (
            (successful_events / total_events * 100) if total_events > 0 else 0
        )

        # Trigger frequency analysis
        trigger_counts = {}
        for event in self.decision_engine.scaling_history:
            trigger_counts[event.trigger.value] = (
                trigger_counts.get(event.trigger.value, 0) + 1
            )

        return {
            "current_status": {
                "auto_scaling_active": self.auto_scaling_active,
                "current_workers": current_workers,
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "trends": trends,
            },
            "thresholds": asdict(self.decision_engine.thresholds),
            "scaling_history": {
                "recent_events": recent_events,
                "total_events": total_events,
                "success_rate_percent": success_rate,
                "trigger_frequency": trigger_counts,
            },
            "worker_pool": {
                "type": self.worker_manager.worker_type,
                "current_count": current_workers,
                "min_workers": self.decision_engine.thresholds.min_workers,
                "max_workers": self.decision_engine.thresholds.max_workers,
            },
        }

    def export_scaling_report(self, filename: Optional[str] = None) -> Path:
        """Export comprehensive scaling report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auto_scaling_report_{timestamp}.json"

        report_path = self.storage_dir / filename
        dashboard_data = self.get_scaling_dashboard()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Auto-scaling report exported: {report_path}")
        return report_path

    def cleanup(self):
        """Cleanup all auto-scaling resources."""
        self.stop_auto_scaling()
        self.worker_manager.shutdown()
        logger.info("AutoScaler cleanup completed")
