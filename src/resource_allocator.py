"""
Resource Allocator - Dynamic resource allocation system.

This module provides intelligent resource allocation algorithms that automatically
adjust processing resources based on dataset size, complexity, and system capabilities.
"""

import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import statistics
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Current system resource availability."""

    cpu_cores: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    disk_space_available_gb: float
    network_bandwidth_mbps: Optional[float] = None


@dataclass
class DatasetCharacteristics:
    """Characteristics of the dataset being processed."""

    total_tickets: int
    avg_text_length: float
    max_text_length: int
    min_text_length: int
    text_complexity_score: float  # 0-1, higher = more complex
    language_diversity: float  # 0-1, higher = more languages
    estimated_processing_time_per_ticket: float  # seconds
    total_size_mb: float


@dataclass
class AllocationRecommendation:
    """Resource allocation recommendation."""

    recommended_workers: int
    memory_per_worker_mb: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    batch_size: int
    processing_mode: str  # 'memory_optimized', 'speed_optimized', 'balanced'
    estimated_duration_minutes: float
    estimated_peak_memory_gb: float
    estimated_cost: float
    confidence_score: float  # 0-1, how confident we are in this recommendation
    reasoning: List[str]  # Explanation for the recommendation


@dataclass
class HistoricalPerformance:
    """Historical performance data for learning."""

    dataset_size: int
    workers_used: int
    actual_duration_minutes: float
    peak_memory_gb: float
    success_rate: float
    throughput_tickets_per_minute: float
    cost_per_ticket: float
    timestamp: datetime


class DatasetAnalyzer:
    """Analyzes dataset characteristics for resource planning."""

    def __init__(self):
        self.analysis_cache = {}

    def analyze_dataset(
        self, file_path: Path, sample_size: int = 1000
    ) -> DatasetCharacteristics:
        """
        Analyze dataset characteristics for resource planning.

        Args:
            file_path: Path to dataset file
            sample_size: Number of rows to sample for analysis

        Returns:
            DatasetCharacteristics with analysis results
        """
        logger.info(f"Analyzing dataset characteristics: {file_path}")

        try:
            import pandas as pd

            # Read sample of data
            if file_path.suffix.lower() == ".csv":
                df_sample = pd.read_csv(file_path, nrows=sample_size, dtype=str)
            else:
                df_sample = pd.read_excel(file_path, nrows=sample_size, dtype=str)

            # Calculate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            # Estimate total rows
            if len(df_sample) == sample_size:
                # Estimate total rows based on file size
                estimated_total_rows = int(
                    (
                        file_size_mb
                        / (df_sample.memory_usage(deep=True).sum() / 1024 / 1024)
                    )
                    * len(df_sample)
                )
            else:
                estimated_total_rows = len(df_sample)

            # Analyze text characteristics
            text_lengths = df_sample["text"].fillna("").str.len()
            avg_length = text_lengths.mean()
            max_length = text_lengths.max()
            min_length = text_lengths.min()

            # Calculate text complexity (simplified heuristic)
            complexity_factors = []

            # Length complexity
            complexity_factors.append(
                min(avg_length / 1000, 1.0)
            )  # 0-1 based on avg length

            # Character diversity (more diverse = more complex)
            sample_texts = df_sample["text"].fillna("").head(100)
            unique_chars = set("".join(sample_texts))
            char_diversity = min(
                len(unique_chars) / 200, 1.0
            )  # 0-1 based on character variety
            complexity_factors.append(char_diversity)

            # Sentence structure complexity (rough estimate)
            avg_sentences = sample_texts.str.count(r"[.!?]").mean()
            sentence_complexity = min(
                avg_sentences / 10, 1.0
            )  # 0-1 based on sentences per text
            complexity_factors.append(sentence_complexity)

            text_complexity = statistics.mean(complexity_factors)

            # Estimate language diversity (simplified)
            # Count special characters that might indicate different languages
            special_char_ratio = (
                sample_texts.str.count(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]").mean()
                / avg_length
                if avg_length > 0
                else 0
            )
            language_diversity = min(special_char_ratio * 10, 1.0)

            # Estimate processing time per ticket (based on complexity and length)
            base_time = 0.1  # 0.1 seconds base time
            length_factor = avg_length / 1000  # Longer texts take more time
            complexity_factor = text_complexity * 2  # Complex texts take longer
            estimated_time_per_ticket = base_time + length_factor + complexity_factor

            characteristics = DatasetCharacteristics(
                total_tickets=estimated_total_rows,
                avg_text_length=avg_length,
                max_text_length=max_length,
                min_text_length=min_length,
                text_complexity_score=text_complexity,
                language_diversity=language_diversity,
                estimated_processing_time_per_ticket=estimated_time_per_ticket,
                total_size_mb=file_size_mb,
            )

            logger.info(
                f"Dataset analysis complete: {estimated_total_rows:,} tickets, avg length {avg_length:.0f} chars, complexity {text_complexity:.2f}"
            )
            return characteristics

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            # Return default characteristics
            return DatasetCharacteristics(
                total_tickets=1000,
                avg_text_length=500,
                max_text_length=2000,
                min_text_length=50,
                text_complexity_score=0.5,
                language_diversity=0.3,
                estimated_processing_time_per_ticket=0.5,
                total_size_mb=10.0,
            )


class SystemProfiler:
    """Profiles system resources and capabilities."""

    def __init__(self):
        self.historical_measurements = deque(maxlen=100)

    def get_current_resources(self) -> SystemResources:
        """Get current system resource availability."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)

            # Memory information
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent

            # Disk space
            disk = psutil.disk_usage("/")
            disk_available_gb = disk.free / (1024**3)

            resources = SystemResources(
                cpu_cores=cpu_count,
                cpu_usage_percent=cpu_usage,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                memory_usage_percent=memory_usage_percent,
                disk_space_available_gb=disk_available_gb,
            )

            # Store for trend analysis
            self.historical_measurements.append(resources)

            return resources

        except Exception as e:
            logger.error(f"Error profiling system resources: {str(e)}")
            # Return conservative defaults
            return SystemResources(
                cpu_cores=4,
                cpu_usage_percent=50.0,
                memory_total_gb=8.0,
                memory_available_gb=4.0,
                memory_usage_percent=50.0,
                disk_space_available_gb=10.0,
            )

    def get_resource_trends(self) -> Dict[str, float]:
        """Analyze resource usage trends."""
        if len(self.historical_measurements) < 3:
            return {"trend": "insufficient_data"}

        recent = list(self.historical_measurements)[-10:]  # Last 10 measurements

        # Calculate trends
        cpu_trend = statistics.mean([r.cpu_usage_percent for r in recent])
        memory_trend = statistics.mean([r.memory_usage_percent for r in recent])

        return {
            "avg_cpu_usage": cpu_trend,
            "avg_memory_usage": memory_trend,
            "measurements_count": len(recent),
            "trend_direction": "stable",  # Could be enhanced with actual trend calculation
        }


class PerformanceLearner:
    """Learns from historical performance data to improve recommendations."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.performance_history: List[HistoricalPerformance] = []
        self.load_historical_data()

    def record_performance(
        self,
        dataset_size: int,
        workers_used: int,
        actual_duration_minutes: float,
        peak_memory_gb: float,
        success_rate: float,
        throughput_tickets_per_minute: float,
        cost_per_ticket: float,
    ):
        """Record actual performance data for learning."""
        performance = HistoricalPerformance(
            dataset_size=dataset_size,
            workers_used=workers_used,
            actual_duration_minutes=actual_duration_minutes,
            peak_memory_gb=peak_memory_gb,
            success_rate=success_rate,
            throughput_tickets_per_minute=throughput_tickets_per_minute,
            cost_per_ticket=cost_per_ticket,
            timestamp=datetime.now(),
        )

        self.performance_history.append(performance)
        self.save_historical_data()

        logger.info(
            f"Recorded performance: {dataset_size:,} tickets in {actual_duration_minutes:.1f}min with {workers_used} workers"
        )

    def get_similar_workloads(
        self, dataset_size: int, tolerance: float = 0.3
    ) -> List[HistoricalPerformance]:
        """Find historical workloads similar to current dataset size."""
        similar = []

        for perf in self.performance_history:
            size_ratio = abs(perf.dataset_size - dataset_size) / max(
                dataset_size, perf.dataset_size
            )
            if size_ratio <= tolerance:
                similar.append(perf)

        # Sort by similarity (closest size first)
        similar.sort(key=lambda p: abs(p.dataset_size - dataset_size))
        return similar

    def predict_performance(self, dataset_size: int, workers: int) -> Dict[str, float]:
        """Predict performance based on historical data."""
        similar_workloads = self.get_similar_workloads(dataset_size)

        if not similar_workloads:
            # No historical data, use defaults
            return {
                "predicted_duration_minutes": dataset_size
                * 0.01,  # 0.01 min per ticket
                "predicted_throughput": 100.0,  # 100 tickets per minute
                "confidence": 0.1,
            }

        # Use weighted average based on similarity
        total_weight = 0
        weighted_duration = 0
        weighted_throughput = 0

        for perf in similar_workloads[:5]:  # Use top 5 similar workloads
            # Weight by similarity and recency
            size_similarity = 1 - (
                abs(perf.dataset_size - dataset_size)
                / max(dataset_size, perf.dataset_size)
            )
            worker_similarity = 1 - (
                abs(perf.workers_used - workers) / max(workers, perf.workers_used)
            )
            recency_weight = min(
                1.0, 30 / max(1, (datetime.now() - perf.timestamp).days)
            )  # Prefer recent data

            weight = size_similarity * worker_similarity * recency_weight
            total_weight += weight

            # Scale duration based on worker difference
            worker_factor = perf.workers_used / workers if workers > 0 else 1
            adjusted_duration = perf.actual_duration_minutes * worker_factor

            weighted_duration += adjusted_duration * weight
            weighted_throughput += perf.throughput_tickets_per_minute * weight

        if total_weight > 0:
            predicted_duration = weighted_duration / total_weight
            predicted_throughput = weighted_throughput / total_weight
            confidence = min(1.0, total_weight / len(similar_workloads))
        else:
            predicted_duration = dataset_size * 0.01
            predicted_throughput = 100.0
            confidence = 0.2

        return {
            "predicted_duration_minutes": predicted_duration,
            "predicted_throughput": predicted_throughput,
            "confidence": confidence,
            "similar_workloads_count": len(similar_workloads),
        }

    def save_historical_data(self):
        """Save historical performance data to disk."""
        try:
            history_file = self.storage_dir / "performance_history.json"

            # Convert to serializable format
            data = []
            for perf in self.performance_history:
                data.append({**asdict(perf), "timestamp": perf.timestamp.isoformat()})

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")

    def load_historical_data(self):
        """Load historical performance data from disk."""
        try:
            history_file = self.storage_dir / "performance_history.json"

            if not history_file.exists():
                return

            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.performance_history = []
            for item in data:
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                self.performance_history.append(HistoricalPerformance(**item))

            logger.info(
                f"Loaded {len(self.performance_history)} historical performance records"
            )

        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")


class ResourceAllocator:
    """
    Main resource allocator that provides intelligent resource allocation recommendations.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("database/resource_allocation")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_analyzer = DatasetAnalyzer()
        self.system_profiler = SystemProfiler()
        self.performance_learner = PerformanceLearner(self.storage_dir)

        logger.info("ResourceAllocator initialized")

    def analyze_and_recommend(
        self,
        file_path: Path,
        target_mode: str = "balanced",  # "speed_optimized", "memory_optimized", "cost_optimized"
        max_workers_override: Optional[int] = None,
        max_memory_gb_override: Optional[float] = None,
    ) -> AllocationRecommendation:
        """
        Analyze dataset and system resources to provide allocation recommendation.

        Args:
            file_path: Path to dataset file
            target_mode: Optimization target
            max_workers_override: Override maximum workers
            max_memory_gb_override: Override memory limit

        Returns:
            AllocationRecommendation with optimal configuration
        """
        logger.info(f"Analyzing resource allocation for {file_path}")

        # Analyze dataset characteristics
        dataset_chars = self.dataset_analyzer.analyze_dataset(file_path)

        # Get current system resources
        system_resources = self.system_profiler.get_current_resources()

        # Get historical performance insights
        performance_insights = self.performance_learner.predict_performance(
            dataset_chars.total_tickets, min(system_resources.cpu_cores, 8)
        )

        # Generate recommendation based on analysis
        recommendation = self._generate_recommendation(
            dataset_chars,
            system_resources,
            performance_insights,
            target_mode,
            max_workers_override,
            max_memory_gb_override,
        )

        logger.info(
            f"Resource allocation recommendation: {recommendation.recommended_workers} workers, {recommendation.processing_mode} mode"
        )
        return recommendation

    def _generate_recommendation(
        self,
        dataset: DatasetCharacteristics,
        system: SystemResources,
        performance: Dict[str, float],
        target_mode: str,
        max_workers_override: Optional[int],
        max_memory_override: Optional[float],
    ) -> AllocationRecommendation:
        """Generate optimal resource allocation recommendation."""

        reasoning = []

        # Determine base worker count
        if dataset.total_tickets < 1000:
            base_workers = 2
            reasoning.append("Small dataset: using minimal workers")
        elif dataset.total_tickets < 10000:
            base_workers = min(4, system.cpu_cores)
            reasoning.append("Medium dataset: using moderate workers")
        elif dataset.total_tickets < 100000:
            base_workers = min(8, system.cpu_cores)
            reasoning.append("Large dataset: using more workers")
        else:
            base_workers = min(16, system.cpu_cores * 2)
            reasoning.append("Very large dataset: using maximum efficient workers")

        # Adjust based on system constraints
        available_memory_gb = system.memory_available_gb
        if available_memory_gb < 2:
            base_workers = min(base_workers, 2)
            reasoning.append("Limited memory: reducing workers")
        elif available_memory_gb > 8:
            base_workers = min(base_workers * 2, system.cpu_cores * 2)
            reasoning.append("Ample memory: allowing more workers")

        # Apply overrides
        if max_workers_override:
            base_workers = min(base_workers, max_workers_override)
            reasoning.append(f"User override: max workers {max_workers_override}")

        # Adjust for target mode
        if target_mode == "speed_optimized":
            recommended_workers = min(base_workers * 2, system.cpu_cores * 2)
            memory_per_worker = min(
                1024, int(available_memory_gb * 1024 / recommended_workers)
            )
            chunk_size = 150000
            batch_size = 100
            reasoning.append("Speed optimized: maximizing workers and chunk sizes")

        elif target_mode == "memory_optimized":
            recommended_workers = max(1, base_workers // 2)
            memory_per_worker = 256
            chunk_size = 50000
            batch_size = 25
            reasoning.append("Memory optimized: minimizing memory usage")

        elif target_mode == "cost_optimized":
            recommended_workers = max(2, base_workers // 2)
            memory_per_worker = 512
            chunk_size = 100000
            batch_size = 50
            reasoning.append("Cost optimized: balancing performance and resource usage")

        else:  # balanced
            recommended_workers = base_workers
            memory_per_worker = min(
                768, int(available_memory_gb * 1024 / recommended_workers)
            )
            chunk_size = 100000
            batch_size = 50
            reasoning.append("Balanced mode: optimal performance/resource ratio")

        # Apply memory override
        if max_memory_override:
            max_memory_mb = int(max_memory_override * 1024)
            if recommended_workers * memory_per_worker > max_memory_mb:
                recommended_workers = max(1, max_memory_mb // memory_per_worker)
                reasoning.append(
                    f"Memory override: adjusted workers for {max_memory_override}GB limit"
                )

        # Calculate chunk overlap (10% of chunk size)
        chunk_overlap = chunk_size // 10

        # Estimate processing metrics
        estimated_duration = (
            dataset.total_tickets
            * dataset.estimated_processing_time_per_ticket
            / recommended_workers
            / 60
        )
        estimated_peak_memory = (
            recommended_workers * memory_per_worker / 1024
        )  # Convert to GB

        # Use historical data if available
        if performance["confidence"] > 0.3:
            estimated_duration = performance["predicted_duration_minutes"]
            reasoning.append(
                f"Using historical data (confidence: {performance['confidence']:.2f})"
            )

        # Estimate cost (simplified)
        base_cost_per_ticket = 0.0003  # ~$0.0003 with Gemini 2.5 Flash
        worker_efficiency = min(
            2.0, recommended_workers / 4
        )  # Efficiency bonus for more workers
        estimated_cost = (
            dataset.total_tickets * base_cost_per_ticket / worker_efficiency
        )

        # Calculate confidence score
        confidence_factors = [
            performance["confidence"],  # Historical data confidence
            min(1.0, dataset.total_tickets / 1000),  # Dataset size confidence
            min(1.0, available_memory_gb / 4),  # System resource confidence
        ]
        confidence_score = statistics.mean(confidence_factors)

        return AllocationRecommendation(
            recommended_workers=recommended_workers,
            memory_per_worker_mb=memory_per_worker,
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
            batch_size=batch_size,
            processing_mode=target_mode,
            estimated_duration_minutes=estimated_duration,
            estimated_peak_memory_gb=estimated_peak_memory,
            estimated_cost=estimated_cost,
            confidence_score=confidence_score,
            reasoning=reasoning,
        )

    def validate_recommendation(
        self, recommendation: AllocationRecommendation
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a recommendation is feasible given current system resources.

        Args:
            recommendation: Allocation recommendation to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        system = self.system_profiler.get_current_resources()

        # Check worker count
        if recommendation.recommended_workers > system.cpu_cores * 4:
            issues.append(
                f"Too many workers: {recommendation.recommended_workers} > {system.cpu_cores * 4} (4x CPU cores)"
            )

        # Check memory requirements
        total_memory_needed = (
            recommendation.recommended_workers
            * recommendation.memory_per_worker_mb
            / 1024
        )
        if total_memory_needed > system.memory_available_gb:
            issues.append(
                f"Insufficient memory: need {total_memory_needed:.1f}GB, available {system.memory_available_gb:.1f}GB"
            )

        # Check if system is already under load
        if system.cpu_usage_percent > 80:
            issues.append(
                f"High CPU usage: {system.cpu_usage_percent:.1f}%, may impact performance"
            )

        if system.memory_usage_percent > 85:
            issues.append(
                f"High memory usage: {system.memory_usage_percent:.1f}%, may cause issues"
            )

        # Check disk space for temporary files
        estimated_temp_space = (
            recommendation.estimated_peak_memory_gb * 2
        )  # Conservative estimate
        if estimated_temp_space > system.disk_space_available_gb:
            issues.append(
                f"Insufficient disk space: need ~{estimated_temp_space:.1f}GB for temp files"
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_allocation_report(
        self, recommendation: AllocationRecommendation
    ) -> Dict[str, Any]:
        """Generate comprehensive allocation report."""
        system = self.system_profiler.get_current_resources()
        is_valid, issues = self.validate_recommendation(recommendation)

        return {
            "recommendation": asdict(recommendation),
            "system_resources": asdict(system),
            "validation": {"is_valid": is_valid, "issues": issues},
            "resource_utilization": {
                "cpu_cores_used": recommendation.recommended_workers,
                "cpu_utilization_percent": (
                    recommendation.recommended_workers / system.cpu_cores
                )
                * 100,
                "memory_used_gb": recommendation.estimated_peak_memory_gb,
                "memory_utilization_percent": (
                    recommendation.estimated_peak_memory_gb / system.memory_total_gb
                )
                * 100,
            },
            "performance_estimates": {
                "throughput_tickets_per_minute": (
                    60 / recommendation.estimated_duration_minutes
                    if recommendation.estimated_duration_minutes > 0
                    else 0
                ),
                "cost_per_ticket": (
                    recommendation.estimated_cost
                    / max(1, recommendation.estimated_duration_minutes)
                    if recommendation.estimated_duration_minutes > 0
                    else 0
                ),
                "efficiency_score": recommendation.confidence_score
                * (1 - len(issues) * 0.2),  # Reduce score for issues
            },
        }
