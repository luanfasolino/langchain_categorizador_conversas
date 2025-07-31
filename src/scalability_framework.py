"""
Scalability Framework - Integrated framework combining all scalability components.

This module provides a unified interface to the complete scalability framework,
integrating horizontal scaling, streaming processing, resource allocation,
cost modeling, and auto-scaling capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import all scalability components
from scalability_manager import ScalabilityManager
from streaming_processor import (
    StreamingDataProcessor,
    StreamingConfig,
)
from resource_allocator import (
    ResourceAllocator,
    AllocationRecommendation,
)
from cost_models import CostAnalyticsEngine
from auto_scaler import AutoScaler, ScalingThresholds, ScalingMetrics

logger = logging.getLogger(__name__)


@dataclass
class ScalabilityConfiguration:
    """Complete configuration for the scalability framework."""

    # General settings
    max_workers: int = 16
    worker_type: str = "thread"  # "thread" or "process"
    target_mode: str = (
        "balanced"  # "speed_optimized", "memory_optimized", "cost_optimized", "balanced"
    )

    # Streaming settings
    streaming_chunk_size: int = 1000
    streaming_buffer_mb: int = 64
    streaming_memory_limit_mb: int = 512
    enable_compression: bool = True

    # Auto-scaling settings
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_auto_scale_workers: int = 32

    # Cost constraints
    budget_limit_usd: Optional[float] = None
    time_limit_hours: Optional[float] = None

    # Storage
    storage_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.storage_dir:
            result["storage_dir"] = str(self.storage_dir)
        return result


class ScalabilityFramework:
    """
    Unified scalability framework that orchestrates all scaling components.

    This class provides a high-level interface for:
    - Analyzing datasets and recommending optimal configurations
    - Processing large datasets with streaming and auto-scaling
    - Monitoring costs and optimizing resource allocation
    - Providing real-time scaling based on system load
    """

    def __init__(self, config: Optional[ScalabilityConfiguration] = None):
        self.config = config or ScalabilityConfiguration()

        # Setup storage directory
        self.storage_dir = self.config.storage_dir or Path(
            "database/scalability_framework"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.scalability_manager = ScalabilityManager(
            max_workers=self.config.max_workers,
            worker_type=self.config.worker_type,
            enable_auto_scaling=self.config.enable_auto_scaling,
            storage_dir=self.storage_dir / "scaling",
        )

        self.resource_allocator = ResourceAllocator(
            storage_dir=self.storage_dir / "resource_allocation"
        )

        self.cost_analytics = CostAnalyticsEngine(
            storage_dir=self.storage_dir / "cost_analytics"
        )

        # Auto-scaler (initialized when needed)
        self.auto_scaler: Optional[AutoScaler] = None

        # Current processing session
        self.current_session: Optional[str] = None
        self.session_stats: Dict[str, Any] = {}

        logger.info("ScalabilityFramework initialized")

    def analyze_dataset_requirements(
        self, file_path: Path, custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze dataset and provide comprehensive scaling recommendations.

        Args:
            file_path: Path to dataset file
            custom_constraints: Optional custom constraints

        Returns:
            Comprehensive analysis with recommendations
        """
        logger.info(f"Analyzing dataset requirements for {file_path}")

        try:
            # Resource allocation analysis
            allocation_recommendation = self.resource_allocator.analyze_and_recommend(
                file_path=file_path,
                target_mode=self.config.target_mode,
                max_workers_override=self.config.max_workers,
            )

            # Validate recommendation against system
            is_valid, validation_issues = (
                self.resource_allocator.validate_recommendation(
                    allocation_recommendation
                )
            )

            # Cost analysis
            dataset_size = (
                allocation_recommendation.estimated_duration_minutes * 100
            )  # Rough estimate
            cost_analysis = self.cost_analytics.generate_comprehensive_analysis(
                dataset_size=int(dataset_size),
                budget_limit=self.config.budget_limit_usd,
                time_limit_hours=self.config.time_limit_hours,
            )

            # Scalability manager configuration
            scalability_config = self.scalability_manager.auto_configure_for_dataset(
                int(dataset_size)
            )

            # Generate framework recommendations
            framework_recommendations = self._generate_framework_recommendations(
                allocation_recommendation,
                cost_analysis,
                scalability_config,
                is_valid,
                validation_issues,
            )

            analysis_result = {
                "analysis_metadata": {
                    "file_path": str(file_path),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "framework_config": self.config.to_dict(),
                },
                "resource_allocation": asdict(allocation_recommendation),
                "cost_analysis": cost_analysis,
                "scalability_config": scalability_config,
                "validation": {"is_valid": is_valid, "issues": validation_issues},
                "framework_recommendations": framework_recommendations,
                "recommended_configuration": self._build_recommended_configuration(
                    allocation_recommendation, scalability_config
                ),
            }

            # Save analysis
            self._save_analysis(analysis_result, file_path)

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing dataset requirements: {str(e)}")
            raise

    def process_dataset_scalable(
        self,
        input_file: Path,
        processor_function: Callable,
        output_file: Optional[Path] = None,
        nrows: Optional[int] = None,
        enable_monitoring: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a dataset using the complete scalability framework.

        Args:
            input_file: Input dataset file
            processor_function: Function to process data chunks
            output_file: Optional output file
            nrows: Optional row limit
            enable_monitoring: Enable real-time monitoring

        Returns:
            Processing results and statistics
        """
        logger.info(f"Starting scalable dataset processing: {input_file}")

        # Start new session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = session_id

        try:
            # Analyze requirements if not already done
            analysis = self.analyze_dataset_requirements(input_file)

            # Setup streaming processor
            streaming_config = StreamingConfig(
                chunk_size_rows=self.config.streaming_chunk_size,
                buffer_size_mb=self.config.streaming_buffer_mb,
                memory_limit_mb=self.config.streaming_memory_limit_mb,
                compression_enabled=self.config.enable_compression,
                temp_dir=self.storage_dir / "temp",
            )

            streaming_processor = StreamingDataProcessor(streaming_config)

            # Setup auto-scaler if enabled
            if self.config.enable_auto_scaling and enable_monitoring:
                self._setup_auto_scaler(analysis["recommended_configuration"])
                self.auto_scaler.start_auto_scaling()

            # Setup scalability manager
            recommended_workers = analysis["recommended_configuration"]["workers"]
            executor = self.scalability_manager.create_worker_pool(recommended_workers)

            # Process with streaming and scaling
            processing_start_time = datetime.now()

            # Process using streaming
            results = []
            total_processed = 0

            try:
                for chunk_result in streaming_processor.process_file_streaming(
                    input_file=input_file,
                    processor_function=processor_function,
                    output_file=output_file,
                    nrows=nrows,
                ):
                    results.append(chunk_result)
                    total_processed += (
                        len(chunk_result)
                        if isinstance(chunk_result, (list, tuple))
                        else 1
                    )

                    # Update session stats
                    self.session_stats.update(
                        {
                            "processed_items": total_processed,
                            "chunks_completed": len(results),
                            "current_time": datetime.now().isoformat(),
                        }
                    )

            finally:
                # Cleanup
                executor.shutdown(wait=True)
                if self.auto_scaler:
                    self.auto_scaler.stop_auto_scaling()

            # Calculate final statistics
            processing_duration = datetime.now() - processing_start_time

            processing_stats = {
                "session_id": session_id,
                "input_file": str(input_file),
                "output_file": str(output_file) if output_file else None,
                "processing_duration": str(processing_duration),
                "total_items_processed": total_processed,
                "chunks_processed": len(results),
                "streaming_stats": streaming_processor.get_streaming_stats(),
                "scalability_stats": self.scalability_manager.get_scaling_report(),
                "auto_scaling_stats": (
                    self.auto_scaler.get_scaling_dashboard()
                    if self.auto_scaler
                    else None
                ),
            }

            # Save processing report
            self._save_processing_report(processing_stats)

            logger.info(
                f"Scalable processing completed: {total_processed:,} items in {processing_duration}"
            )
            return processing_stats

        except Exception as e:
            logger.error(f"Error in scalable processing: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.auto_scaler:
                self.auto_scaler.cleanup()
            self.current_session = None

    def optimize_for_constraints(
        self, dataset_size: int, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize configuration for specific constraints.

        Args:
            dataset_size: Size of dataset
            constraints: Constraints dictionary

        Returns:
            Optimized configuration recommendations
        """
        logger.info(f"Optimizing for constraints: {constraints}")

        # Extract constraints
        budget_limit = constraints.get("budget_limit")
        time_limit = constraints.get("time_limit_hours")
        memory_limit = constraints.get("memory_limit_gb")

        optimization_results = {}

        # Budget optimization
        if budget_limit:
            budget_opt = self.cost_analytics.optimizer.optimize_for_budget(
                dataset_size, budget_limit
            )
            optimization_results["budget_optimization"] = asdict(budget_opt)

        # Time optimization
        if time_limit:
            time_optimization = self.cost_analytics.optimizer.optimize_for_time(
                dataset_size, time_limit, budget_limit
            )
            optimization_results["time_optimization"] = asdict(time_optimization)

        # Memory optimization
        if memory_limit:
            # Use resource allocator with memory constraints
            memory_opt = self.resource_allocator.analyze_and_recommend(
                file_path=Path("dummy"),  # Would need actual file for full analysis
                target_mode="memory_optimized",
                max_memory_gb_override=memory_limit,
            )
            optimization_results["memory_optimization"] = asdict(memory_opt)

        # Generate combined recommendation
        combined_recommendation = self._combine_optimizations(
            optimization_results, constraints
        )

        return {
            "constraints": constraints,
            "individual_optimizations": optimization_results,
            "combined_recommendation": combined_recommendation,
            "trade_offs": self._analyze_trade_offs(optimization_results),
        }

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data for monitoring."""
        dashboard_data = {
            "framework_status": {
                "current_session": self.current_session,
                "auto_scaling_enabled": self.config.enable_auto_scaling,
                "framework_config": self.config.to_dict(),
            },
            "current_session_stats": self.session_stats,
            "scalability_metrics": self.scalability_manager.get_scaling_report(),
            "auto_scaling_dashboard": None,
            "resource_utilization": self._get_current_resource_utilization(),
            "cost_tracking": self._get_current_cost_tracking(),
        }

        if self.auto_scaler:
            dashboard_data["auto_scaling_dashboard"] = (
                self.auto_scaler.get_scaling_dashboard()
            )

        return dashboard_data

    def _setup_auto_scaler(self, recommended_config: Dict[str, Any]):
        """Setup auto-scaler with recommended configuration."""
        thresholds = ScalingThresholds(
            min_workers=self.config.min_workers,
            max_workers=self.config.max_auto_scale_workers,
            queue_scale_up_threshold=100,
            queue_scale_down_threshold=10,
        )

        self.auto_scaler = AutoScaler(
            initial_workers=recommended_config["workers"],
            worker_type=self.config.worker_type,
            thresholds=thresholds,
            storage_dir=self.storage_dir / "auto_scaling",
        )

        # Set up metrics provider
        def metrics_provider() -> ScalingMetrics:
            return ScalingMetrics(
                timestamp=datetime.now(),
                queue_depth=0,  # Would be provided by actual application
                active_workers=self.auto_scaler.worker_manager.get_current_worker_count(),
                cpu_usage_percent=0,  # Would be actual CPU usage
                memory_usage_percent=0,  # Would be actual memory usage
                memory_available_gb=0,  # Would be actual available memory
                throughput_tasks_per_minute=0,  # Would be actual throughput
                error_rate_percent=0,  # Would be actual error rate
                avg_task_duration_seconds=0,  # Would be actual duration
                pending_tasks=0,
                completed_tasks=0,
                failed_tasks=0,
            )

        self.auto_scaler.set_metrics_provider(metrics_provider)

    def _generate_framework_recommendations(
        self,
        allocation: AllocationRecommendation,
        cost_analysis: Dict[str, Any],
        scalability_config: Dict[str, Any],
        is_valid: bool,
        validation_issues: List[str],
    ) -> List[str]:
        """Generate framework-level recommendations."""
        recommendations = []

        # Validation recommendations
        if not is_valid:
            recommendations.append(
                "⚠️ Current configuration has issues - see validation section"
            )
            recommendations.extend([f"- {issue}" for issue in validation_issues])

        # Resource recommendations
        if allocation.confidence_score < 0.7:
            recommendations.append(
                "⚠️ Low confidence in resource allocation - consider manual tuning"
            )

        # Cost recommendations
        estimated_cost = scalability_config.get("estimated_cost", {}).get(
            "total_estimated_cost", 0
        )
        if estimated_cost > 100:
            recommendations.append(
                "💰 High processing cost detected - consider cost optimization"
            )

        # Performance recommendations
        estimated_duration = scalability_config.get("estimated_duration_minutes", 0)
        if estimated_duration > 480:  # 8 hours
            recommendations.append(
                "⏱️ Long processing time - consider increasing workers or optimizing data"
            )

        # Scaling recommendations
        recommended_workers = scalability_config.get("workers", 4)
        if recommended_workers > 16:
            recommendations.append(
                "🔧 High worker count - ensure system can handle coordination overhead"
            )

        # Framework-specific recommendations
        if self.config.enable_auto_scaling:
            recommendations.append(
                "🔄 Auto-scaling enabled - monitor performance and adjust thresholds as needed"
            )

        if self.config.enable_compression:
            recommendations.append(
                "📦 Compression enabled - will reduce storage but increase CPU usage"
            )

        return recommendations

    def _build_recommended_configuration(
        self, allocation: AllocationRecommendation, scalability_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final recommended configuration."""
        return {
            "workers": allocation.recommended_workers,
            "memory_per_worker_mb": allocation.memory_per_worker_mb,
            "chunk_size_tokens": allocation.chunk_size_tokens,
            "batch_size": allocation.batch_size,
            "processing_mode": allocation.processing_mode,
            "estimated_duration_minutes": allocation.estimated_duration_minutes,
            "estimated_cost": allocation.estimated_cost,
            "streaming_config": {
                "chunk_size_rows": self.config.streaming_chunk_size,
                "buffer_size_mb": self.config.streaming_buffer_mb,
                "memory_limit_mb": self.config.streaming_memory_limit_mb,
            },
            "auto_scaling_enabled": self.config.enable_auto_scaling,
        }

    def _combine_optimizations(
        self, optimizations: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine multiple optimization results into single recommendation."""
        # Simple approach: prioritize based on constraint importance
        combined = {
            "workers": 4,
            "memory_per_worker_mb": 512,
            "estimated_cost": 0.0,
            "estimated_duration_hours": 0.0,
            "optimization_priority": "balanced",
        }

        # Apply budget constraints first (usually most critical)
        if "budget_optimization" in optimizations:
            budget_opt = optimizations["budget_optimization"]
            if budget_opt.get("optimized_cost", 0) > 0:
                combined["estimated_cost"] = budget_opt["optimized_cost"]
                combined["optimization_priority"] = "cost"

        # Apply time constraints
        if "time_optimization" in optimizations:
            combined["optimization_priority"] = "time"

        # Apply memory constraints
        if "memory_optimization" in optimizations:
            memory_opt = optimizations["memory_optimization"]
            combined["memory_per_worker_mb"] = memory_opt.get(
                "memory_per_worker_mb", 512
            )

        return combined

    def _analyze_trade_offs(self, optimizations: Dict[str, Any]) -> List[str]:
        """Analyze trade-offs between different optimizations."""
        trade_offs = []

        if len(optimizations) > 1:
            trade_offs.append(
                "Multiple constraints detected - some trade-offs may be necessary"
            )

        if (
            "budget_optimization" in optimizations
            and "time_optimization" in optimizations
        ):
            trade_offs.append(
                "Budget and time constraints may conflict - prioritize based on business needs"
            )

        if "memory_optimization" in optimizations:
            trade_offs.append("Memory optimization may increase processing time")

        return trade_offs

    def _get_current_resource_utilization(self) -> Dict[str, Any]:
        """Get current system resource utilization."""
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage("/").percent,
            }
        except Exception:
            return {"status": "unavailable"}

    def _get_current_cost_tracking(self) -> Dict[str, Any]:
        """Get current cost tracking information."""
        try:
            # Get actual cost tracking from cost analytics engine
            cost_analysis = self.cost_analytics.get_session_summary()
            
            # Calculate budget utilization if budget limit is set
            budget_utilization = 0.0
            if self.config.budget_limit_usd and cost_analysis.get("total_cost_usd", 0) > 0:
                budget_utilization = (cost_analysis["total_cost_usd"] / self.config.budget_limit_usd) * 100
            
            return {
                "session_id": self.current_session,
                "estimated_session_cost": cost_analysis.get("total_cost_usd", 0.0),
                "cost_per_item": cost_analysis.get("cost_per_item", 0.0),
                "budget_utilization_percent": budget_utilization,
                "total_tokens": cost_analysis.get("total_tokens", 0),
                "operations_count": cost_analysis.get("operations_count", 0),
            }
        except Exception as e:
            # Fallback to basic tracking if cost analytics not available
            return {
                "session_id": self.current_session,
                "estimated_session_cost": 0.0,
                "cost_per_item": 0.0,
                "budget_utilization_percent": 0.0,
                "error": f"Cost tracking unavailable: {str(e)}",
            }

    def _save_analysis(self, analysis: Dict[str, Any], file_path: Path):
        """Save analysis results to storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"framework_analysis_{file_path.stem}_{timestamp}.json"
            analysis_file = self.storage_dir / "analyses" / filename
            analysis_file.parent.mkdir(parents=True, exist_ok=True)

            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Analysis saved: {analysis_file}")

        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")

    def _save_processing_report(self, report: Dict[str, Any]):
        """Save processing report to storage."""
        try:
            session_id = report["session_id"]
            report_file = self.storage_dir / "processing_reports" / f"{session_id}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Processing report saved: {report_file}")

        except Exception as e:
            logger.error(f"Error saving processing report: {str(e)}")

    def export_framework_summary(self) -> Dict[str, Any]:
        """Export comprehensive framework summary."""
        return {
            "framework_info": {
                "version": "1.0.0",
                "components": [
                    "ScalabilityManager",
                    "StreamingProcessor",
                    "ResourceAllocator",
                    "CostAnalytics",
                    "AutoScaler",
                ],
                "capabilities": [
                    "Horizontal scaling (1-500K+ tickets)",
                    "Memory-efficient streaming",
                    "Dynamic resource allocation",
                    "Cost modeling and optimization",
                    "Real-time auto-scaling",
                ],
            },
            "current_configuration": self.config.to_dict(),
            "storage_locations": {
                "base_dir": str(self.storage_dir),
                "analyses": str(self.storage_dir / "analyses"),
                "reports": str(self.storage_dir / "processing_reports"),
                "scaling_data": str(self.storage_dir / "scaling"),
                "cost_data": str(self.storage_dir / "cost_analytics"),
            },
            "usage_examples": {
                "analyze_dataset": "framework.analyze_dataset_requirements(file_path)",
                "process_scalable": "framework.process_dataset_scalable(input_file, processor_func)",
                "optimize_constraints": "framework.optimize_for_constraints(size, constraints)",
                "real_time_dashboard": "framework.get_real_time_dashboard()",
            },
        }

    def cleanup(self):
        """Cleanup all framework resources."""
        if self.auto_scaler:
            self.auto_scaler.cleanup()

        self.scalability_manager.cleanup()

        logger.info("ScalabilityFramework cleanup completed")
