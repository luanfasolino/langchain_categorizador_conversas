"""
Cost Models and Analytics - Comprehensive cost modeling system.

This module provides cost modeling framework with linear scaling projections,
cost optimization algorithms, and ROI analysis for different scaling scenarios.
"""

import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class CostComponent(Enum):
    """Different cost components in the processing pipeline."""

    AI_API_CALLS = "ai_api_calls"
    INFRASTRUCTURE = "infrastructure"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    COMPUTE_TIME = "compute_time"
    OVERHEAD = "overhead"


@dataclass
class CostBreakdown:
    """Detailed breakdown of processing costs."""

    ai_api_cost: float
    infrastructure_cost: float
    storage_cost: float
    bandwidth_cost: float
    compute_cost: float
    overhead_cost: float
    total_cost: float

    def get_component_cost(self, component: CostComponent) -> float:
        """Get cost for specific component."""
        component_map = {
            CostComponent.AI_API_CALLS: self.ai_api_cost,
            CostComponent.INFRASTRUCTURE: self.infrastructure_cost,
            CostComponent.STORAGE: self.storage_cost,
            CostComponent.BANDWIDTH: self.bandwidth_cost,
            CostComponent.COMPUTE_TIME: self.compute_cost,
            CostComponent.OVERHEAD: self.overhead_cost,
        }
        return component_map.get(component, 0.0)


@dataclass
class ScalingScenario:
    """Definition of a scaling scenario for analysis."""

    name: str
    dataset_size: int
    workers: int
    processing_mode: str
    estimated_duration_hours: float
    infrastructure_type: str  # "local", "cloud", "hybrid"


@dataclass
class CostProjection:
    """Cost projection for different dataset sizes."""

    dataset_sizes: List[int]
    total_costs: List[float]
    cost_per_ticket: List[float]
    duration_hours: List[float]
    workers_needed: List[int]
    cost_efficiency_score: List[float]


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation."""

    current_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percent: float
    recommended_changes: List[str]
    trade_offs: List[str]
    confidence_score: float


class CostCalculator:
    """Calculator for different types of processing costs."""

    # Base pricing (can be configured)
    PRICING = {
        # AI API costs (per 1K tokens)
        "gemini_input_1k_tokens": 0.075 / 1000,  # $0.075 per 1M tokens
        "gemini_output_1k_tokens": 0.30 / 1000,  # $0.30 per 1M tokens
        # Infrastructure costs (per hour)
        "worker_hour_local": 0.001,  # Electricity + depreciation
        "worker_hour_cloud": 0.05,  # Cloud compute instance
        # Storage costs (per GB per month)
        "storage_gb_month": 0.023,
        # Bandwidth costs (per GB)
        "bandwidth_gb": 0.09,
        # Fixed overhead (per processing session)
        "session_overhead": 0.10,
    }

    def __init__(self, custom_pricing: Optional[Dict[str, float]] = None):
        if custom_pricing:
            self.PRICING.update(custom_pricing)

    def calculate_ai_api_cost(
        self, input_tokens: int, output_tokens: int, model: str = "gemini"
    ) -> float:
        """Calculate AI API costs based on token usage."""
        input_cost = (input_tokens / 1000) * self.PRICING[f"{model}_input_1k_tokens"]
        output_cost = (output_tokens / 1000) * self.PRICING[f"{model}_output_1k_tokens"]
        return input_cost + output_cost

    def calculate_infrastructure_cost(
        self, workers: int, duration_hours: float, infrastructure_type: str = "local"
    ) -> float:
        """Calculate infrastructure costs."""
        cost_per_worker_hour = self.PRICING[f"worker_hour_{infrastructure_type}"]
        return workers * duration_hours * cost_per_worker_hour

    def calculate_storage_cost(
        self, data_size_gb: float, storage_duration_days: int = 30
    ) -> float:
        """Calculate storage costs."""
        months = storage_duration_days / 30
        return data_size_gb * self.PRICING["storage_gb_month"] * months

    def calculate_bandwidth_cost(self, data_transfer_gb: float) -> float:
        """Calculate bandwidth costs."""
        return data_transfer_gb * self.PRICING["bandwidth_gb"]

    def calculate_total_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        workers: int,
        duration_hours: float,
        data_size_gb: float,
        infrastructure_type: str = "local",
        storage_days: int = 30,
        data_transfer_gb: float = 0,
    ) -> CostBreakdown:
        """Calculate comprehensive cost breakdown."""

        ai_cost = self.calculate_ai_api_cost(input_tokens, output_tokens)
        infra_cost = self.calculate_infrastructure_cost(
            workers, duration_hours, infrastructure_type
        )
        storage_cost = self.calculate_storage_cost(data_size_gb, storage_days)
        bandwidth_cost = self.calculate_bandwidth_cost(data_transfer_gb)

        # Compute cost is typically included in infrastructure, but can be separate for cloud
        compute_cost = 0.0
        if infrastructure_type == "cloud":
            compute_cost = duration_hours * 0.02  # Additional compute charges

        overhead_cost = self.PRICING["session_overhead"]

        total = (
            ai_cost
            + infra_cost
            + storage_cost
            + bandwidth_cost
            + compute_cost
            + overhead_cost
        )

        return CostBreakdown(
            ai_api_cost=ai_cost,
            infrastructure_cost=infra_cost,
            storage_cost=storage_cost,
            bandwidth_cost=bandwidth_cost,
            compute_cost=compute_cost,
            overhead_cost=overhead_cost,
            total_cost=total,
        )


class LinearScalingModel:
    """Model for linear cost scaling projections."""

    def __init__(self, base_cost_per_ticket: float = 0.0003):  # ~$0.0003 per ticket with Gemini 2.5 Flash
        self.base_cost_per_ticket = base_cost_per_ticket
        self.cost_calculator = CostCalculator()

    def project_costs(
        self,
        dataset_sizes: List[int],
        workers_per_size: Optional[List[int]] = None,
        infrastructure_type: str = "local",
    ) -> CostProjection:
        """
        Project costs for different dataset sizes.

        Args:
            dataset_sizes: List of dataset sizes to analyze
            workers_per_size: Workers for each size (auto-calculated if None)
            infrastructure_type: Type of infrastructure

        Returns:
            CostProjection with detailed projections
        """
        logger.info(f"Projecting costs for {len(dataset_sizes)} dataset sizes")

        total_costs = []
        cost_per_ticket = []
        duration_hours = []
        workers_needed = []
        efficiency_scores = []

        for i, size in enumerate(dataset_sizes):
            # Auto-calculate workers if not provided
            if workers_per_size:
                workers = workers_per_size[i]
            else:
                workers = self._estimate_optimal_workers(size)

            # Estimate processing characteristics
            duration = self._estimate_duration(size, workers)

            # Estimate token usage
            avg_tokens_per_ticket = 2000  # Rough estimate
            input_tokens = size * avg_tokens_per_ticket
            output_tokens = size * 200  # Estimated output tokens

            # Calculate data characteristics
            data_size_gb = size * 0.5 / 1024  # Rough estimate: 0.5KB per ticket

            # Calculate costs
            cost_breakdown = self.cost_calculator.calculate_total_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                workers=workers,
                duration_hours=duration,
                data_size_gb=data_size_gb,
                infrastructure_type=infrastructure_type,
            )

            # Calculate efficiency score (lower cost per ticket = higher efficiency)
            base_efficiency = self.base_cost_per_ticket
            actual_cost_per_ticket = cost_breakdown.total_cost / size
            efficiency = min(1.0, base_efficiency / actual_cost_per_ticket)

            # Store results
            total_costs.append(cost_breakdown.total_cost)
            cost_per_ticket.append(actual_cost_per_ticket)
            duration_hours.append(duration)
            workers_needed.append(workers)
            efficiency_scores.append(efficiency)

        return CostProjection(
            dataset_sizes=dataset_sizes,
            total_costs=total_costs,
            cost_per_ticket=cost_per_ticket,
            duration_hours=duration_hours,
            workers_needed=workers_needed,
            cost_efficiency_score=efficiency_scores,
        )

    def _estimate_optimal_workers(self, dataset_size: int) -> int:
        """Estimate optimal number of workers for dataset size."""
        if dataset_size < 1000:
            return 2
        elif dataset_size < 10000:
            return 4
        elif dataset_size < 50000:
            return 8
        elif dataset_size < 100000:
            return 16
        elif dataset_size < 500000:
            return 32
        else:
            return 64

    def _estimate_duration(self, dataset_size: int, workers: int) -> float:
        """Estimate processing duration in hours."""
        # Base processing rate: 100 tickets per worker per hour
        base_rate = 100

        # Efficiency factor (more workers = some overhead)
        efficiency = 1.0 - (workers - 1) * 0.02  # 2% overhead per additional worker
        efficiency = max(0.5, efficiency)  # Minimum 50% efficiency

        effective_rate = base_rate * efficiency
        duration = dataset_size / (workers * effective_rate)

        return max(0.1, duration)  # Minimum 0.1 hours


class CostOptimizer:
    """Optimizer for finding cost-effective configurations."""

    def __init__(self):
        self.scaling_model = LinearScalingModel()
        self.cost_calculator = CostCalculator()

    def optimize_for_budget(
        self, dataset_size: int, budget_limit: float, infrastructure_type: str = "local"
    ) -> OptimizationRecommendation:
        """
        Find optimal configuration within budget constraints.

        Args:
            dataset_size: Size of dataset to process
            budget_limit: Maximum budget available
            infrastructure_type: Infrastructure type

        Returns:
            OptimizationRecommendation with optimal configuration
        """
        logger.info(
            f"Optimizing for budget: ${budget_limit:.2f}, dataset: {dataset_size:,} tickets"
        )

        # Test different worker configurations
        worker_options = [2, 4, 8, 16, 32, 64]
        valid_configs = []

        for workers in worker_options:
            if workers > dataset_size // 100:  # Don't use more workers than reasonable
                continue

            # Calculate cost for this configuration
            projection = self.scaling_model.project_costs(
                [dataset_size], [workers], infrastructure_type
            )

            cost = projection.total_costs[0]
            duration = projection.duration_hours[0]
            cost_per_ticket = projection.cost_per_ticket[0]

            if cost <= budget_limit:
                valid_configs.append(
                    {
                        "workers": workers,
                        "cost": cost,
                        "duration": duration,
                        "cost_per_ticket": cost_per_ticket,
                        "efficiency": projection.cost_efficiency_score[0],
                    }
                )

        if not valid_configs:
            # No configuration fits budget
            min_cost_config = self.scaling_model.project_costs(
                [dataset_size], [2], infrastructure_type
            )
            min_cost = min_cost_config.total_costs[0]

            return OptimizationRecommendation(
                current_cost=min_cost,
                optimized_cost=min_cost,
                savings_amount=0,
                savings_percent=0,
                recommended_changes=[
                    f"Budget insufficient: minimum cost ${min_cost:.2f} > budget ${budget_limit:.2f}"
                ],
                trade_offs=["Consider increasing budget or reducing dataset size"],
                confidence_score=0.9,
            )

        # Find best configuration (highest efficiency within budget)
        best_config = max(valid_configs, key=lambda x: x["efficiency"])

        # Compare with baseline (minimum workers)
        baseline_config = self.scaling_model.project_costs(
            [dataset_size], [2], infrastructure_type
        )
        baseline_cost = baseline_config.total_costs[0]

        savings = baseline_cost - best_config["cost"]
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0

        recommendations = [
            f"Use {best_config['workers']} workers for optimal efficiency",
            f"Expected duration: {best_config['duration']:.1f} hours",
            f"Cost per ticket: ${best_config['cost_per_ticket']:.4f}",
        ]

        trade_offs = []
        if best_config["workers"] > 8:
            trade_offs.append("Higher worker count may have coordination overhead")
        if best_config["duration"] < 1:
            trade_offs.append("Very fast processing - ensure system can handle load")

        return OptimizationRecommendation(
            current_cost=baseline_cost,
            optimized_cost=best_config["cost"],
            savings_amount=max(0, savings),
            savings_percent=max(0, savings_percent),
            recommended_changes=recommendations,
            trade_offs=trade_offs,
            confidence_score=0.8,
        )

    def optimize_for_time(
        self,
        dataset_size: int,
        time_limit_hours: float,
        max_budget: Optional[float] = None,
    ) -> OptimizationRecommendation:
        """
        Find optimal configuration within time constraints.

        Args:
            dataset_size: Size of dataset to process
            time_limit_hours: Maximum time allowed
            max_budget: Optional budget constraint

        Returns:
            OptimizationRecommendation with time-optimized configuration
        """
        logger.info(
            f"Optimizing for time: {time_limit_hours:.1f}h, dataset: {dataset_size:,} tickets"
        )

        # Calculate minimum workers needed to meet time constraint
        # Assume base rate of 100 tickets per worker per hour
        min_workers_needed = math.ceil(dataset_size / (100 * time_limit_hours))

        # Test worker configurations starting from minimum needed
        worker_options = [w for w in [2, 4, 8, 16, 32, 64] if w >= min_workers_needed]

        if not worker_options:
            # Time constraint impossible to meet
            return OptimizationRecommendation(
                current_cost=0,
                optimized_cost=0,
                savings_amount=0,
                savings_percent=0,
                recommended_changes=[
                    f"Time constraint impossible: need at least {min_workers_needed} workers"
                ],
                trade_offs=["Consider increasing time limit or reducing dataset size"],
                confidence_score=0.9,
            )

        valid_configs = []

        for workers in worker_options:
            projection = self.scaling_model.project_costs([dataset_size], [workers])

            cost = projection.total_costs[0]
            duration = projection.duration_hours[0]

            # Check constraints
            within_time = duration <= time_limit_hours
            within_budget = (max_budget is None) or (cost <= max_budget)

            if within_time and within_budget:
                valid_configs.append(
                    {
                        "workers": workers,
                        "cost": cost,
                        "duration": duration,
                        "cost_per_ticket": projection.cost_per_ticket[0],
                    }
                )

        if not valid_configs:
            # No valid configuration
            min_config = self.scaling_model.project_costs(
                [dataset_size], [worker_options[0]]
            )
            return OptimizationRecommendation(
                current_cost=min_config.total_costs[0],
                optimized_cost=min_config.total_costs[0],
                savings_amount=0,
                savings_percent=0,
                recommended_changes=[
                    "No configuration meets both time and budget constraints"
                ],
                trade_offs=["Consider relaxing time or budget constraints"],
                confidence_score=0.8,
            )

        # Choose configuration with lowest cost that meets time constraint
        best_config = min(valid_configs, key=lambda x: x["cost"])

        recommendations = [
            f"Use {best_config['workers']} workers to meet time constraint",
            f"Expected duration: {best_config['duration']:.1f} hours",
            f"Total cost: ${best_config['cost']:.2f}",
        ]

        return OptimizationRecommendation(
            current_cost=best_config["cost"],
            optimized_cost=best_config["cost"],
            savings_amount=0,
            savings_percent=0,
            recommended_changes=recommendations,
            trade_offs=[],
            confidence_score=0.9,
        )


class ROIAnalyzer:
    """Analyzer for Return on Investment calculations."""

    def __init__(self):
        self.cost_calculator = CostCalculator()

    def calculate_automation_roi(
        self,
        dataset_size: int,
        manual_cost_per_ticket: float,
        automated_cost_per_ticket: float,
        implementation_cost: float = 0,
        time_savings_hours: float = 0,
        hourly_rate: float = 50,
    ) -> Dict[str, Any]:
        """
        Calculate ROI for automating ticket processing.

        Args:
            dataset_size: Number of tickets to process
            manual_cost_per_ticket: Cost to process manually
            automated_cost_per_ticket: Cost to process automatically
            implementation_cost: One-time implementation cost
            time_savings_hours: Hours saved by automation
            hourly_rate: Value of time saved per hour

        Returns:
            ROI analysis dictionary
        """
        # Calculate costs
        manual_total_cost = dataset_size * manual_cost_per_ticket
        automated_total_cost = (
            dataset_size * automated_cost_per_ticket + implementation_cost
        )

        # Calculate savings
        direct_savings = manual_total_cost - automated_total_cost
        time_value_savings = time_savings_hours * hourly_rate
        total_savings = direct_savings + time_value_savings

        # Calculate ROI metrics
        roi_percent = (
            (total_savings / automated_total_cost * 100)
            if automated_total_cost > 0
            else 0
        )
        payback_period_tickets = implementation_cost / max(
            0.01, manual_cost_per_ticket - automated_cost_per_ticket
        )

        return {
            "costs": {
                "manual_total": manual_total_cost,
                "automated_total": automated_total_cost,
                "implementation": implementation_cost,
            },
            "savings": {
                "direct_savings": direct_savings,
                "time_value_savings": time_value_savings,
                "total_savings": total_savings,
            },
            "roi_metrics": {
                "roi_percent": roi_percent,
                "payback_period_tickets": payback_period_tickets,
                "cost_reduction_percent": (
                    (direct_savings / manual_total_cost * 100)
                    if manual_total_cost > 0
                    else 0
                ),
            },
            "analysis": {
                "is_profitable": total_savings > 0,
                "break_even_point": payback_period_tickets,
                "efficiency_gain": manual_cost_per_ticket
                / max(0.01, automated_cost_per_ticket),
            },
        }

    def analyze_scaling_roi(
        self,
        base_dataset_size: int,
        scaled_dataset_sizes: List[int],
        base_infrastructure_cost: float = 1000,  # One-time setup cost
    ) -> Dict[str, Any]:
        """
        Analyze ROI for scaling infrastructure to handle larger datasets.

        Args:
            base_dataset_size: Current dataset size
            scaled_dataset_sizes: Target dataset sizes for scaling
            base_infrastructure_cost: Cost to implement scaling infrastructure

        Returns:
            Scaling ROI analysis
        """
        scaling_model = LinearScalingModel()

        # Calculate current costs
        base_projection = scaling_model.project_costs([base_dataset_size])
        base_cost_per_ticket = base_projection.cost_per_ticket[0]

        # Calculate scaled costs
        scaled_projections = scaling_model.project_costs(scaled_dataset_sizes)

        roi_analyses = []
        for i, size in enumerate(scaled_dataset_sizes):
            scaled_cost_per_ticket = scaled_projections.cost_per_ticket[i]

            # ROI for processing this larger dataset
            naive_cost = (
                size * base_cost_per_ticket
            )  # Cost without scaling optimization
            optimized_cost = scaled_projections.total_costs[i] + (
                base_infrastructure_cost / len(scaled_dataset_sizes)
            )

            savings = naive_cost - optimized_cost
            roi_percent = (savings / optimized_cost * 100) if optimized_cost > 0 else 0

            roi_analyses.append(
                {
                    "dataset_size": size,
                    "naive_cost": naive_cost,
                    "optimized_cost": optimized_cost,
                    "savings": savings,
                    "roi_percent": roi_percent,
                    "cost_per_ticket": scaled_cost_per_ticket,
                }
            )

        return {
            "base_analysis": {
                "dataset_size": base_dataset_size,
                "cost_per_ticket": base_cost_per_ticket,
            },
            "scaling_analysis": roi_analyses,
            "infrastructure_cost": base_infrastructure_cost,
            "summary": {
                "total_potential_savings": sum(
                    analysis["savings"] for analysis in roi_analyses
                ),
                "average_roi_percent": statistics.mean(
                    analysis["roi_percent"] for analysis in roi_analyses
                ),
                "scaling_efficiency": min(
                    roi_analyses, key=lambda x: x["cost_per_ticket"]
                ),
            },
        }


class CostAnalyticsEngine:
    """Main engine for cost modeling and analytics."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("database/cost_analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.cost_calculator = CostCalculator()
        self.scaling_model = LinearScalingModel()
        self.optimizer = CostOptimizer()
        self.roi_analyzer = ROIAnalyzer()

        logger.info("CostAnalyticsEngine initialized")

    def generate_comprehensive_analysis(
        self,
        dataset_size: int,
        scenarios: Optional[List[ScalingScenario]] = None,
        budget_limit: Optional[float] = None,
        time_limit_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cost analysis report.

        Args:
            dataset_size: Size of dataset to analyze
            scenarios: Custom scenarios to analyze
            budget_limit: Budget constraint for optimization
            time_limit_hours: Time constraint for optimization

        Returns:
            Comprehensive analysis report
        """
        logger.info(
            f"Generating comprehensive cost analysis for {dataset_size:,} tickets"
        )

        # Default scenarios if none provided
        if not scenarios:
            scenarios = [
                ScalingScenario(
                    "Conservative", dataset_size, 4, "memory_optimized", 8.0, "local"
                ),
                ScalingScenario("Balanced", dataset_size, 8, "balanced", 4.0, "local"),
                ScalingScenario(
                    "Aggressive", dataset_size, 16, "speed_optimized", 2.0, "cloud"
                ),
            ]

        # Cost projections for different dataset sizes
        test_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
        test_sizes = [
            size for size in test_sizes if size <= dataset_size * 5
        ]  # Reasonable range

        cost_projections = self.scaling_model.project_costs(test_sizes)

        # Scenario analysis
        scenario_analysis = []
        for scenario in scenarios:
            scenario_projection = self.scaling_model.project_costs(
                [scenario.dataset_size],
                [scenario.workers],
                scenario.infrastructure_type,
            )

            scenario_analysis.append(
                {
                    "scenario": asdict(scenario),
                    "cost_breakdown": scenario_projection,
                    "cost_per_ticket": scenario_projection.cost_per_ticket[0],
                    "total_cost": scenario_projection.total_costs[0],
                }
            )

        # Optimization recommendations
        optimization_results = {}

        if budget_limit:
            optimization_results["budget_optimization"] = (
                self.optimizer.optimize_for_budget(dataset_size, budget_limit)
            )

        if time_limit_hours:
            optimization_results["time_optimization"] = (
                self.optimizer.optimize_for_time(dataset_size, time_limit_hours)
            )

        # ROI analysis
        roi_analysis = self.roi_analyzer.calculate_automation_roi(
            dataset_size=dataset_size,
            manual_cost_per_ticket=5.0,  # Assume $5 manual processing
            automated_cost_per_ticket=(
                cost_projections.cost_per_ticket[0]
                if cost_projections.cost_per_ticket
                else 0.048
            ),
            implementation_cost=2000,  # Assume $2000 implementation
            time_savings_hours=dataset_size * 0.01,  # 0.01 hours saved per ticket
            hourly_rate=50,
        )

        # Scaling ROI
        scaling_sizes = [
            size
            for size in [dataset_size * 2, dataset_size * 5, dataset_size * 10]
            if size <= 1000000
        ]
        scaling_roi = self.roi_analyzer.analyze_scaling_roi(dataset_size, scaling_sizes)

        # Cost efficiency recommendations
        efficiency_recommendations = self._generate_efficiency_recommendations(
            cost_projections
        )

        # Compile comprehensive report
        report = {
            "analysis_metadata": {
                "dataset_size": dataset_size,
                "analysis_date": datetime.now().isoformat(),
                "scenarios_analyzed": len(scenarios),
            },
            "cost_projections": asdict(cost_projections),
            "scenario_analysis": scenario_analysis,
            "optimization_results": {
                k: asdict(v) for k, v in optimization_results.items()
            },
            "roi_analysis": roi_analysis,
            "scaling_roi": scaling_roi,
            "efficiency_recommendations": efficiency_recommendations,
            "summary": {
                "recommended_scenario": min(
                    scenario_analysis, key=lambda x: x["cost_per_ticket"]
                ),
                "cost_range": {
                    "min_cost_per_ticket": min(cost_projections.cost_per_ticket),
                    "max_cost_per_ticket": max(cost_projections.cost_per_ticket),
                    "optimal_cost_per_ticket": min(cost_projections.cost_per_ticket),
                },
                "scaling_efficiency": len(
                    [c for c in cost_projections.cost_efficiency_score if c > 0.8]
                ),
            },
        }

        # Save report
        self._save_analysis_report(report, dataset_size)

        return report

    def _generate_efficiency_recommendations(
        self, projections: CostProjection
    ) -> List[str]:
        """Generate efficiency recommendations based on cost projections."""
        recommendations = []

        # Find most efficient size
        best_efficiency_idx = projections.cost_efficiency_score.index(
            max(projections.cost_efficiency_score)
        )
        best_size = projections.dataset_sizes[best_efficiency_idx]

        recommendations.append(
            f"Most cost-efficient dataset size: {best_size:,} tickets"
        )

        # Check for economies of scale
        if len(projections.cost_per_ticket) > 1:
            cost_trend = (
                projections.cost_per_ticket[-1] - projections.cost_per_ticket[0]
            )
            if cost_trend < 0:
                recommendations.append(
                    "Economies of scale detected: larger datasets are more cost-efficient"
                )
            else:
                recommendations.append(
                    "Diseconomies of scale detected: consider batch processing"
                )

        # Worker efficiency recommendations
        avg_workers = statistics.mean(projections.workers_needed)
        if avg_workers > 16:
            recommendations.append(
                "High worker count detected: ensure coordination overhead is managed"
            )
        elif avg_workers < 4:
            recommendations.append(
                "Low worker utilization: consider increasing parallelism"
            )

        return recommendations

    def _save_analysis_report(self, report: Dict[str, Any], dataset_size: int):
        """Save analysis report to storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_analysis_{dataset_size}_{timestamp}.json"
            filepath = self.storage_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Cost analysis report saved: {filepath}")

        except Exception as e:
            logger.error(f"Error saving analysis report: {str(e)}")

    def export_cost_model_summary(self) -> Dict[str, Any]:
        """Export summary of cost models and pricing."""
        return {
            "pricing_model": self.cost_calculator.PRICING,
            "base_cost_per_ticket": self.scaling_model.base_cost_per_ticket,
            "model_assumptions": {
                "avg_tokens_per_ticket": 2000,
                "output_tokens_per_ticket": 200,
                "base_processing_rate": "100 tickets per worker per hour",
                "efficiency_degradation": "2% per additional worker",
            },
            "cost_components": [component.value for component in CostComponent],
            "optimization_strategies": [
                "Budget-constrained optimization",
                "Time-constrained optimization",
                "ROI maximization",
                "Cost-efficiency optimization",
            ],
        }
