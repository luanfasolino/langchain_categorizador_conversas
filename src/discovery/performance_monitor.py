"""
Performance Monitoring and Validation System

This module implements comprehensive monitoring, validation, and reporting
for the Opção D discovery-application pipeline with cost tracking,
performance metrics, and A/B testing capabilities.
"""

import logging
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt

# Import fallback for testing
try:
    from ..base_processor import BaseProcessor  # noqa: F401
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class CostMetrics:
    """Cost tracking metrics"""

    discovery_cost_usd: float
    application_cost_usd: float
    total_cost_usd: float
    cost_per_1k_tickets: float
    cost_per_category: float
    target_cost_per_1k: float
    meets_cost_target: bool
    cost_breakdown: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""

    total_processing_time: float
    discovery_time: float
    application_time: float
    avg_time_per_ticket: float
    throughput_tickets_per_second: float
    target_time_minutes: float
    meets_time_target: bool
    bottleneck_phase: str


@dataclass
class AccuracyMetrics:
    """Accuracy and quality metrics"""

    avg_confidence: float
    high_confidence_ratio: float
    classification_rate: float
    category_coverage: float
    consistency_score: float
    target_confidence: float
    meets_accuracy_target: bool
    validation_errors: List[str]


@dataclass
class ValidationReport:
    """Comprehensive validation report"""

    timestamp: str
    dataset_info: Dict[str, Any]
    cost_metrics: CostMetrics
    performance_metrics: PerformanceMetrics
    accuracy_metrics: AccuracyMetrics
    compliance_summary: Dict[str, bool]
    recommendations: List[str]
    next_actions: List[str]


class PerformanceMonitor:
    """
    Comprehensive performance monitoring and validation system
    for the Discovery-Application pipeline.
    """

    def __init__(
        self,
        monitoring_dir: Optional[Path] = None,
        cost_target_per_1k: float = 0.20,
        time_target_minutes: float = 25,
        confidence_target: float = 0.85,
    ):
        """
        Initialize the Performance Monitor.

        Args:
            monitoring_dir: Directory for monitoring outputs
            cost_target_per_1k: Target cost per 1000 tickets (Opção D)
            time_target_minutes: Target processing time in minutes
            confidence_target: Target average confidence level
        """
        # Setup directories
        if monitoring_dir is None:
            monitoring_dir = Path.cwd() / "database" / "monitoring"

        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Target configurations (Opção D specifications)
        self.cost_target_per_1k = cost_target_per_1k
        self.time_target_minutes = time_target_minutes
        self.confidence_target = confidence_target

        # Monitoring state
        self.current_session = None
        self.metrics_history = []
        self.validation_reports = []

        self.logger.info(
            f"PerformanceMonitor initialized with targets: "
            f"${cost_target_per_1k:.2f}/1K tickets, "
            f"{time_target_minutes}min, "
            f"{confidence_target:.2f} confidence"
        )

    def start_monitoring_session(self, session_name: str, dataset_info: Dict[str, Any]):
        """Start a new monitoring session."""
        self.current_session = {
            "name": session_name,
            "start_time": time.time(),
            "dataset_info": dataset_info,
            "phase_timings": {},
            "cost_tracking": defaultdict(float),
            "events": [],
        }

        self.logger.info(f"Started monitoring session: {session_name}")
        self.logger.info(f"Dataset: {dataset_info.get('total_tickets', 'unknown')} tickets")

    def record_phase_start(self, phase_name: str, phase_info: Dict[str, Any] = None):
        """Record the start of a processing phase."""
        if not self.current_session:
            raise ValueError("No active monitoring session")

        phase_info = phase_info or {}

        self.current_session["phase_timings"][phase_name] = {
            "start_time": time.time(),
            "info": phase_info,
        }

        self.current_session["events"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "phase_start",
                "phase": phase_name,
                "info": phase_info,
            }
        )

        self.logger.info(f"Phase started: {phase_name}")

    def record_phase_end(
        self,
        phase_name: str,
        cost_usd: float = 0.0,
        additional_metrics: Dict[str, Any] = None,
    ):
        """Record the end of a processing phase."""
        if not self.current_session:
            raise ValueError("No active monitoring session")

        if phase_name not in self.current_session["phase_timings"]:
            raise ValueError(f"Phase {phase_name} was not started")

        additional_metrics = additional_metrics or {}
        end_time = time.time()
        start_time = self.current_session["phase_timings"][phase_name]["start_time"]
        duration = end_time - start_time

        self.current_session["phase_timings"][phase_name]["end_time"] = end_time
        self.current_session["phase_timings"][phase_name]["duration"] = duration
        self.current_session["phase_timings"][phase_name]["cost_usd"] = cost_usd
        self.current_session["phase_timings"][phase_name]["metrics"] = additional_metrics

        self.current_session["cost_tracking"][phase_name] += cost_usd

        self.current_session["events"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "phase_end",
                "phase": phase_name,
                "duration": duration,
                "cost_usd": cost_usd,
                "metrics": additional_metrics,
            }
        )

        self.logger.info(f"Phase completed: {phase_name} - {duration:.1f}s, ${cost_usd:.4f}")

    def validate_pipeline_results(
        self, orchestration_metrics, categories_path: Path, results_path: Path
    ) -> ValidationReport:
        """
        Validate complete pipeline results against Opção D targets.

        Args:
            orchestration_metrics: Metrics from TwoPhaseOrchestrator
            categories_path: Path to discovered categories
            results_path: Path to classification results

        Returns:
            ValidationReport with comprehensive validation
        """
        self.logger.info("Starting comprehensive pipeline validation...")

        # Load data for validation
        with open(categories_path, "r", encoding="utf-8") as f:
            categories_data = json.load(f)

        results_df = pd.read_csv(results_path)

        # Calculate cost metrics
        cost_metrics = self._calculate_cost_metrics(orchestration_metrics)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(orchestration_metrics)

        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(
            results_df, categories_data, orchestration_metrics
        )

        # Check compliance
        compliance_summary = {
            "meets_cost_target": cost_metrics.meets_cost_target,
            "meets_time_target": performance_metrics.meets_time_target,
            "meets_accuracy_target": accuracy_metrics.meets_accuracy_target,
            "overall_compliant": all(
                [
                    cost_metrics.meets_cost_target,
                    performance_metrics.meets_time_target,
                    accuracy_metrics.meets_accuracy_target,
                ]
            ),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            cost_metrics, performance_metrics, accuracy_metrics
        )

        # Generate next actions
        next_actions = self._generate_next_actions(compliance_summary, recommendations)

        # Create validation report
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            dataset_info=(self.current_session["dataset_info"] if self.current_session else {}),
            cost_metrics=cost_metrics,
            performance_metrics=performance_metrics,
            accuracy_metrics=accuracy_metrics,
            compliance_summary=compliance_summary,
            recommendations=recommendations,
            next_actions=next_actions,
        )

        # Save report
        self._save_validation_report(report)

        self.logger.info("Pipeline validation completed")
        return report

    def _calculate_cost_metrics(self, orchestration_metrics) -> CostMetrics:
        """Calculate comprehensive cost metrics."""

        # Extract cost data
        discovery_cost = getattr(orchestration_metrics, "discovery_cost", 0.0) or 0.0
        application_cost = getattr(orchestration_metrics, "application_cost", 0.0) or 0.0
        total_cost = orchestration_metrics.total_cost_usd
        cost_per_1k = orchestration_metrics.cost_per_1k_tickets

        # Calculate cost per category
        categories_discovered = orchestration_metrics.categories_discovered
        cost_per_category = total_cost / categories_discovered if categories_discovered > 0 else 0.0

        # Check target compliance
        meets_cost_target = cost_per_1k <= self.cost_target_per_1k

        # Create cost breakdown
        cost_breakdown = {
            "discovery_phase": discovery_cost,
            "application_phase": application_cost,
            "overhead": max(0, total_cost - discovery_cost - application_cost),
        }

        return CostMetrics(
            discovery_cost_usd=discovery_cost,
            application_cost_usd=application_cost,
            total_cost_usd=total_cost,
            cost_per_1k_tickets=cost_per_1k,
            cost_per_category=cost_per_category,
            target_cost_per_1k=self.cost_target_per_1k,
            meets_cost_target=meets_cost_target,
            cost_breakdown=cost_breakdown,
        )

    def _calculate_performance_metrics(self, orchestration_metrics) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        # Extract timing data
        total_time = orchestration_metrics.total_processing_time
        discovery_time = orchestration_metrics.discovery_time
        application_time = orchestration_metrics.application_time

        # Calculate derived metrics
        total_tickets = orchestration_metrics.total_tickets
        avg_time_per_ticket = total_time / total_tickets if total_tickets > 0 else 0.0
        throughput = total_tickets / total_time if total_time > 0 else 0.0

        # Determine bottleneck
        if discovery_time > application_time:
            bottleneck_phase = "discovery"
        elif application_time > discovery_time:
            bottleneck_phase = "application"
        else:
            bottleneck_phase = "balanced"

        # Check target compliance (convert target from minutes to seconds)
        target_time_seconds = self.time_target_minutes * 60
        meets_time_target = total_time <= target_time_seconds

        return PerformanceMetrics(
            total_processing_time=total_time,
            discovery_time=discovery_time,
            application_time=application_time,
            avg_time_per_ticket=avg_time_per_ticket,
            throughput_tickets_per_second=throughput,
            target_time_minutes=self.time_target_minutes,
            meets_time_target=meets_time_target,
            bottleneck_phase=bottleneck_phase,
        )

    def _calculate_accuracy_metrics(
        self, results_df: pd.DataFrame, categories_data: Dict, orchestration_metrics
    ) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics."""

        # Basic accuracy metrics
        avg_confidence = orchestration_metrics.avg_confidence
        classification_rate = orchestration_metrics.classification_rate

        # High confidence ratio
        confidences = results_df["confidence"].astype(float)
        high_confidence_ratio = (confidences >= self.confidence_target).mean()

        # Category coverage (percentage of discovered categories actually used)
        total_categories = len(categories_data.get("categories", []))
        used_category_ids = set()

        for cat_ids_str in results_df["category_ids"].dropna():
            if cat_ids_str:
                used_category_ids.update([int(x) for x in cat_ids_str.split(",")])

        category_coverage = (
            len(used_category_ids) / total_categories if total_categories > 0 else 0.0
        )

        # Consistency score (based on confidence distribution)
        confidence_std = confidences.std()
        consistency_score = (
            max(0, 1 - (confidence_std / avg_confidence)) if avg_confidence > 0 else 0
        )

        # Check target compliance
        meets_accuracy_target = avg_confidence >= self.confidence_target

        # Identify validation errors
        validation_errors = []

        if avg_confidence < self.confidence_target:
            validation_errors.append(
                f"Average confidence {avg_confidence:.3f} below target {self.confidence_target:.3f}"
            )

        if classification_rate < 0.95:
            validation_errors.append(f"Classification rate {classification_rate:.1%} below 95%")

        if category_coverage < 0.8:
            validation_errors.append(f"Category coverage {category_coverage:.1%} below 80%")

        return AccuracyMetrics(
            avg_confidence=avg_confidence,
            high_confidence_ratio=high_confidence_ratio,
            classification_rate=classification_rate,
            category_coverage=category_coverage,
            consistency_score=consistency_score,
            target_confidence=self.confidence_target,
            meets_accuracy_target=meets_accuracy_target,
            validation_errors=validation_errors,
        )

    def _generate_recommendations(
        self,
        cost_metrics: CostMetrics,
        performance_metrics: PerformanceMetrics,
        accuracy_metrics: AccuracyMetrics,
    ) -> List[str]:
        """Generate optimization recommendations."""

        recommendations = []

        # Cost optimization recommendations
        if not cost_metrics.meets_cost_target:
            overage = cost_metrics.cost_per_1k_tickets - cost_metrics.target_cost_per_1k
            recommendations.append(
                f"Cost exceeds target by ${overage:.4f} per 1K tickets. "
                "Consider reducing batch sizes or optimizing prompts."
            )

            if cost_metrics.discovery_cost_usd > cost_metrics.application_cost_usd:
                recommendations.append(
                    "Discovery phase is more expensive. Consider reducing sample rate or chunk sizes."
                )

        # Performance optimization recommendations
        if not performance_metrics.meets_time_target:
            time_overage = (
                performance_metrics.total_processing_time / 60
            ) - performance_metrics.target_time_minutes
            recommendations.append(
                f"Processing time exceeds target by {time_overage:.1f} minutes. "
                f"Consider increasing parallelization or optimizing {performance_metrics.bottleneck_phase} phase."
            )

        # Accuracy optimization recommendations
        if not accuracy_metrics.meets_accuracy_target:
            recommendations.append(
                f"Average confidence {accuracy_metrics.avg_confidence:.3f} below target. "
                "Consider improving category definitions or using more examples."
            )

        if accuracy_metrics.category_coverage < 0.8:
            recommendations.append(
                f"Low category coverage ({accuracy_metrics.category_coverage:.1%}). "
                "Some discovered categories may be unused - consider consolidation."
            )

        if accuracy_metrics.consistency_score < 0.8:
            recommendations.append(
                "Low consistency in confidence scores. Consider reviewing prompt templates."
            )

        # General optimization recommendations
        if performance_metrics.throughput_tickets_per_second < 1.0:
            recommendations.append(
                "Low throughput detected. Consider batch optimization or parallel processing."
            )

        return recommendations

    def _generate_next_actions(
        self, compliance_summary: Dict[str, bool], recommendations: List[str]
    ) -> List[str]:
        """Generate specific next actions based on validation results."""

        next_actions = []

        if compliance_summary["overall_compliant"]:
            next_actions.append("✅ All Opção D targets met - pipeline ready for production")
            next_actions.append("Monitor performance in production environment")
            next_actions.append("Schedule periodic validation reviews")
        else:
            next_actions.append("❌ Opção D targets not fully met - optimization required")

            if not compliance_summary["meets_cost_target"]:
                next_actions.append("🔧 Optimize cost: Review API usage and prompt efficiency")
                next_actions.append("📊 Analyze cost breakdown by phase")

            if not compliance_summary["meets_time_target"]:
                next_actions.append("⚡ Optimize performance: Increase parallelization")
                next_actions.append("🔍 Profile bottleneck phases")

            if not compliance_summary["meets_accuracy_target"]:
                next_actions.append("🎯 Improve accuracy: Review category definitions")
                next_actions.append("📝 Add more training examples")

        # Add A/B testing recommendation
        next_actions.append("🧪 Consider A/B testing against single-phase approaches")

        return next_actions

    def _save_validation_report(self, report: ValidationReport):
        """Save validation report to file."""

        report_filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.monitoring_dir / report_filename

        # Convert dataclasses to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "dataset_info": report.dataset_info,
            "cost_metrics": asdict(report.cost_metrics),
            "performance_metrics": asdict(report.performance_metrics),
            "accuracy_metrics": asdict(report.accuracy_metrics),
            "compliance_summary": report.compliance_summary,
            "recommendations": report.recommendations,
            "next_actions": report.next_actions,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        self.validation_reports.append(report)
        self.logger.info(f"Validation report saved to {report_path}")

    def generate_performance_dashboard(
        self,
        validation_reports: List[ValidationReport] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate visual performance dashboard."""

        if validation_reports is None:
            validation_reports = self.validation_reports

        if not validation_reports:
            raise ValueError("No validation reports available for dashboard generation")

        if output_path is None:
            output_path = (
                self.monitoring_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Opção D Performance Dashboard", fontsize=16, fontweight="bold")

        # Extract data from reports
        costs_per_1k = [report.cost_metrics.cost_per_1k_tickets for report in validation_reports]
        processing_times = [
            report.performance_metrics.total_processing_time / 60 for report in validation_reports
        ]
        confidences = [report.accuracy_metrics.avg_confidence for report in validation_reports]
        classification_rates = [
            report.accuracy_metrics.classification_rate for report in validation_reports
        ]

        # Plot 1: Cost Tracking
        ax1.plot(range(len(costs_per_1k)), costs_per_1k, "b-o", label="Actual Cost")
        ax1.axhline(
            y=self.cost_target_per_1k,
            color="r",
            linestyle="--",
            label=f"Target (${self.cost_target_per_1k:.2f})",
        )
        ax1.set_title("Cost per 1K Tickets")
        ax1.set_ylabel("Cost (USD)")
        ax1.set_xlabel("Validation Run")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Processing Time
        ax2.plot(range(len(processing_times)), processing_times, "g-o", label="Actual Time")
        ax2.axhline(
            y=self.time_target_minutes,
            color="r",
            linestyle="--",
            label=f"Target ({self.time_target_minutes}min)",
        )
        ax2.set_title("Processing Time")
        ax2.set_ylabel("Time (minutes)")
        ax2.set_xlabel("Validation Run")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Confidence Levels
        ax3.plot(range(len(confidences)), confidences, "m-o", label="Avg Confidence")
        ax3.axhline(
            y=self.confidence_target,
            color="r",
            linestyle="--",
            label=f"Target ({self.confidence_target:.2f})",
        )
        ax3.set_title("Average Confidence")
        ax3.set_ylabel("Confidence Score")
        ax3.set_xlabel("Validation Run")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Classification Rate
        ax4.plot(
            range(len(classification_rates)),
            classification_rates,
            "c-o",
            label="Classification Rate",
        )
        ax4.axhline(y=0.95, color="r", linestyle="--", label="Target (95%)")
        ax4.set_title("Classification Success Rate")
        ax4.set_ylabel("Success Rate")
        ax4.set_xlabel("Validation Run")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Performance dashboard saved to {output_path}")
        return output_path

    def compare_against_baseline(
        self,
        current_metrics,
        baseline_metrics,
        comparison_name: str = "Current vs Baseline",
    ) -> Dict[str, Any]:
        """Compare current performance against baseline metrics."""

        comparison_results = {
            "comparison_name": comparison_name,
            "timestamp": datetime.now().isoformat(),
            "improvements": [],
            "regressions": [],
            "summary": {},
        }

        # Cost comparison
        cost_change = current_metrics.cost_per_1k_tickets - baseline_metrics.cost_per_1k_tickets
        cost_pct_change = (cost_change / baseline_metrics.cost_per_1k_tickets) * 100

        if cost_change < 0:
            comparison_results["improvements"].append(
                f"Cost reduced by ${abs(cost_change):.4f} ({abs(cost_pct_change):.1f}%)"
            )
        else:
            comparison_results["regressions"].append(
                f"Cost increased by ${cost_change:.4f} ({cost_pct_change:.1f}%)"
            )

        # Time comparison
        time_change = current_metrics.total_processing_time - baseline_metrics.total_processing_time
        time_pct_change = (time_change / baseline_metrics.total_processing_time) * 100

        if time_change < 0:
            comparison_results["improvements"].append(
                f"Time reduced by {abs(time_change):.1f}s ({abs(time_pct_change):.1f}%)"
            )
        else:
            comparison_results["regressions"].append(
                f"Time increased by {time_change:.1f}s ({time_pct_change:.1f}%)"
            )

        # Confidence comparison
        conf_change = current_metrics.avg_confidence - baseline_metrics.avg_confidence

        if conf_change > 0:
            comparison_results["improvements"].append(f"Confidence improved by {conf_change:.3f}")
        else:
            comparison_results["regressions"].append(
                f"Confidence decreased by {abs(conf_change):.3f}"
            )

        # Summary
        comparison_results["summary"] = {
            "total_improvements": len(comparison_results["improvements"]),
            "total_regressions": len(comparison_results["regressions"]),
            "cost_change_usd": cost_change,
            "time_change_seconds": time_change,
            "confidence_change": conf_change,
            "overall_better": len(comparison_results["improvements"])
            > len(comparison_results["regressions"]),
        }

        return comparison_results

    def run_ab_test_simulation(
        self,
        method_a_results: Path,
        method_b_results: Path,
        test_name: str = "A/B Test",
    ) -> Dict[str, Any]:
        """Simulate A/B test between two different approaches."""

        self.logger.info(f"Running A/B test: {test_name}")

        # Load results from both methods
        results_a = pd.read_csv(method_a_results)
        results_b = pd.read_csv(method_b_results)

        # Calculate metrics for both methods
        metrics_a = self._calculate_ab_test_metrics(results_a, "Method A")
        metrics_b = self._calculate_ab_test_metrics(results_b, "Method B")

        # Perform statistical comparison
        statistical_results = self._perform_statistical_comparison(results_a, results_b)

        # Generate A/B test report
        ab_test_report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "method_a_metrics": metrics_a,
            "method_b_metrics": metrics_b,
            "statistical_results": statistical_results,
            "recommendation": self._generate_ab_recommendation(
                metrics_a, metrics_b, statistical_results
            ),
        }

        # Save A/B test report
        ab_report_path = (
            self.monitoring_dir
            / f"ab_test_{test_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(ab_report_path, "w", encoding="utf-8") as f:
            json.dump(ab_test_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"A/B test report saved to {ab_report_path}")
        return ab_test_report

    def _calculate_ab_test_metrics(
        self, results_df: pd.DataFrame, method_name: str
    ) -> Dict[str, Any]:
        """Calculate metrics for A/B testing."""

        confidences = results_df["confidence"].astype(float)

        return {
            "method_name": method_name,
            "total_tickets": len(results_df),
            "avg_confidence": confidences.mean(),
            "median_confidence": confidences.median(),
            "confidence_std": confidences.std(),
            "high_confidence_ratio": (confidences >= self.confidence_target).mean(),
            "classification_rate": (results_df["category_ids"] != "").mean(),
            "unique_categories_used": len(
                set(
                    [
                        int(x)
                        for cat_ids in results_df["category_ids"].dropna()
                        if cat_ids
                        for x in cat_ids.split(",")
                    ]
                )
            ),
        }

    def _perform_statistical_comparison(
        self, results_a: pd.DataFrame, results_b: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two result sets."""

        from scipy import stats

        conf_a = results_a["confidence"].astype(float)
        conf_b = results_b["confidence"].astype(float)

        # T-test for confidence differences
        t_stat, p_value = stats.ttest_ind(conf_a, conf_b)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(conf_a) - 1) * conf_a.var() + (len(conf_b) - 1) * conf_b.var())
            / (len(conf_a) + len(conf_b) - 2)
        )
        cohens_d = (conf_a.mean() - conf_b.mean()) / pooled_std

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant_difference": p_value < 0.05,
            "effect_size_interpretation": self._interpret_effect_size(abs(cohens_d)),
        }

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_ab_recommendation(
        self, metrics_a: Dict, metrics_b: Dict, statistical_results: Dict
    ) -> str:
        """Generate recommendation based on A/B test results."""

        if not statistical_results["significant_difference"]:
            return (
                "No statistically significant difference found. "
                "Either method can be used with similar performance."
            )

        # Compare key metrics
        a_better_confidence = metrics_a["avg_confidence"] > metrics_b["avg_confidence"]
        a_better_classification = (
            metrics_a["classification_rate"] > metrics_b["classification_rate"]
        )

        if a_better_confidence and a_better_classification:
            return "Recommend Method A: significantly better confidence and classification rate"
        elif not a_better_confidence and not a_better_classification:
            return "Recommend Method B: significantly better confidence and classification rate"
        else:
            return (
                "Mixed results: consider business priorities. "
                f"Method A better for {'confidence' if a_better_confidence else 'classification rate'}, "
                f"Method B better for {'classification rate' if a_better_confidence else 'confidence'}."
            )


# Utility functions for integration
def create_performance_monitor(
    monitoring_dir: str,
    cost_target: float = 0.20,
    time_target: float = 25,
    confidence_target: float = 0.85,
) -> PerformanceMonitor:
    """Create a configured performance monitor."""

    return PerformanceMonitor(
        monitoring_dir=Path(monitoring_dir),
        cost_target_per_1k=cost_target,
        time_target_minutes=time_target,
        confidence_target=confidence_target,
    )


def validate_opçao_d_compliance(
    orchestration_metrics,
    categories_path: str,
    results_path: str,
    monitoring_dir: str = "database/monitoring",
) -> ValidationReport:
    """
    Validate Opção D compliance with standard targets.

    Args:
        orchestration_metrics: Metrics from TwoPhaseOrchestrator
        categories_path: Path to categories JSON
        results_path: Path to results CSV
        monitoring_dir: Directory for monitoring outputs

    Returns:
        ValidationReport with compliance assessment
    """
    monitor = create_performance_monitor(monitoring_dir)

    return monitor.validate_pipeline_results(
        orchestration_metrics=orchestration_metrics,
        categories_path=Path(categories_path),
        results_path=Path(results_path),
    )


if __name__ == "__main__":
    # Example usage
    monitor = create_performance_monitor("database/monitoring")

    # Start monitoring session
    monitor.start_monitoring_session(
        "test_session", {"total_tickets": 1000, "dataset": "test_data.csv"}
    )

    # Record phases
    monitor.record_phase_start("discovery", {"sample_rate": 0.15})
    # ... (processing would happen here)
    monitor.record_phase_end("discovery", cost_usd=0.25)

    monitor.record_phase_start("application", {"batch_size": 100})
    # ... (processing would happen here)
    monitor.record_phase_end("application", cost_usd=0.75)

    print("Performance monitoring example completed")
