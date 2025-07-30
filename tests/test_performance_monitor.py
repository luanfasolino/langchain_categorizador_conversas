"""
Tests for PerformanceMonitor module
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import tempfile
from discovery.performance_monitor import (
    PerformanceMonitor,
    CostMetrics,
    PerformanceMetrics,
    AccuracyMetrics,
    ValidationReport,
    create_performance_monitor,
    validate_opçao_d_compliance,
)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class"""

    @pytest.fixture
    def sample_orchestration_metrics(self):
        """Create sample orchestration metrics for testing"""
        from discovery.orchestrator import OrchestrationMetrics

        return OrchestrationMetrics(
            total_tickets=1000,
            discovery_sample_size=150,
            categories_discovered=5,
            total_processing_time=300.0,  # 5 minutes
            discovery_time=60.0,  # 1 minute
            application_time=240.0,  # 4 minutes
            total_cost_usd=1.50,
            cost_per_1k_tickets=1.50,
            avg_confidence=0.89,
            classification_rate=0.94,
            meets_cost_target=False,  # Exceeds 0.20 target
            meets_confidence_target=True,  # Above 0.85 target
        )

    @pytest.fixture
    def sample_classification_results(self):
        """Create sample classification results DataFrame"""
        return pd.DataFrame(
            {
                "ticket_id": ["T001", "T002", "T003", "T004", "T005"],
                "category_ids": ["[1]", "[2]", "[1,3]", "[2]", "[1]"],
                "category_names": [
                    "Problemas de Pagamento",
                    "Alterações de Reserva",
                    "Problemas de Pagamento,Problemas Técnicos",
                    "Alterações de Reserva",
                    "Problemas de Pagamento",
                ],
                "confidence": [0.95, 0.87, 0.82, 0.91, 0.93],
                "processing_time": [1.2, 1.5, 2.1, 1.3, 1.1],
                "tokens_used": [150, 180, 220, 160, 140],
            }
        )

    @pytest.fixture
    def sample_categories(self):
        """Sample categories for testing"""
        return {
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                },
                {
                    "id": 3,
                    "technical_name": "technical_issues",
                    "display_name": "Problemas Técnicos",
                },
            ]
        }

    @pytest.fixture
    def mock_monitor(self):
        """Create a mocked PerformanceMonitor instance"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(database_dir=Path(tmp_dir))
            yield monitor

    def test_cost_metrics_dataclass(self):
        """Test CostMetrics dataclass functionality"""
        cost_metrics = CostMetrics(
            total_cost_usd=2.50,
            cost_per_1k_tickets=2.50,
            discovery_cost=0.50,
            application_cost=2.00,
            target_cost=0.20,
            meets_target=False,
        )

        assert cost_metrics.total_cost_usd == 2.50
        assert cost_metrics.cost_per_1k_tickets == 2.50
        assert not cost_metrics.meets_target

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass functionality"""
        perf_metrics = PerformanceMetrics(
            total_processing_time=300.0,
            discovery_time=60.0,
            application_time=240.0,
            throughput_tickets_per_second=3.33,
            avg_processing_time_per_ticket=0.30,
        )

        assert perf_metrics.total_processing_time == 300.0
        assert perf_metrics.throughput_tickets_per_second == 3.33

    def test_accuracy_metrics_dataclass(self):
        """Test AccuracyMetrics dataclass functionality"""
        accuracy_metrics = AccuracyMetrics(
            avg_confidence=0.89,
            confidence_std=0.05,
            classification_rate=0.94,
            high_confidence_rate=0.80,
            low_confidence_rate=0.05,
            meets_confidence_target=True,
        )

        assert accuracy_metrics.avg_confidence == 0.89
        assert accuracy_metrics.meets_confidence_target

    def test_validation_report_dataclass(self):
        """Test ValidationReport dataclass functionality"""
        cost_metrics = CostMetrics(2.50, 2.50, 0.50, 2.00, 0.20, False)
        perf_metrics = PerformanceMetrics(300.0, 60.0, 240.0, 3.33, 0.30)
        accuracy_metrics = AccuracyMetrics(0.89, 0.05, 0.94, 0.80, 0.05, True)

        report = ValidationReport(
            cost_metrics=cost_metrics,
            performance_metrics=perf_metrics,
            accuracy_metrics=accuracy_metrics,
            overall_compliance=False,
            recommendations=["Optimize costs", "Maintain quality"],
        )

        assert not report.overall_compliance
        assert len(report.recommendations) == 2

    def test_monitor_initialization(self, mock_monitor):
        """Test PerformanceMonitor initialization"""
        assert isinstance(mock_monitor.database_dir, Path)
        assert hasattr(mock_monitor, "logger")

    def test_analyze_cost_metrics(self, mock_monitor, sample_orchestration_metrics):
        """Test cost metrics analysis"""
        cost_metrics = mock_monitor.analyze_cost_metrics(sample_orchestration_metrics)

        assert isinstance(cost_metrics, CostMetrics)
        assert cost_metrics.total_cost_usd == 1.50
        assert cost_metrics.cost_per_1k_tickets == 1.50
        assert cost_metrics.discovery_cost == 0.0  # Default when not available
        assert cost_metrics.application_cost == 0.0  # Default when not available
        assert cost_metrics.target_cost == 0.20
        assert not cost_metrics.meets_target  # 1.50 > 0.20

    def test_analyze_performance_metrics(
        self, mock_monitor, sample_orchestration_metrics
    ):
        """Test performance metrics analysis"""
        perf_metrics = mock_monitor.analyze_performance_metrics(
            sample_orchestration_metrics
        )

        assert isinstance(perf_metrics, PerformanceMetrics)
        assert perf_metrics.total_processing_time == 300.0
        assert perf_metrics.discovery_time == 60.0
        assert perf_metrics.application_time == 240.0
        assert abs(perf_metrics.throughput_tickets_per_second - 3.33) < 0.01
        assert perf_metrics.avg_processing_time_per_ticket == 0.30

    def test_analyze_accuracy_metrics(self, mock_monitor, sample_orchestration_metrics):
        """Test accuracy metrics analysis"""
        accuracy_metrics = mock_monitor.analyze_accuracy_metrics(
            sample_orchestration_metrics
        )

        assert isinstance(accuracy_metrics, AccuracyMetrics)
        assert accuracy_metrics.avg_confidence == 0.89
        assert accuracy_metrics.classification_rate == 0.94
        assert accuracy_metrics.meets_confidence_target

    def test_analyze_classification_quality(
        self, mock_monitor, sample_classification_results, sample_categories
    ):
        """Test classification quality analysis"""
        quality_metrics = mock_monitor.analyze_classification_quality(
            sample_classification_results, sample_categories
        )

        assert isinstance(quality_metrics, dict)
        assert "confidence_distribution" in quality_metrics
        assert "category_distribution" in quality_metrics
        assert "quality_indicators" in quality_metrics

        # Test confidence distribution
        conf_dist = quality_metrics["confidence_distribution"]
        assert "mean" in conf_dist
        assert "std" in conf_dist
        assert "percentiles" in conf_dist

        # Test category distribution
        cat_dist = quality_metrics["category_distribution"]
        assert len(cat_dist) > 0

    def test_generate_recommendations(self, mock_monitor):
        """Test recommendation generation"""
        cost_metrics = CostMetrics(2.50, 2.50, 0.50, 2.00, 0.20, False)
        perf_metrics = PerformanceMetrics(300.0, 60.0, 240.0, 3.33, 0.30)
        accuracy_metrics = AccuracyMetrics(0.75, 0.05, 0.94, 0.60, 0.15, False)

        recommendations = mock_monitor.generate_recommendations(
            cost_metrics, perf_metrics, accuracy_metrics
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend cost optimization
        cost_rec = any("cost" in rec.lower() for rec in recommendations)
        # Should recommend confidence improvement
        conf_rec = any("confidence" in rec.lower() for rec in recommendations)
        assert cost_rec or conf_rec

    def test_create_validation_report(
        self,
        mock_monitor,
        sample_orchestration_metrics,
        sample_classification_results,
        sample_categories,
    ):
        """Test complete validation report creation"""
        report = mock_monitor.create_validation_report(
            sample_orchestration_metrics,
            sample_classification_results,
            sample_categories,
        )

        assert isinstance(report, ValidationReport)
        assert isinstance(report.cost_metrics, CostMetrics)
        assert isinstance(report.performance_metrics, PerformanceMetrics)
        assert isinstance(report.accuracy_metrics, AccuracyMetrics)
        assert isinstance(report.recommendations, list)

    def test_save_validation_report(
        self,
        mock_monitor,
        sample_orchestration_metrics,
        sample_classification_results,
        sample_categories,
    ):
        """Test saving validation report to file"""
        report = mock_monitor.create_validation_report(
            sample_orchestration_metrics,
            sample_classification_results,
            sample_categories,
        )

        output_path = mock_monitor.database_dir / "test_validation_report.json"
        mock_monitor.save_validation_report(report, output_path)

        assert output_path.exists()

        # Verify content
        import json

        with open(output_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert "cost_metrics" in saved_data
        assert "performance_metrics" in saved_data
        assert "accuracy_metrics" in saved_data
        assert "overall_compliance" in saved_data
        assert "recommendations" in saved_data

    def test_load_validation_report(self, mock_monitor):
        """Test loading validation report from file"""
        # Create test report data
        test_data = {
            "cost_metrics": {
                "total_cost_usd": 1.50,
                "cost_per_1k_tickets": 1.50,
                "discovery_cost": 0.50,
                "application_cost": 1.00,
                "target_cost": 0.20,
                "meets_target": False,
            },
            "performance_metrics": {
                "total_processing_time": 300.0,
                "discovery_time": 60.0,
                "application_time": 240.0,
                "throughput_tickets_per_second": 3.33,
                "avg_processing_time_per_ticket": 0.30,
            },
            "accuracy_metrics": {
                "avg_confidence": 0.89,
                "confidence_std": 0.05,
                "classification_rate": 0.94,
                "high_confidence_rate": 0.80,
                "low_confidence_rate": 0.05,
                "meets_confidence_target": True,
            },
            "overall_compliance": False,
            "recommendations": ["Optimize costs"],
            "timestamp": "2024-01-01T10:00:00",
        }

        # Save test data
        test_path = mock_monitor.database_dir / "test_load_report.json"
        import json

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2)

        # Load and verify
        loaded_report = mock_monitor.load_validation_report(test_path)
        assert isinstance(loaded_report, ValidationReport)
        assert loaded_report.cost_metrics.total_cost_usd == 1.50
        assert not loaded_report.overall_compliance

    def test_create_performance_monitor_function(self):
        """Test the create_performance_monitor utility function"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = create_performance_monitor(database_dir=tmp_dir)
            assert isinstance(monitor, PerformanceMonitor)
            assert monitor.database_dir == Path(tmp_dir)

    def test_validate_opcao_d_compliance_function(self):
        """Test the validate_opçao_d_compliance utility function"""
        cost_metrics = CostMetrics(0.15, 0.15, 0.05, 0.10, 0.20, True)
        accuracy_metrics = AccuracyMetrics(0.90, 0.03, 0.95, 0.85, 0.02, True)

        compliance = validate_opçao_d_compliance(cost_metrics, accuracy_metrics)
        assert compliance

        # Test non-compliance
        cost_metrics_bad = CostMetrics(0.50, 0.50, 0.20, 0.30, 0.20, False)
        compliance_bad = validate_opçao_d_compliance(cost_metrics_bad, accuracy_metrics)
        assert not compliance_bad

    def test_confidence_distribution_analysis(
        self, mock_monitor, sample_classification_results
    ):
        """Test confidence distribution analysis"""
        conf_analysis = mock_monitor._analyze_confidence_distribution(
            sample_classification_results["confidence"]
        )

        assert "mean" in conf_analysis
        assert "std" in conf_analysis
        assert "percentiles" in conf_analysis
        assert "high_confidence_rate" in conf_analysis
        assert "low_confidence_rate" in conf_analysis

        assert 0 <= conf_analysis["mean"] <= 1
        assert conf_analysis["std"] >= 0

    def test_category_distribution_analysis(
        self, mock_monitor, sample_classification_results, sample_categories
    ):
        """Test category distribution analysis"""
        cat_analysis = mock_monitor._analyze_category_distribution(
            sample_classification_results, sample_categories
        )

        assert isinstance(cat_analysis, dict)
        for category in sample_categories["categories"]:
            cat_name = category["display_name"]
            if cat_name in cat_analysis:
                assert cat_analysis[cat_name] >= 0

    def test_processing_time_analysis(
        self, mock_monitor, sample_classification_results
    ):
        """Test processing time analysis"""
        if "processing_time" in sample_classification_results.columns:
            time_analysis = mock_monitor._analyze_processing_times(
                sample_classification_results["processing_time"]
            )

            assert "mean_time" in time_analysis
            assert "median_time" in time_analysis
            assert "total_time" in time_analysis
            assert all(val >= 0 for val in time_analysis.values())

    def test_token_usage_analysis(self, mock_monitor, sample_classification_results):
        """Test token usage analysis"""
        if "tokens_used" in sample_classification_results.columns:
            token_analysis = mock_monitor._analyze_token_usage(
                sample_classification_results["tokens_used"]
            )

            assert "total_tokens" in token_analysis
            assert "avg_tokens_per_ticket" in token_analysis
            assert "token_efficiency" in token_analysis
            assert all(val >= 0 for val in token_analysis.values())

    def test_export_metrics_csv(
        self,
        mock_monitor,
        sample_orchestration_metrics,
        sample_classification_results,
        sample_categories,
    ):
        """Test exporting metrics to CSV format"""
        report = mock_monitor.create_validation_report(
            sample_orchestration_metrics,
            sample_classification_results,
            sample_categories,
        )

        csv_path = mock_monitor.database_dir / "metrics_export.csv"
        mock_monitor.export_metrics_csv(report, csv_path)

        assert csv_path.exists()

        # Verify CSV content
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert not df.empty
        assert len(df.columns) > 0

    def test_generate_performance_summary(self, mock_monitor):
        """Test performance summary generation"""
        cost_metrics = CostMetrics(0.18, 0.18, 0.05, 0.13, 0.20, True)
        perf_metrics = PerformanceMetrics(120.0, 30.0, 90.0, 8.33, 0.12)
        accuracy_metrics = AccuracyMetrics(0.91, 0.04, 0.96, 0.88, 0.03, True)

        summary = mock_monitor.generate_performance_summary(
            cost_metrics, perf_metrics, accuracy_metrics
        )

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "cost" in summary.lower()
        assert "performance" in summary.lower()
        assert "accuracy" in summary.lower()
