"""
Tests for the Scalability Framework.

This module contains comprehensive tests for all scalability components,
including unit tests and integration tests.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch

# Import modules to test
# Note: Tests should be run with PYTHONPATH=src environment variable
# Example: PYTHONPATH=src python -m pytest tests/
# Or use proper package structure with __init__.py files

from scalability_framework import ScalabilityFramework, ScalabilityConfiguration
from scalability_manager import ScalabilityManager
from streaming_processor import (
    StreamingDataProcessor,
    StreamingConfig,
    ChunkedFileReader,
)
from resource_allocator import (
    ResourceAllocator,
    DatasetCharacteristics,
    AllocationRecommendation,
)
from cost_models import CostCalculator, LinearScalingModel, CostOptimizer
from auto_scaler import AutoScaler, ScalingThresholds


class TestScalabilityManager:
    """Test cases for ScalabilityManager."""

    def test_initialization(self):
        """Test ScalabilityManager initialization."""
        manager = ScalabilityManager(max_workers=8, worker_type="thread")

        assert manager.max_workers == 8
        assert manager.worker_type == "thread"
        assert len(manager.RESOURCE_PROFILES) > 0
        assert manager.active_workers == {}

    def test_get_optimal_resource_profile(self):
        """Test resource profile selection for different dataset sizes."""
        manager = ScalabilityManager()

        # Test small dataset
        profile_small = manager.get_optimal_resource_profile(500)
        assert profile_small.recommended_workers <= 4

        # Test large dataset
        profile_large = manager.get_optimal_resource_profile(100000)
        assert profile_large.recommended_workers >= 8

        # Test very large dataset
        profile_xlarge = manager.get_optimal_resource_profile(500000)
        assert profile_xlarge.recommended_workers >= 16

    def test_estimate_processing_cost(self):
        """Test cost estimation."""
        manager = ScalabilityManager()

        cost_estimate = manager.estimate_processing_cost(10000)

        assert "base_processing_cost" in cost_estimate
        assert "total_estimated_cost" in cost_estimate
        assert "workers_needed" in cost_estimate
        assert cost_estimate["total_estimated_cost"] > 0
        assert cost_estimate["workers_needed"] > 0

    def test_auto_configure_for_dataset(self):
        """Test automatic configuration."""
        manager = ScalabilityManager()

        config = manager.auto_configure_for_dataset(5000)

        assert "dataset_size" in config
        assert "workers" in config
        assert "estimated_duration_minutes" in config
        assert config["dataset_size"] == 5000
        assert config["workers"] > 0


class TestStreamingProcessor:
    """Test cases for StreamingDataProcessor."""

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create sample data
            data = {
                "ticket_id": [f"TICKET_{i:06d}" for i in range(100)],
                "category": ["TEXT"] * 100,
                "text": [f"Sample text content for ticket {i}" * 5 for i in range(100)],
                "sender": ["USER" if i % 2 == 0 else "AGENT" for i in range(100)],
            }
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False, sep=";")

        return Path(f.name)

    def test_streaming_config(self):
        """Test StreamingConfig creation."""
        config = StreamingConfig(
            chunk_size_rows=500, buffer_size_mb=32, memory_limit_mb=256
        )

        assert config.chunk_size_rows == 500
        assert config.buffer_size_mb == 32
        assert config.memory_limit_mb == 256

    def test_chunked_file_reader(self, temp_csv_file):
        """Test ChunkedFileReader functionality."""
        config = StreamingConfig(chunk_size_rows=25)
        reader = ChunkedFileReader(config)

        chunks = list(reader.read_csv_chunks(temp_csv_file, sep=";"))

        assert len(chunks) == 4  # 100 rows / 25 per chunk
        assert len(chunks[0]) == 25
        assert "ticket_id" in chunks[0].columns

    def test_streaming_data_processor(self, temp_csv_file):
        """Test StreamingDataProcessor."""
        config = StreamingConfig(chunk_size_rows=30)
        processor = StreamingDataProcessor(config)

        def mock_processor_function(chunk, chunk_number):
            return [{"processed": len(chunk), "chunk": chunk_number}]

        # Test processing
        results = list(
            processor.process_file_streaming(
                input_file=temp_csv_file, processor_function=mock_processor_function
            )
        )

        assert len(results) > 0
        assert all(isinstance(result, list) for result in results)

        # Cleanup
        temp_csv_file.unlink()


class TestResourceAllocator:
    """Test cases for ResourceAllocator."""

    @pytest.fixture
    def temp_dataset_file(self):
        """Create temporary dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data = {
                "ticket_id": [f"T{i}" for i in range(50)],
                "text": [
                    f"Sample ticket text content number {i}" * 10 for i in range(50)
                ],
                "category": ["TEXT"] * 50,
                "sender": ["USER"] * 50,
            }
            pd.DataFrame(data).to_csv(f.name, index=False, sep=";")
        return Path(f.name)

    def test_dataset_analyzer(self, temp_dataset_file):
        """Test DatasetAnalyzer."""
        allocator = ResourceAllocator()

        characteristics = allocator.dataset_analyzer.analyze_dataset(
            temp_dataset_file, sample_size=25
        )

        assert isinstance(characteristics, DatasetCharacteristics)
        assert characteristics.total_tickets > 0
        assert characteristics.avg_text_length > 0
        assert 0 <= characteristics.text_complexity_score <= 1

        # Cleanup
        temp_dataset_file.unlink()

    def test_system_profiler(self):
        """Test SystemProfiler."""
        allocator = ResourceAllocator()

        resources = allocator.system_profiler.get_current_resources()

        assert resources.cpu_cores > 0
        assert resources.memory_total_gb > 0
        assert 0 <= resources.cpu_usage_percent <= 100
        assert 0 <= resources.memory_usage_percent <= 100

    @patch("src.resource_allocator.Path.exists")
    def test_analyze_and_recommend(self, mock_exists, temp_dataset_file):
        """Test analysis and recommendation generation."""
        mock_exists.return_value = True

        allocator = ResourceAllocator()

        with patch.object(
            allocator.dataset_analyzer, "analyze_dataset"
        ) as mock_analyze:
            mock_analyze.return_value = DatasetCharacteristics(
                total_tickets=1000,
                avg_text_length=500,
                max_text_length=2000,
                min_text_length=50,
                text_complexity_score=0.6,
                language_diversity=0.3,
                estimated_processing_time_per_ticket=0.5,
                total_size_mb=10.0,
            )

            recommendation = allocator.analyze_and_recommend(temp_dataset_file)

            assert isinstance(recommendation, AllocationRecommendation)
            assert recommendation.recommended_workers > 0
            assert recommendation.memory_per_worker_mb > 0
            assert recommendation.confidence_score >= 0

        # Cleanup
        temp_dataset_file.unlink()


class TestCostModels:
    """Test cases for cost modeling components."""

    def test_cost_calculator(self):
        """Test CostCalculator functionality."""
        calculator = CostCalculator()

        # Test AI API cost calculation
        ai_cost = calculator.calculate_ai_api_cost(1000, 500)
        assert ai_cost > 0

        # Test infrastructure cost
        infra_cost = calculator.calculate_infrastructure_cost(4, 2.0, "local")
        assert infra_cost > 0

        # Test total cost calculation
        total_cost = calculator.calculate_total_cost(
            input_tokens=10000,
            output_tokens=2000,
            workers=4,
            duration_hours=1.0,
            data_size_gb=1.0,
        )

        assert total_cost.total_cost > 0
        assert total_cost.ai_api_cost > 0
        assert total_cost.infrastructure_cost > 0

    def test_linear_scaling_model(self):
        """Test LinearScalingModel."""
        model = LinearScalingModel(base_cost_per_ticket=0.05)

        # Test cost projection
        dataset_sizes = [1000, 5000, 10000]
        projection = model.project_costs(dataset_sizes)

        assert len(projection.dataset_sizes) == 3
        assert len(projection.total_costs) == 3
        assert all(cost > 0 for cost in projection.total_costs)
        assert all(workers > 0 for workers in projection.workers_needed)

    def test_cost_optimizer(self):
        """Test CostOptimizer."""
        optimizer = CostOptimizer()

        # Test budget optimization
        budget_optimization = optimizer.optimize_for_budget(
            dataset_size=5000, budget_limit=100.0
        )

        assert budget_optimization.optimized_cost >= 0
        assert len(budget_optimization.recommended_changes) > 0

        # Test time optimization
        time_optimization = optimizer.optimize_for_time(
            dataset_size=5000, time_limit_hours=4.0
        )

        assert len(time_optimization.recommended_changes) > 0


class TestAutoScaler:
    """Test cases for AutoScaler."""

    def test_scaling_thresholds(self):
        """Test ScalingThresholds configuration."""
        thresholds = ScalingThresholds(
            queue_scale_up_threshold=150,
            cpu_scale_up_threshold=80.0,
            min_workers=1,
            max_workers=32,
        )

        assert thresholds.queue_scale_up_threshold == 150
        assert thresholds.cpu_scale_up_threshold == 80.0
        assert thresholds.min_workers == 1
        assert thresholds.max_workers == 32

    def test_auto_scaler_initialization(self):
        """Test AutoScaler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaler = AutoScaler(
                initial_workers=4, worker_type="thread", storage_dir=Path(temp_dir)
            )

            assert scaler.worker_manager.get_current_worker_count() == 4
            assert not scaler.auto_scaling_active

            # Cleanup
            scaler.cleanup()

    def test_manual_scaling(self):
        """Test manual scaling functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaler = AutoScaler(initial_workers=2, storage_dir=Path(temp_dir))

            # Test scaling up
            success = scaler.manual_scale(4, "Test scale up")
            assert success
            assert scaler.worker_manager.get_current_worker_count() == 4

            # Test scaling down
            success = scaler.manual_scale(2, "Test scale down")
            assert success
            assert scaler.worker_manager.get_current_worker_count() == 2

            # Cleanup
            scaler.cleanup()


class TestScalabilityFramework:
    """Integration tests for the complete ScalabilityFramework."""

    @pytest.fixture
    def temp_framework_dir(self):
        """Create temporary directory for framework."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_dataset(self, temp_framework_dir):
        """Create sample dataset."""
        dataset_file = temp_framework_dir / "sample_data.csv"

        data = {
            "ticket_id": [f"TICKET_{i:06d}" for i in range(200)],
            "category": ["TEXT"] * 200,
            "text": [f"Sample ticket content for analysis {i}" * 8 for i in range(200)],
            "sender": ["USER" if i % 3 != 0 else "AGENT" for i in range(200)],
        }

        pd.DataFrame(data).to_csv(dataset_file, index=False, sep=";")
        return dataset_file

    def test_framework_initialization(self, temp_framework_dir):
        """Test framework initialization."""
        config = ScalabilityConfiguration(max_workers=4, storage_dir=temp_framework_dir)

        framework = ScalabilityFramework(config)

        assert framework.config.max_workers == 4
        assert framework.storage_dir == temp_framework_dir
        assert framework.scalability_manager is not None
        assert framework.resource_allocator is not None
        assert framework.cost_analytics is not None

        # Cleanup
        framework.cleanup()

    def test_analyze_dataset_requirements(self, temp_framework_dir, sample_dataset):
        """Test complete dataset analysis."""
        config = ScalabilityConfiguration(storage_dir=temp_framework_dir)
        framework = ScalabilityFramework(config)

        try:
            analysis = framework.analyze_dataset_requirements(sample_dataset)

            # Verify analysis structure
            assert "analysis_metadata" in analysis
            assert "resource_allocation" in analysis
            assert "cost_analysis" in analysis
            assert "scalability_config" in analysis
            assert "validation" in analysis
            assert "framework_recommendations" in analysis
            assert "recommended_configuration" in analysis

            # Verify recommended configuration
            rec_config = analysis["recommended_configuration"]
            assert "workers" in rec_config
            assert "memory_per_worker_mb" in rec_config
            assert "estimated_duration_minutes" in rec_config
            assert rec_config["workers"] > 0

        finally:
            framework.cleanup()

    def test_optimize_for_constraints(self, temp_framework_dir):
        """Test constraint optimization."""
        config = ScalabilityConfiguration(storage_dir=temp_framework_dir)
        framework = ScalabilityFramework(config)

        try:
            constraints = {
                "budget_limit": 50.0,
                "time_limit_hours": 3.0,
                "memory_limit_gb": 8.0,
            }

            optimization = framework.optimize_for_constraints(
                dataset_size=5000, constraints=constraints
            )

            assert "constraints" in optimization
            assert "individual_optimizations" in optimization
            assert "combined_recommendation" in optimization
            assert "trade_offs" in optimization

            combined = optimization["combined_recommendation"]
            assert "workers" in combined
            assert "optimization_priority" in combined

        finally:
            framework.cleanup()

    def test_get_real_time_dashboard(self, temp_framework_dir):
        """Test real-time dashboard functionality."""
        config = ScalabilityConfiguration(storage_dir=temp_framework_dir)
        framework = ScalabilityFramework(config)

        try:
            dashboard = framework.get_real_time_dashboard()

            assert "framework_status" in dashboard
            assert "current_session_stats" in dashboard
            assert "scalability_metrics" in dashboard
            assert "resource_utilization" in dashboard
            assert "cost_tracking" in dashboard

            status = dashboard["framework_status"]
            assert "auto_scaling_enabled" in status
            assert "framework_config" in status

        finally:
            framework.cleanup()

    def test_export_framework_summary(self, temp_framework_dir):
        """Test framework summary export."""
        config = ScalabilityConfiguration(storage_dir=temp_framework_dir)
        framework = ScalabilityFramework(config)

        try:
            summary = framework.export_framework_summary()

            assert "framework_info" in summary
            assert "current_configuration" in summary
            assert "storage_locations" in summary
            assert "usage_examples" in summary

            info = summary["framework_info"]
            assert "version" in info
            assert "components" in info
            assert "capabilities" in info
            assert len(info["components"]) >= 5  # Should have 5 main components

        finally:
            framework.cleanup()


class TestErrorHandling:
    """Test error handling across the framework."""

    def test_invalid_file_handling(self):
        """Test handling of invalid input files."""
        config = ScalabilityConfiguration()
        framework = ScalabilityFramework(config)

        try:
            # Test with non-existent file
            invalid_file = Path("/nonexistent/file.csv")

            # Use specific exception types for better error handling
            with pytest.raises((FileNotFoundError, ValueError, OSError)) as exc_info:
                framework.analyze_dataset_requirements(invalid_file)
            
            # Verify the error is related to file access
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["file", "not found", "does not exist", "no such file"])

        finally:
            framework.cleanup()

    def test_resource_constraint_violations(self, temp_framework_dir):
        """Test handling of resource constraint violations."""
        # Test with impossible constraints
        config = ScalabilityConfiguration(
            max_workers=1000, storage_dir=temp_framework_dir  # Unrealistic
        )
        framework = ScalabilityFramework(config)

        try:
            constraints = {
                "budget_limit": 0.01,  # Very low budget
                "time_limit_hours": 0.01,  # Very short time
                "memory_limit_gb": 0.1,  # Very low memory
            }

            # Should not crash, but should provide warnings
            optimization = framework.optimize_for_constraints(10000, constraints)

            # Should contain information about constraints not being met
            assert "trade_offs" in optimization

        finally:
            framework.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
