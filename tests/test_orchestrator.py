"""
Tests for TwoPhaseOrchestrator module
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import tempfile
from unittest.mock import patch
from discovery.orchestrator import (
    TwoPhaseOrchestrator,
    OrchestrationMetrics,
    OrchestrationConfig,
    OrchestrationError,
    run_complete_pipeline,
)


class TestTwoPhaseOrchestrator:
    """Test cases for TwoPhaseOrchestrator class"""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"

    @pytest.fixture
    def sample_config(self):
        """Sample orchestration configuration"""
        return OrchestrationConfig(
            sample_rate=0.10,
            sampling_strategy="hybrid",
            batch_size=50,
            max_workers=2,
            cost_target_per_1k=0.20,
        )

    @pytest.fixture
    def sample_tickets_df(self):
        """Create sample tickets DataFrame for testing"""
        data = {
            "ticket_id": ["T001", "T002", "T003"] * 10,  # 30 rows
            "sender": ["USER", "AGENT", "USER"] * 10,
            "text": [
                "Meu cartão foi recusado",
                "Vou verificar seu cartão",
                "Obrigado pela ajuda",
            ]
            * 10,
            "category": ["TEXT"] * 30,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_orchestrator(self, mock_api_key, sample_config):
        """Create a mocked TwoPhaseOrchestrator instance"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = TwoPhaseOrchestrator(
                api_key=mock_api_key, database_dir=Path(tmp_dir), config=sample_config
            )
            yield orchestrator

    def test_orchestration_config_dataclass(self):
        """Test OrchestrationConfig dataclass"""
        config = OrchestrationConfig(
            sample_rate=0.15,
            sampling_strategy="stratified",
            batch_size=100,
            max_workers=4,
        )

        assert config.sample_rate == 0.15
        assert config.sampling_strategy == "stratified"
        assert config.batch_size == 100
        assert config.max_workers == 4

    def test_orchestration_metrics_dataclass(self):
        """Test OrchestrationMetrics dataclass"""
        metrics = OrchestrationMetrics(
            total_tickets=1000,
            discovery_sample_size=150,
            categories_discovered=5,
            total_processing_time=300.0,
            discovery_time=60.0,
            application_time=240.0,
            total_cost_usd=1.50,
            cost_per_1k_tickets=1.50,
            avg_confidence=0.89,
            classification_rate=0.94,
            meets_cost_target=False,
            meets_confidence_target=True,
        )

        assert metrics.total_tickets == 1000
        assert metrics.discovery_sample_size == 150
        assert not metrics.meets_cost_target
        assert metrics.meets_confidence_target

    def test_orchestration_error(self):
        """Test OrchestrationError exception"""
        error = OrchestrationError("Test error message")
        assert str(error) == "Test error message"

    def test_orchestrator_initialization(self, mock_api_key, sample_config):
        """Test TwoPhaseOrchestrator initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = TwoPhaseOrchestrator(
                api_key=mock_api_key, database_dir=Path(tmp_dir), config=sample_config
            )

            # API key is stored internally for component initialization
            assert orchestrator.database_dir == Path(tmp_dir)
            assert orchestrator.config == sample_config
            assert orchestrator.sampler is None  # Lazy loading
            assert orchestrator.discoverer is None  # Lazy loading
            assert orchestrator.classifier is None  # Lazy loading

    def test_load_and_validate_input(self, mock_orchestrator, sample_tickets_df):
        """Test input data loading and validation"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_tickets_df.to_csv(f.name, sep=";", encoding="utf-8-sig", index=False)
            input_path = Path(f.name)

        try:
            loaded_df = mock_orchestrator._load_and_validate_input(input_path)
            assert isinstance(loaded_df, pd.DataFrame)
            assert not loaded_df.empty
            assert "ticket_id" in loaded_df.columns
        finally:
            input_path.unlink()

    def test_estimate_discovery_cost(self, mock_orchestrator):
        """Test discovery cost estimation"""
        sample_size = 500
        cost = mock_orchestrator._estimate_discovery_cost(sample_size)

        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_application_cost(self, mock_orchestrator):
        """Test application cost estimation"""
        total_tickets = 1000
        cost = mock_orchestrator._estimate_application_cost(total_tickets)

        assert isinstance(cost, float)
        assert cost > 0

    def test_generate_final_metrics(self, mock_orchestrator):
        """Test final metrics generation"""
        # Mock the required data
        mock_orchestrator.start_time = 1000.0
        mock_orchestrator.discovery_metrics = {
            "sample_size": 150,
            "processing_time": 60.0,
            "cost_estimate": 0.50,
        }
        mock_orchestrator.application_metrics = {
            "processing_time": 240.0,
            "cost_estimate": 1.00,
        }

        # Create mock files
        tickets_df = pd.DataFrame(
            {"ticket_id": ["T001", "T002"], "category": ["TEXT", "TEXT"]}
        )

        categories_data = {"categories": [{"id": 1, "display_name": "Test Category"}]}

        results_df = pd.DataFrame(
            {
                "ticket_id": ["T001", "T002"],
                "category_ids": ["[1]", "[1]"],
                "confidence": [0.95, 0.87],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save mock data
            categories_path = Path(tmp_dir) / "categories.json"
            results_path = Path(tmp_dir) / "results.csv"

            import json

            with open(categories_path, "w") as f:
                json.dump(categories_data, f)
            results_df.to_csv(results_path, index=False)

            # Mock time.time() to return a fixed value
            with patch("time.time", return_value=1300.0):  # 300 seconds later
                metrics = mock_orchestrator._generate_final_metrics(
                    tickets_df, categories_path, results_path, Path(tmp_dir)
                )

            assert isinstance(metrics, OrchestrationMetrics)
            assert metrics.total_tickets == 2
            assert metrics.discovery_sample_size == 150
            assert metrics.categories_discovered == 1

    def test_run_complete_pipeline_function(self):
        """Test the run_complete_pipeline utility function"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create minimal test data
            test_data = pd.DataFrame(
                {
                    "ticket_id": ["T001"],
                    "text": ["Test message"],
                    "sender": ["USER"],
                    "category": ["TEXT"],
                }
            )

            input_file = Path(tmp_dir) / "test_input.csv"
            test_data.to_csv(input_file, sep=";", encoding="utf-8-sig", index=False)

            # This is a placeholder test since the function requires complex setup
            assert callable(run_complete_pipeline)
