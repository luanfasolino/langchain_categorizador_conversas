"""
Tests for IntelligentSampler module
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
from discovery.intelligent_sampler import IntelligentSampler, create_diversity_report


class TestIntelligentSampler:
    """Test cases for IntelligentSampler class"""

    @pytest.fixture
    def sample_tickets_df(self):
        """Create sample tickets DataFrame for testing"""
        data = {
            "ticket_id": [f"T{i:03d}" for i in range(1, 101)],  # 100 tickets
            "text": [f"Sample ticket text {i}" for i in range(1, 101)],
            "sender": ["USER"] * 100,
            "ticket_created_at": [
                f"2024-01-{(i % 31) + 1:02d} 10:00:00" for i in range(100)
            ],
            "category": ["TEXT"] * 100,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sampler_stratified(self):
        """Create stratified sampler instance"""
        return IntelligentSampler(strategy="stratified", random_state=42)

    @pytest.fixture
    def sampler_diversity(self):
        """Create diversity sampler instance"""
        return IntelligentSampler(strategy="diversity", random_state=42)

    @pytest.fixture
    def sampler_hybrid(self):
        """Create hybrid sampler instance"""
        return IntelligentSampler(strategy="hybrid", random_state=42)

    def test_initialization_stratified(self):
        """Test IntelligentSampler initialization with stratified strategy"""
        sampler = IntelligentSampler(strategy="stratified", random_state=42)
        assert sampler.strategy == "stratified"
        assert sampler.random_state == 42

    def test_initialization_diversity(self):
        """Test IntelligentSampler initialization with diversity strategy"""
        sampler = IntelligentSampler(strategy="diversity", random_state=42)
        assert sampler.strategy == "diversity"
        assert sampler.random_state == 42

    def test_initialization_hybrid(self):
        """Test IntelligentSampler initialization with hybrid strategy"""
        sampler = IntelligentSampler(strategy="hybrid", random_state=42)
        assert sampler.strategy == "hybrid"
        assert sampler.random_state == 42

    def test_initialization_invalid_strategy(self):
        """Test IntelligentSampler initialization with invalid strategy"""
        with pytest.raises(ValueError, match="Invalid strategy"):
            IntelligentSampler(strategy="invalid", random_state=42)

    def test_stratified_sampling(self, sampler_stratified, sample_tickets_df):
        """Test stratified sampling strategy"""
        sample_size = 0.15  # 15%
        sampled_df = sampler_stratified.sample_tickets(
            sample_tickets_df, sample_size=sample_size, min_tickets=1, max_tickets=50
        )

        assert isinstance(sampled_df, pd.DataFrame)
        assert len(sampled_df) > 0
        assert len(sampled_df) <= len(sample_tickets_df)
        # Should maintain roughly the same proportion
        expected_size = int(len(sample_tickets_df) * sample_size)
        assert abs(len(sampled_df) - expected_size) <= 5  # Allow some variance

    def test_diversity_sampling(self, sampler_diversity, sample_tickets_df):
        """Test diversity sampling strategy"""
        sample_size = 0.10  # 10%
        sampled_df = sampler_diversity.sample_tickets(
            sample_tickets_df, sample_size=sample_size, min_tickets=1, max_tickets=50
        )

        assert isinstance(sampled_df, pd.DataFrame)
        assert len(sampled_df) > 0
        assert len(sampled_df) <= len(sample_tickets_df)

    def test_hybrid_sampling(self, sampler_hybrid, sample_tickets_df):
        """Test hybrid sampling strategy"""
        sample_size = 0.20  # 20%
        sampled_df = sampler_hybrid.sample_tickets(
            sample_tickets_df, sample_size=sample_size, min_tickets=1, max_tickets=50
        )

        assert isinstance(sampled_df, pd.DataFrame)
        assert len(sampled_df) > 0
        assert len(sampled_df) <= len(sample_tickets_df)

    def test_sample_tickets_with_constraints(
        self, sampler_stratified, sample_tickets_df
    ):
        """Test sampling with min/max constraints"""
        sampled_df = sampler_stratified.sample_tickets(
            sample_tickets_df, sample_size=0.05, min_tickets=20, max_tickets=30  # 5%
        )

        assert isinstance(sampled_df, pd.DataFrame)
        assert 20 <= len(sampled_df) <= 30

    def test_calculate_adaptive_constraints(self, sampler_stratified):
        """Test adaptive constraint calculation"""
        # Small dataset
        constraints = sampler_stratified._calculate_adaptive_constraints(1000)
        min_tickets, max_tickets = constraints
        assert min_tickets > 0
        assert max_tickets > min_tickets

        # Medium dataset
        constraints = sampler_stratified._calculate_adaptive_constraints(15000)
        min_tickets, max_tickets = constraints
        assert min_tickets > 0
        assert max_tickets > min_tickets

        # Large dataset
        constraints = sampler_stratified._calculate_adaptive_constraints(50000)
        min_tickets, max_tickets = constraints
        assert min_tickets > 0
        assert max_tickets > min_tickets

    def test_validate_sample_representativeness(
        self, sampler_stratified, sample_tickets_df
    ):
        """Test sample representativeness validation"""
        sampled_df = sampler_stratified.sample_tickets(
            sample_tickets_df, sample_size=0.20
        )

        metrics = sampler_stratified.validate_sample_representativeness(
            sample_tickets_df, sampled_df
        )

        assert isinstance(metrics, dict)
        assert "sample_size" in metrics
        assert "sample_ratio" in metrics
        assert "unique_tickets_ratio" in metrics
        assert metrics["sample_size"] > 0
        assert 0 < metrics["sample_ratio"] <= 1

    def test_sample_tickets_invalid_input(self, sampler_stratified):
        """Test sampling with invalid input"""
        # Invalid DataFrame
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            sampler_stratified.sample_tickets("invalid", sample_size=0.1)

        # Invalid sample size
        with pytest.raises(ValueError, match="sample_size must be between 0 and 1"):
            sampler_stratified.sample_tickets(pd.DataFrame(), sample_size=1.5)

    def test_create_diversity_report_function(
        self, sampler_stratified, sample_tickets_df
    ):
        """Test the create_diversity_report utility function"""
        sampled_df = sampler_stratified.sample_tickets(
            sample_tickets_df, sample_size=0.15
        )

        report = create_diversity_report(
            sampler_stratified, sample_tickets_df, sampled_df
        )

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Sampling Diversity Report" in report
        assert sampler_stratified.strategy in report
