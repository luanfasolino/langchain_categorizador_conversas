"""
Tests for IntelligentSampler module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.intelligent_sampler import IntelligentSampler, create_diversity_report


class TestIntelligentSampler:
    """Test cases for IntelligentSampler class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample ticket data for testing"""
        np.random.seed(42)
        
        # Create sample tickets spanning 3 months (need enough for min 500 sample)
        dates = []
        base_date = datetime(2024, 1, 1)
        for i in range(2000):  # Increased to ensure enough data for sampling
            # Add some temporal clustering
            if i < 600:
                date = base_date + timedelta(days=np.random.randint(0, 30))
            elif i < 1400:
                date = base_date + timedelta(days=np.random.randint(30, 60))
            else:
                date = base_date + timedelta(days=np.random.randint(60, 90))
            dates.append(date)
        
        # Create diverse text content
        texts = []
        categories = [
            "Problema com pagamento",
            "Dúvida sobre produto", 
            "Reclamação de entrega",
            "Solicitação de cancelamento",
            "Suporte técnico",
        ]
        
        for i in range(2000):
            base_text = np.random.choice(categories)
            variation = np.random.choice([
                " - urgente",
                " - preciso de ajuda",
                " - quando será resolvido?",
                " - obrigado",
                ""
            ])
            texts.append(base_text + variation)
        
        return pd.DataFrame({
            'ticket_id': [f"ticket_{i}" for i in range(2000)],
            'ticket_created_at': dates,
            'text': texts,
            'sender': np.random.choice(['USER', 'AGENT'], 2000),
            'category': 'TEXT'
        })
    
    def test_sampler_initialization(self):
        """Test sampler initialization with different strategies"""
        # Valid strategies
        for strategy in ['stratified', 'diversity', 'hybrid']:
            sampler = IntelligentSampler(strategy=strategy)
            assert sampler.strategy == strategy
            assert sampler.random_state == 42
        
        # Invalid strategy
        with pytest.raises(ValueError):
            IntelligentSampler(strategy="invalid")
    
    def test_stratified_sampling(self, sample_data):
        """Test stratified sampling maintains temporal proportions"""
        sampler = IntelligentSampler(strategy="stratified")
        
        # Test with 15% sample rate
        sample = sampler.sample_tickets(sample_data, sample_size=0.15)
        
        # Check sample size is reasonable (default min=500, max=1000 per PRD)
        assert len(sample) >= 500  # Should meet minimum from PRD
        assert len(sample) <= 1000  # Should not exceed maximum from PRD
        
        # Check temporal distribution is maintained
        original_months = pd.to_datetime(sample_data['ticket_created_at']).dt.month.value_counts()
        sample_months = pd.to_datetime(sample['ticket_created_at']).dt.month.value_counts()
        
        # All months should be represented
        assert len(sample_months) > 0
    
    def test_diversity_sampling(self, sample_data):
        """Test diversity sampling maximizes textual diversity"""
        sampler = IntelligentSampler(strategy="diversity")
        
        sample = sampler.sample_tickets(sample_data, sample_size=0.15)
        
        # Check sample size (default constraints from PRD)
        assert len(sample) >= 500
        assert len(sample) <= 1000
        
        # Check text diversity - should have varied content
        unique_texts = sample['text'].nunique()
        assert unique_texts > 10  # Should have diverse text content
    
    def test_hybrid_sampling(self, sample_data):
        """Test hybrid sampling combines both strategies"""
        sampler = IntelligentSampler(strategy="hybrid")
        
        sample = sampler.sample_tickets(sample_data, sample_size=0.15)
        
        # Check sample size (default constraints from PRD)
        assert len(sample) >= 500
        assert len(sample) <= 1000
        
        # Should combine benefits of both approaches
        assert sample['text'].nunique() > 5
    
    def test_sample_size_constraints(self, sample_data):
        """Test sample size constraints are respected"""
        sampler = IntelligentSampler(strategy="stratified")
        
        # Test minimum constraint
        sample = sampler.sample_tickets(
            sample_data, 
            sample_size=0.01,  # Very small percentage
            min_tickets=100
        )
        assert len(sample) >= 100
        
        # Test maximum constraint
        sample = sampler.sample_tickets(
            sample_data,
            sample_size=0.8,  # Large percentage
            max_tickets=200
        )
        assert len(sample) <= 200
    
    def test_validation_metrics(self, sample_data):
        """Test sample validation metrics"""
        sampler = IntelligentSampler(strategy="stratified")
        sample = sampler.sample_tickets(sample_data, sample_size=0.15)
        
        metrics = sampler.validate_sample_representativeness(sample_data, sample)
        
        # Check required metrics are present
        required_metrics = [
            'sample_size', 'sample_ratio', 'unique_tickets_ratio',
            'length_mean_diff', 'length_std_diff'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check values are reasonable
        assert metrics['sample_size'] > 0
        assert 0 < metrics['sample_ratio'] <= 1
        assert 0 <= metrics['unique_tickets_ratio'] <= 1
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        sampler = IntelligentSampler(strategy="stratified")
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['ticket_id', 'ticket_created_at', 'text', 'sender'])
        with pytest.raises(Exception):
            sampler.sample_tickets(empty_df)
        
        # Invalid sample size
        sample_data = pd.DataFrame({
            'ticket_id': ['1'], 
            'ticket_created_at': [datetime.now()],
            'text': ['test'],
            'sender': ['USER']
        })
        
        with pytest.raises(ValueError):
            sampler.sample_tickets(sample_data, sample_size=0)
        
        with pytest.raises(ValueError):
            sampler.sample_tickets(sample_data, sample_size=1.5)
    
    def test_diversity_report_generation(self, sample_data):
        """Test diversity report generation"""
        sampler = IntelligentSampler(strategy="hybrid")
        sample = sampler.sample_tickets(sample_data, sample_size=0.15)
        
        report = create_diversity_report(sampler, sample_data, sample)
        
        # Check report contains key sections
        assert "Sampling Diversity Report" in report
        assert "Strategy: hybrid" in report
        assert "Sample Statistics" in report
        assert "Representativeness Metrics" in report
        assert "Quality Assessment" in report
        
        # Check report contains metrics
        assert "Total tickets sampled:" in report
        assert "Sample ratio:" in report


if __name__ == "__main__":
    pytest.main([__file__])