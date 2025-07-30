"""
Tests for PerformanceMonitor module
"""

import pytest
import pandas as pd
import json
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.performance_monitor import (
    PerformanceMonitor,
    CostMetrics,
    PerformanceMetrics,
    AccuracyMetrics,
    ValidationReport,
    create_performance_monitor,
    validate_opçao_d_compliance
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
            discovery_time=60.0,          # 1 minute
            application_time=240.0,       # 4 minutes
            total_cost_usd=1.50,
            cost_per_1k_tickets=1.50,
            avg_confidence=0.89,
            classification_rate=0.94,
            meets_cost_target=False,       # Exceeds 0.20 target
            meets_confidence_target=True   # Above 0.85 target
        )
    
    @pytest.fixture
    def sample_categories_data(self):
        """Sample categories data for testing"""
        return {
            "version": "1.0",
            "categories": [
                {"id": 1, "display_name": "Payment Issues"},
                {"id": 2, "display_name": "Booking Changes"},
                {"id": 3, "display_name": "Technical Issues"},
                {"id": 4, "display_name": "Customer Service"},
                {"id": 5, "display_name": "Product Information"}
            ]
        }
    
    @pytest.fixture
    def sample_results_df(self):
        """Sample classification results for testing"""
        data = {
            'ticket_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'category_ids': ['1', '2', '3,4', '5', ''],
            'category_names': ['Payment Issues', 'Booking Changes', 'Technical Issues,Customer Service', 'Product Information', ''],
            'confidence': [0.95, 0.88, 0.92, 0.85, 0.0],
            'processing_time': [0.12, 0.15, 0.18, 0.11, 0.05],
            'tokens_used': [120, 130, 140, 110, 50]
        }
        return pd.DataFrame(data)
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(
                monitoring_dir=Path(tmp_dir),
                cost_target_per_1k=0.25,
                time_target_minutes=30,
                confidence_target=0.90
            )
            
            assert monitor.monitoring_dir == Path(tmp_dir)
            assert monitor.cost_target_per_1k == 0.25
            assert monitor.time_target_minutes == 30
            assert monitor.confidence_target == 0.90
            assert monitor.current_session is None
            assert len(monitor.metrics_history) == 0
    
    def test_performance_monitor_default_initialization(self):
        """Test PerformanceMonitor with default values"""
        with patch('pathlib.Path.mkdir'):
            monitor = PerformanceMonitor()
            
            assert monitor.cost_target_per_1k == 0.20  # Opção D default
            assert monitor.time_target_minutes == 25
            assert monitor.confidence_target == 0.85
    
    def test_monitoring_session_lifecycle(self):
        """Test monitoring session start and management"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            dataset_info = {'total_tickets': 500, 'source': 'test_data.csv'}
            
            # Start session
            monitor.start_monitoring_session("test_session", dataset_info)
            
            assert monitor.current_session is not None
            assert monitor.current_session['name'] == "test_session"
            assert monitor.current_session['dataset_info'] == dataset_info
            assert 'start_time' in monitor.current_session
            assert 'phase_timings' in monitor.current_session
            assert 'cost_tracking' in monitor.current_session
            assert 'events' in monitor.current_session
    
    def test_phase_recording(self):
        """Test phase start and end recording"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Start session
            monitor.start_monitoring_session("test", {'total_tickets': 100})
            
            # Record phase start
            phase_info = {'sample_rate': 0.15}
            monitor.record_phase_start("discovery", phase_info)
            
            assert "discovery" in monitor.current_session['phase_timings']
            assert 'start_time' in monitor.current_session['phase_timings']['discovery']
            assert monitor.current_session['phase_timings']['discovery']['info'] == phase_info
            
            # Record phase end
            additional_metrics = {'categories_found': 5}
            monitor.record_phase_end("discovery", cost_usd=0.50, additional_metrics=additional_metrics)
            
            discovery_timing = monitor.current_session['phase_timings']['discovery']
            assert 'end_time' in discovery_timing
            assert 'duration' in discovery_timing
            assert discovery_timing['cost_usd'] == 0.50
            assert discovery_timing['metrics'] == additional_metrics
            assert monitor.current_session['cost_tracking']['discovery'] == 0.50
    
    def test_phase_recording_errors(self):
        """Test error handling in phase recording"""
        monitor = PerformanceMonitor()
        
        # Test recording without active session
        with pytest.raises(ValueError, match="No active monitoring session"):
            monitor.record_phase_start("test_phase")
        
        with pytest.raises(ValueError, match="No active monitoring session"):
            monitor.record_phase_end("test_phase")
        
        # Start session and test ending non-existent phase
        monitor.start_monitoring_session("test", {})
        
        with pytest.raises(ValueError, match="Phase nonexistent was not started"):
            monitor.record_phase_end("nonexistent")
    
    def test_cost_metrics_calculation(self, sample_orchestration_metrics):
        """Test cost metrics calculation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir), cost_target_per_1k=0.20)
            
            cost_metrics = monitor._calculate_cost_metrics(sample_orchestration_metrics)
            
            assert isinstance(cost_metrics, CostMetrics)
            assert cost_metrics.total_cost_usd == 1.50
            assert cost_metrics.cost_per_1k_tickets == 1.50
            assert cost_metrics.target_cost_per_1k == 0.20
            assert cost_metrics.meets_cost_target is False  # 1.50 > 0.20
            assert cost_metrics.cost_per_category == 1.50 / 5  # 5 categories
            assert 'discovery_phase' in cost_metrics.cost_breakdown
            assert 'application_phase' in cost_metrics.cost_breakdown
    
    def test_performance_metrics_calculation(self, sample_orchestration_metrics):
        """Test performance metrics calculation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir), time_target_minutes=5.0)
            
            performance_metrics = monitor._calculate_performance_metrics(sample_orchestration_metrics)
            
            assert isinstance(performance_metrics, PerformanceMetrics)
            assert performance_metrics.total_processing_time == 300.0
            assert performance_metrics.discovery_time == 60.0
            assert performance_metrics.application_time == 240.0
            assert performance_metrics.avg_time_per_ticket == 300.0 / 1000  # 1000 tickets
            assert performance_metrics.throughput_tickets_per_second == 1000 / 300.0
            assert performance_metrics.bottleneck_phase == "application"  # 240 > 60
            assert performance_metrics.meets_time_target is False  # 5 min = 300s, so exactly at target
    
    def test_accuracy_metrics_calculation(self, sample_orchestration_metrics, 
                                        sample_categories_data, sample_results_df):
        """Test accuracy metrics calculation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir), confidence_target=0.85)
            
            accuracy_metrics = monitor._calculate_accuracy_metrics(
                sample_results_df, sample_categories_data, sample_orchestration_metrics
            )
            
            assert isinstance(accuracy_metrics, AccuracyMetrics)
            assert accuracy_metrics.avg_confidence == 0.89
            assert accuracy_metrics.classification_rate == 0.94
            assert accuracy_metrics.target_confidence == 0.85
            assert accuracy_metrics.meets_accuracy_target is True  # 0.89 > 0.85
            
            # Check category coverage (4 out of 5 categories used: 1, 2, 3, 4, 5)
            expected_coverage = 5 / 5  # All categories used
            assert abs(accuracy_metrics.category_coverage - expected_coverage) < 0.01
            
            # Check validation errors
            assert isinstance(accuracy_metrics.validation_errors, list)
    
    def test_validation_report_generation(self, sample_orchestration_metrics,
                                        sample_categories_data, sample_results_df):
        """Test complete validation report generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Create test files
            categories_path = Path(tmp_dir) / "categories.json"
            results_path = Path(tmp_dir) / "results.csv"
            
            with open(categories_path, 'w') as f:
                json.dump(sample_categories_data, f)
            
            sample_results_df.to_csv(results_path, index=False)
            
            # Start monitoring session
            monitor.start_monitoring_session("test_validation", {'total_tickets': 1000})
            
            # Generate validation report
            report = monitor.validate_pipeline_results(
                sample_orchestration_metrics, categories_path, results_path
            )
            
            assert isinstance(report, ValidationReport)
            assert report.timestamp is not None
            assert isinstance(report.cost_metrics, CostMetrics)
            assert isinstance(report.performance_metrics, PerformanceMetrics)
            assert isinstance(report.accuracy_metrics, AccuracyMetrics)
            assert isinstance(report.compliance_summary, dict)
            assert isinstance(report.recommendations, list)
            assert isinstance(report.next_actions, list)
            
            # Check compliance summary
            assert 'meets_cost_target' in report.compliance_summary
            assert 'meets_time_target' in report.compliance_summary
            assert 'meets_accuracy_target' in report.compliance_summary
            assert 'overall_compliant' in report.compliance_summary
    
    def test_recommendations_generation(self):
        """Test recommendations generation logic"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Create test metrics that don't meet targets
            cost_metrics = CostMetrics(
                discovery_cost_usd=1.0,
                application_cost_usd=0.5,
                total_cost_usd=1.5,
                cost_per_1k_tickets=1.5,  # Exceeds 0.20 target
                cost_per_category=0.3,
                target_cost_per_1k=0.20,
                meets_cost_target=False,
                cost_breakdown={'discovery_phase': 1.0, 'application_phase': 0.5}
            )
            
            performance_metrics = PerformanceMetrics(
                total_processing_time=1800.0,  # 30 minutes, exceeds 25 min target
                discovery_time=600.0,
                application_time=1200.0,
                avg_time_per_ticket=1.8,
                throughput_tickets_per_second=0.56,
                target_time_minutes=25.0,
                meets_time_target=False,
                bottleneck_phase="application"
            )
            
            accuracy_metrics = AccuracyMetrics(
                avg_confidence=0.80,  # Below 0.85 target
                high_confidence_ratio=0.60,
                classification_rate=0.90,
                category_coverage=0.70,  # Below 0.80
                consistency_score=0.75,  # Below 0.80
                target_confidence=0.85,
                meets_accuracy_target=False,
                validation_errors=["Low confidence"]
            )
            
            recommendations = monitor._generate_recommendations(
                cost_metrics, performance_metrics, accuracy_metrics
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            # Check that recommendations address the issues
            rec_text = ' '.join(recommendations).lower()
            assert 'cost exceeds target' in rec_text
            assert 'processing time exceeds target' in rec_text
            assert 'confidence' in rec_text
            assert 'category coverage' in rec_text
    
    def test_next_actions_generation(self):
        """Test next actions generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Test compliant case
            compliant_summary = {
                'meets_cost_target': True,
                'meets_time_target': True,
                'meets_accuracy_target': True,
                'overall_compliant': True
            }
            
            actions = monitor._generate_next_actions(compliant_summary, [])
            assert any('All Opção D targets met' in action for action in actions)
            assert any('production' in action.lower() for action in actions)
            
            # Test non-compliant case
            non_compliant_summary = {
                'meets_cost_target': False,
                'meets_time_target': False,
                'meets_accuracy_target': False,
                'overall_compliant': False
            }
            
            actions = monitor._generate_next_actions(non_compliant_summary, [])
            assert any('targets not fully met' in action for action in actions)
            assert any('Optimize cost' in action for action in actions)
            assert any('Optimize performance' in action for action in actions)
            assert any('Improve accuracy' in action for action in actions)
    
    def test_ab_test_metrics_calculation(self, sample_results_df):
        """Test A/B test metrics calculation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            metrics = monitor._calculate_ab_test_metrics(sample_results_df, "Test Method")
            
            assert metrics['method_name'] == "Test Method"
            assert metrics['total_tickets'] == 5
            assert abs(metrics['avg_confidence'] - 0.72) < 0.01  # (0.95+0.88+0.92+0.85+0.0)/5
            assert metrics['classification_rate'] == 0.8  # 4 out of 5 have categories
            assert metrics['unique_categories_used'] == 5  # Categories 1,2,3,4,5 used
    
    @patch('scipy.stats.ttest_ind')
    def test_statistical_comparison(self, mock_ttest, sample_results_df):
        """Test statistical comparison between two result sets"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Mock t-test results
            mock_ttest.return_value = (2.5, 0.02)  # Significant difference
            
            # Create two similar DataFrames
            results_a = sample_results_df.copy()
            results_b = sample_results_df.copy()
            results_b['confidence'] = results_b['confidence'] * 0.9  # Slightly lower
            
            comparison = monitor._perform_statistical_comparison(results_a, results_b)
            
            assert 't_statistic' in comparison
            assert 'p_value' in comparison
            assert 'cohens_d' in comparison
            assert comparison['significant_difference'] is True  # p < 0.05
            assert 'effect_size_interpretation' in comparison
    
    def test_effect_size_interpretation(self):
        """Test Cohen's d effect size interpretation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            assert monitor._interpret_effect_size(0.1) == "negligible"
            assert monitor._interpret_effect_size(0.3) == "small"
            assert monitor._interpret_effect_size(0.6) == "medium"
            assert monitor._interpret_effect_size(1.0) == "large"
    
    def test_ab_recommendation_generation(self):
        """Test A/B test recommendation generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            metrics_a = {
                'avg_confidence': 0.90,
                'classification_rate': 0.95
            }
            
            metrics_b = {
                'avg_confidence': 0.85,
                'classification_rate': 0.90
            }
            
            # Test with significant difference
            significant_stats = {'significant_difference': True}
            recommendation = monitor._generate_ab_recommendation(metrics_a, metrics_b, significant_stats)
            assert "Method A" in recommendation
            
            # Test with no significant difference
            non_significant_stats = {'significant_difference': False}
            recommendation = monitor._generate_ab_recommendation(metrics_a, metrics_b, non_significant_stats)
            assert "No statistically significant difference" in recommendation
    
    def test_comparison_against_baseline(self, sample_orchestration_metrics):
        """Test baseline comparison functionality"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PerformanceMonitor(monitoring_dir=Path(tmp_dir))
            
            # Create baseline metrics (worse performance)
            baseline_metrics = type(sample_orchestration_metrics)(
                total_tickets=1000,
                discovery_sample_size=150,
                categories_discovered=5,
                total_processing_time=400.0,  # Worse time
                discovery_time=100.0,
                application_time=300.0,
                total_cost_usd=2.00,  # Higher cost
                cost_per_1k_tickets=2.00,
                avg_confidence=0.82,  # Lower confidence
                classification_rate=0.90,
                meets_cost_target=False,
                meets_confidence_target=False
            )
            
            comparison = monitor.compare_against_baseline(
                sample_orchestration_metrics, baseline_metrics, "Test Comparison"
            )
            
            assert comparison['comparison_name'] == "Test Comparison"
            assert 'improvements' in comparison
            assert 'regressions' in comparison
            assert 'summary' in comparison
            
            # Check that improvements are detected
            assert len(comparison['improvements']) > 0
            assert comparison['summary']['overall_better'] is True
    
    def test_utility_functions(self):
        """Test utility functions"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test create_performance_monitor
            monitor = create_performance_monitor(
                monitoring_dir=tmp_dir,
                cost_target=0.25,
                time_target=30,
                confidence_target=0.90
            )
            
            assert isinstance(monitor, PerformanceMonitor)
            assert monitor.cost_target_per_1k == 0.25
            assert monitor.time_target_minutes == 30
            assert monitor.confidence_target == 0.90
    
    def test_dataclass_creation(self):
        """Test dataclass creation and validation"""
        # Test CostMetrics
        cost_metrics = CostMetrics(
            discovery_cost_usd=0.5,
            application_cost_usd=1.0,
            total_cost_usd=1.5,
            cost_per_1k_tickets=1.5,
            cost_per_category=0.3,
            target_cost_per_1k=0.2,
            meets_cost_target=False,
            cost_breakdown={'discovery': 0.5, 'application': 1.0}
        )
        
        assert cost_metrics.total_cost_usd == 1.5
        assert cost_metrics.meets_cost_target is False
        
        # Test PerformanceMetrics
        perf_metrics = PerformanceMetrics(
            total_processing_time=300.0,
            discovery_time=60.0,
            application_time=240.0,
            avg_time_per_ticket=0.3,
            throughput_tickets_per_second=3.33,
            target_time_minutes=25.0,
            meets_time_target=False,
            bottleneck_phase="application"
        )
        
        assert perf_metrics.bottleneck_phase == "application"
        assert perf_metrics.meets_time_target is False
        
        # Test AccuracyMetrics
        acc_metrics = AccuracyMetrics(
            avg_confidence=0.89,
            high_confidence_ratio=0.85,
            classification_rate=0.94,
            category_coverage=0.80,
            consistency_score=0.85,
            target_confidence=0.85,
            meets_accuracy_target=True,
            validation_errors=[]
        )
        
        assert acc_metrics.avg_confidence == 0.89
        assert acc_metrics.meets_accuracy_target is True
        assert len(acc_metrics.validation_errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])