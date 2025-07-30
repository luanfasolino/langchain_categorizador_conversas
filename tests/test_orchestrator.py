"""
Tests for TwoPhaseOrchestrator module
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.orchestrator import (
    TwoPhaseOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMetrics,
    OrchestrationError,
    run_complete_pipeline
)


class TestTwoPhaseOrchestrator:
    """Test cases for TwoPhaseOrchestrator class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"
    
    @pytest.fixture
    def sample_tickets_df(self):
        """Create sample tickets DataFrame for testing"""
        data = {
            'ticket_id': ['T001', 'T001', 'T002', 'T002', 'T003', 'T003'],
            'sender': ['USER', 'AGENT', 'USER', 'AGENT', 'USER', 'AGENT'],
            'text': [
                'Meu cartão foi recusado na compra',
                'Vou verificar o problema com seu cartão',
                'Como alterar minha reserva?',
                'Posso ajudar com a alteração',
                'Site não está carregando',
                'Vamos verificar o problema técnico'
            ],
            'message_sended_at': [
                '2024-01-01 10:00:00', '2024-01-01 10:05:00',
                '2024-01-02 11:00:00', '2024-01-02 11:05:00',
                '2024-01-03 12:00:00', '2024-01-03 12:05:00'
            ],
            'category': ['TEXT'] * 6
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_config(self):
        """Sample orchestration configuration"""
        return OrchestrationConfig(
            sample_rate=0.2,
            sampling_strategy="stratified",
            batch_size=50,
            max_workers=2,
            cost_target_per_1k=0.20,
            confidence_threshold=0.85
        )
    
    @pytest.fixture
    def mock_categories(self):
        """Mock discovered categories"""
        return {
            "version": "1.0",
            "generated_at": "2024-01-01T10:00:00",
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                    "description": "Falhas em transações e cartões",
                    "keywords": ["cartão", "pagamento", "recusado"],
                    "examples": ["Meu cartão foi recusado"]
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                    "description": "Mudanças e cancelamentos",
                    "keywords": ["reserva", "alterar", "cancelar"],
                    "examples": ["Como alterar minha reserva?"]
                }
            ]
        }
    
    @patch('src.discovery.orchestrator.Path')
    def test_orchestrator_initialization(self, mock_path, mock_api_key, sample_config):
        """Test orchestrator initialization"""
        orchestrator = TwoPhaseOrchestrator(
            api_key=mock_api_key,
            config=sample_config
        )
        
        assert orchestrator.config == sample_config
        assert orchestrator.sampler is None  # Lazy loading
        assert orchestrator.discoverer is None
        assert orchestrator.classifier is None
        assert orchestrator.start_time is None
        assert orchestrator.cost_tracker['total_tokens'] == 0
    
    @patch('src.discovery.orchestrator.Path')
    def test_orchestrator_default_config(self, mock_path, mock_api_key):
        """Test orchestrator with default configuration"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        assert isinstance(orchestrator.config, OrchestrationConfig)
        assert orchestrator.config.sample_rate == 0.15
        assert orchestrator.config.sampling_strategy == "hybrid"
        assert orchestrator.config.batch_size == 100
        assert orchestrator.config.max_workers == 4
    
    @patch('src.discovery.orchestrator.Path')
    def test_load_and_validate_input_success(self, mock_path, mock_api_key, sample_tickets_df):
        """Test successful input loading and validation"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_tickets_df.to_csv(tmp_file.name, sep=';', index=False)
            tmp_path = Path(tmp_file.name)
            
            try:
                result_df = orchestrator._load_and_validate_input(tmp_path)
                
                assert len(result_df) == 6  # All tickets are TEXT category
                assert 'ticket_id' in result_df.columns
                assert 'text' in result_df.columns
                assert 'sender' in result_df.columns
                assert 'category' in result_df.columns
                
            finally:
                tmp_path.unlink()  # Clean up
    
    @patch('src.discovery.orchestrator.Path')
    def test_load_and_validate_input_missing_columns(self, mock_path, mock_api_key):
        """Test input validation with missing columns"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        # Create DataFrame with missing required columns
        invalid_df = pd.DataFrame({
            'ticket_id': ['T001'],
            'text': ['Some text']
            # Missing 'sender' and 'category' columns
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            invalid_df.to_csv(tmp_file.name, sep=';', index=False)
            tmp_path = Path(tmp_file.name)
            
            try:
                with pytest.raises(OrchestrationError, match="Missing required columns"):
                    orchestrator._load_and_validate_input(tmp_path)
            finally:
                tmp_path.unlink()
    
    @patch('src.discovery.orchestrator.Path')
    def test_load_and_validate_input_no_text_tickets(self, mock_path, mock_api_key):
        """Test input validation with no TEXT tickets"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        # Create DataFrame with no TEXT category tickets
        no_text_df = pd.DataFrame({
            'ticket_id': ['T001'],
            'text': ['Some text'],
            'sender': ['USER'],
            'category': ['OTHER']  # Not TEXT
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            no_text_df.to_csv(tmp_file.name, sep=';', index=False)
            tmp_path = Path(tmp_file.name)
            
            try:
                with pytest.raises(OrchestrationError, match="No tickets found with category='TEXT'"):
                    orchestrator._load_and_validate_input(tmp_path)
            finally:
                tmp_path.unlink()
    
    @patch('src.discovery.orchestrator.IntelligentSampler')
    @patch('src.discovery.orchestrator.CategoryDiscoverer')
    @patch('src.discovery.orchestrator.Path')
    def test_execute_discovery_phase_success(self, mock_path, mock_discoverer_class, 
                                           mock_sampler_class, mock_api_key, 
                                           sample_tickets_df, mock_categories):
        """Test successful discovery phase execution"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Mock sampler
            mock_sampler = Mock()
            mock_sampler.sample_tickets.return_value = sample_tickets_df.iloc[:2]  # Sample subset
            mock_sampler_class.return_value = mock_sampler
            
            # Mock discoverer
            mock_discoverer = Mock()
            mock_discoverer.discover_categories.return_value = mock_categories
            mock_discoverer_class.return_value = mock_discoverer
            
            # Execute discovery phase
            categories_path = orchestrator._execute_discovery_phase(
                sample_tickets_df, output_dir, force_rediscovery=False
            )
            
            # Verify results
            assert categories_path == output_dir / orchestrator.config.categories_filename
            assert 'sample_size' in orchestrator.discovery_metrics
            assert 'categories_discovered' in orchestrator.discovery_metrics
            assert 'processing_time' in orchestrator.discovery_metrics
            
            # Verify method calls
            mock_sampler.sample_tickets.assert_called_once()
            mock_discoverer.discover_categories.assert_called_once()
    
    @patch('src.discovery.orchestrator.FastClassifier')
    @patch('src.discovery.orchestrator.Path')
    def test_execute_application_phase_success(self, mock_path, mock_classifier_class,
                                             mock_api_key, sample_tickets_df, mock_categories):
        """Test successful application phase execution"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            categories_path = output_dir / "categories.json"
            
            # Create categories file
            with open(categories_path, 'w') as f:
                json.dump(mock_categories, f)
            
            # Mock classifier
            mock_classifier = Mock()
            mock_results_df = pd.DataFrame({
                'ticket_id': ['T001', 'T002', 'T003'],
                'category_ids': ['1', '2', '1'],
                'category_names': ['Payment', 'Booking', 'Payment'],
                'confidence': [0.95, 0.88, 0.92]
            })
            mock_classifier.classify_all_tickets.return_value = mock_results_df
            mock_classifier.get_classification_stats.return_value = {
                'total_tickets': 3,
                'avg_confidence': 0.92,
                'classification_rate': 1.0
            }
            mock_classifier.load_categories.return_value = mock_categories
            mock_classifier_class.return_value = mock_classifier
            
            # Execute application phase
            results_path = orchestrator._execute_application_phase(
                sample_tickets_df, categories_path, output_dir, force_reclassification=False
            )
            
            # Verify results
            assert results_path == output_dir / orchestrator.config.results_filename
            assert 'total_tickets' in orchestrator.application_metrics
            assert 'classified_tickets' in orchestrator.application_metrics
            assert 'processing_time' in orchestrator.application_metrics
            
            # Verify method calls
            mock_classifier.load_categories.assert_called_once_with(categories_path)
            mock_classifier.classify_all_tickets.assert_called_once()
    
    @patch('src.discovery.orchestrator.Path')
    def test_generate_final_metrics(self, mock_path, mock_api_key, sample_tickets_df):
        """Test final metrics generation"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        orchestrator.start_time = 1000.0  # Mock start time
        
        # Set up test data
        orchestrator.discovery_metrics = {
            'sample_size': 2,
            'processing_time': 5.0,
            'cost_estimate': 0.10
        }
        orchestrator.application_metrics = {
            'processing_time': 10.0,
            'cost_estimate': 0.15
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Create mock categories file
            categories_path = output_dir / "categories.json"
            mock_categories = {
                "categories": [
                    {"id": 1, "name": "Payment"},
                    {"id": 2, "name": "Booking"}
                ]
            }
            with open(categories_path, 'w') as f:
                json.dump(mock_categories, f)
            
            # Create mock results file
            results_path = output_dir / "results.csv"
            results_data = {
                'ticket_id': ['T001', 'T002', 'T003'],
                'category_ids': ['1', '2', ''],
                'confidence': [0.95, 0.88, 0.0]
            }
            pd.DataFrame(results_data).to_csv(results_path, index=False)
            
            # Mock time.time() to return a fixed value
            with patch('time.time', return_value=1020.0):  # 20 seconds total
                metrics = orchestrator._generate_final_metrics(
                    sample_tickets_df, categories_path, results_path, output_dir
                )
            
            # Verify metrics
            assert isinstance(metrics, OrchestrationMetrics)
            assert metrics.total_tickets == len(sample_tickets_df)
            assert metrics.discovery_sample_size == 2
            assert metrics.categories_discovered == 2
            assert metrics.total_processing_time == 20.0
            assert metrics.total_cost_usd == 0.25  # 0.10 + 0.15
            assert metrics.cost_per_1k_tickets > 0
            assert 0 <= metrics.avg_confidence <= 1
            assert 0 <= metrics.classification_rate <= 1
    
    @patch('src.discovery.orchestrator.Path')
    def test_save_orchestration_metrics(self, mock_path, mock_api_key):
        """Test saving orchestration metrics"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        # Create test metrics
        metrics = OrchestrationMetrics(
            total_tickets=1000,
            discovery_sample_size=150,
            categories_discovered=5,
            total_processing_time=300.0,
            discovery_time=60.0,
            application_time=240.0,
            total_cost_usd=2.50,
            cost_per_1k_tickets=2.50,
            avg_confidence=0.92,
            classification_rate=0.95,
            meets_cost_target=False,
            meets_confidence_target=True
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            orchestrator._save_orchestration_metrics(metrics, output_dir)
            
            # Verify file was created
            metrics_path = output_dir / orchestrator.config.metrics_filename
            assert metrics_path.exists()
            
            # Verify content
            with open(metrics_path, 'r') as f:
                saved_metrics = json.load(f)
            
            assert 'orchestration_summary' in saved_metrics
            assert 'phase_breakdown' in saved_metrics
            assert 'target_compliance' in saved_metrics
            assert saved_metrics['orchestration_summary']['total_tickets'] == 1000
            assert saved_metrics['target_compliance']['meets_cost_target'] is False
            assert saved_metrics['target_compliance']['meets_confidence_target'] is True
    
    @patch('src.discovery.orchestrator.Path')
    def test_cost_estimation_methods(self, mock_path, mock_api_key):
        """Test cost estimation methods"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        # Test discovery cost estimation
        discovery_cost = orchestrator._estimate_discovery_cost(1000)
        assert discovery_cost > 0
        assert isinstance(discovery_cost, float)
        
        # Test application cost estimation
        application_cost = orchestrator._estimate_application_cost(1000)
        assert application_cost > 0
        assert isinstance(application_cost, float)
        
        # Verify discovery is typically more expensive per ticket
        discovery_per_ticket = orchestrator._estimate_discovery_cost(100) / 100
        application_per_ticket = orchestrator._estimate_application_cost(100) / 100
        assert discovery_per_ticket > application_per_ticket
    
    def test_orchestration_config_dataclass(self):
        """Test OrchestrationConfig dataclass"""
        config = OrchestrationConfig()
        
        # Test default values
        assert config.sample_rate == 0.15
        assert config.sampling_strategy == "hybrid"
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.cost_target_per_1k == 0.20
        assert config.confidence_threshold == 0.85
        
        # Test custom values
        custom_config = OrchestrationConfig(
            sample_rate=0.25,
            sampling_strategy="stratified",
            batch_size=200
        )
        assert custom_config.sample_rate == 0.25
        assert custom_config.sampling_strategy == "stratified"
        assert custom_config.batch_size == 200
        assert custom_config.max_workers == 4  # Default preserved
    
    def test_orchestration_metrics_dataclass(self):
        """Test OrchestrationMetrics dataclass"""
        metrics = OrchestrationMetrics(
            total_tickets=1000,
            discovery_sample_size=150,
            categories_discovered=8,
            total_processing_time=180.5,
            discovery_time=45.2,
            application_time=135.3,
            total_cost_usd=1.75,
            cost_per_1k_tickets=1.75,
            avg_confidence=0.89,
            classification_rate=0.94,
            meets_cost_target=True,
            meets_confidence_target=True
        )
        
        assert metrics.total_tickets == 1000
        assert metrics.discovery_sample_size == 150
        assert metrics.categories_discovered == 8
        assert abs(metrics.total_processing_time - 180.5) < 0.01
        assert metrics.meets_cost_target is True
        assert metrics.meets_confidence_target is True
    
    @patch('src.discovery.orchestrator.TwoPhaseOrchestrator')
    def test_run_complete_pipeline_utility(self, mock_orchestrator_class):
        """Test run_complete_pipeline utility function"""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_metrics = OrchestrationMetrics(
            total_tickets=100, discovery_sample_size=15, categories_discovered=3,
            total_processing_time=30.0, discovery_time=10.0, application_time=20.0,
            total_cost_usd=0.50, cost_per_1k_tickets=5.0, avg_confidence=0.90,
            classification_rate=0.95, meets_cost_target=False, meets_confidence_target=True
        )
        mock_orchestrator.execute_complete_pipeline.return_value = mock_metrics
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Call utility function
        result = run_complete_pipeline(
            input_file="test_input.csv",
            output_dir="test_output",
            api_key="test_key",
            config={'sample_rate': 0.20, 'batch_size': 150}
        )
        
        # Verify results
        assert result == mock_metrics
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator.execute_complete_pipeline.assert_called_once()
    
    @patch('src.discovery.orchestrator.Path')
    def test_error_handling_in_discovery_phase(self, mock_path, mock_api_key, sample_tickets_df):
        """Test error handling in discovery phase"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Mock sampler to raise exception
            with patch('src.discovery.orchestrator.IntelligentSampler') as mock_sampler_class:
                mock_sampler_class.side_effect = Exception("Sampler failed")
                
                with pytest.raises(OrchestrationError, match="Discovery phase failed"):
                    orchestrator._execute_discovery_phase(
                        sample_tickets_df, output_dir, force_rediscovery=True
                    )
    
    @patch('src.discovery.orchestrator.Path')
    def test_error_handling_in_application_phase(self, mock_path, mock_api_key, 
                                                sample_tickets_df, mock_categories):
        """Test error handling in application phase"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            categories_path = output_dir / "categories.json"
            
            # Create categories file
            with open(categories_path, 'w') as f:
                json.dump(mock_categories, f)
            
            # Mock classifier to raise exception
            with patch('src.discovery.orchestrator.FastClassifier') as mock_classifier_class:
                mock_classifier_class.side_effect = Exception("Classifier failed")
                
                with pytest.raises(OrchestrationError, match="Application phase failed"):
                    orchestrator._execute_application_phase(
                        sample_tickets_df, categories_path, output_dir, force_reclassification=True
                    )
    
    @patch('src.discovery.orchestrator.Path')
    def test_skip_existing_categories(self, mock_path, mock_api_key, sample_tickets_df):
        """Test skipping discovery when categories already exist"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            categories_path = output_dir / orchestrator.config.categories_filename
            
            # Create existing categories file
            with open(categories_path, 'w') as f:
                json.dump({"categories": []}, f)
            
            # Execute discovery phase without forcing rediscovery
            result_path = orchestrator._execute_discovery_phase(
                sample_tickets_df, output_dir, force_rediscovery=False
            )
            
            # Verify it was skipped
            assert result_path == categories_path
            assert orchestrator.discovery_metrics['skipped'] is True
            assert orchestrator.discovery_metrics['reason'] == 'categories_exist'
    
    @patch('src.discovery.orchestrator.Path')
    def test_skip_existing_results(self, mock_path, mock_api_key, sample_tickets_df, mock_categories):
        """Test skipping application when results already exist"""
        orchestrator = TwoPhaseOrchestrator(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            categories_path = output_dir / "categories.json"
            results_path = output_dir / orchestrator.config.results_filename
            
            # Create categories file
            with open(categories_path, 'w') as f:
                json.dump(mock_categories, f)
            
            # Create existing results file
            with open(results_path, 'w') as f:
                f.write("existing,results")
            
            # Execute application phase without forcing reclassification
            result_path = orchestrator._execute_application_phase(
                sample_tickets_df, categories_path, output_dir, force_reclassification=False
            )
            
            # Verify it was skipped
            assert result_path == results_path
            assert orchestrator.application_metrics['skipped'] is True
            assert orchestrator.application_metrics['reason'] == 'results_exist'


if __name__ == "__main__":
    pytest.main([__file__])