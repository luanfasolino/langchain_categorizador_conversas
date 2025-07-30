"""
Tests for FastClassifier module
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.fast_classifier import FastClassifier, ClassificationResult, BatchResult, load_and_classify_tickets


class TestFastClassifier:
    """Test cases for FastClassifier class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"
    
    @pytest.fixture
    def sample_categories(self):
        """Sample categories JSON for testing"""
        return {
            "version": "1.0",
            "generated_at": "2024-01-01T10:00:00",
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                    "description": "Falhas em transações e cartões recusados",
                    "keywords": ["cartão", "pagamento", "recusado"],
                    "examples": ["Meu cartão foi recusado", "Erro no pagamento"]
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                    "description": "Mudanças e cancelamentos de reservas",
                    "keywords": ["reserva", "alterar", "cancelar"],
                    "examples": ["Como alterar minha reserva?", "Quero cancelar"]
                },
                {
                    "id": 3,
                    "technical_name": "technical_issues",
                    "display_name": "Problemas Técnicos",
                    "description": "Falhas no site e aplicativo",
                    "keywords": ["site", "carregando", "erro", "app"],
                    "examples": ["Site não carrega", "App travou"]
                }
            ]
        }
    
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
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_classifier_initialization(self, mock_llm, mock_api_key):
        """Test FastClassifier initialization"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(
                api_key=mock_api_key,
                model_name="gemini-2.5-flash",
                temperature=0.1,
                batch_size=50
            )
            
            assert classifier.model_name == "gemini-2.5-flash"
            assert classifier.temperature == 0.1
            assert classifier.batch_size == 50
            assert classifier.max_categories_per_ticket == 3
            assert classifier.confidence_threshold == 0.85
            assert classifier.classify_max_tokens == 32
            assert classifier.target_cost_per_1k == 0.20
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_load_categories(self, mock_llm, mock_api_key, sample_categories):
        """Test loading categories from JSON"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                categories_path = Path(tmp_dir) / "categories.json"
                
                # Save categories to file
                with open(categories_path, 'w') as f:
                    json.dump(sample_categories, f)
                
                # Load categories
                loaded_categories = classifier.load_categories(categories_path)
                
                assert loaded_categories == sample_categories
                assert classifier.categories == sample_categories
                assert classifier.classification_chain is not None
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_prepare_tickets_for_classification(self, mock_llm, mock_api_key, sample_tickets_df):
        """Test ticket preparation for classification"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            prepared = classifier._prepare_tickets_for_classification(sample_tickets_df)
            
            # Should have 3 tickets (T001, T002, T003)
            assert len(prepared) == 3
            
            # Check structure
            for ticket in prepared:
                assert 'ticket_id' in ticket
                assert 'text' in ticket
                assert '[USER]:' in ticket['text']
                assert '[AGENT]:' in ticket['text']
            
            # Check specific ticket
            t001 = next(t for t in prepared if t['ticket_id'] == 'T001')
            assert 'Meu cartão foi recusado na compra' in t001['text']
            assert 'Vou verificar o problema com seu cartão' in t001['text']
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_parse_classification_response(self, mock_llm, mock_api_key):
        """Test parsing of classification responses"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            # Test JSON response
            json_response = '{"categories": [1, 2], "confidence": 0.92}'
            result = classifier._parse_classification_response(json_response)
            assert result['categories'] == [1, 2]
            assert result['confidence'] == 0.92
            
            # Test response with extra text
            mixed_response = 'Analysis: {"categories": [3], "confidence": 0.88} - Done'
            result = classifier._parse_classification_response(mixed_response)
            assert result['categories'] == [3]
            assert result['confidence'] == 0.88
            
            # Test fallback parsing
            text_response = 'Categories: 1 and 2'
            result = classifier._parse_classification_response(text_response)
            assert 1 in result['categories']
            assert 2 in result['categories']
            assert result['confidence'] == 0.7
            
            # Test invalid response
            invalid_response = 'No valid data'
            result = classifier._parse_classification_response(invalid_response)
            assert result['categories'] == []
            assert result['confidence'] == 0.0
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_get_category_name(self, mock_llm, mock_api_key, sample_categories):
        """Test getting category names by ID"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            classifier.categories = sample_categories
            
            # Test valid category IDs
            assert classifier._get_category_name(1) == "Problemas de Pagamento"
            assert classifier._get_category_name(2) == "Alterações de Reserva"
            assert classifier._get_category_name(3) == "Problemas Técnicos"
            
            # Test invalid category ID
            assert classifier._get_category_name(999) is None
            
            # Test with no categories loaded
            classifier.categories = None
            assert classifier._get_category_name(1) is None
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_estimate_cost(self, mock_llm, mock_api_key):
        """Test cost estimation"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            # Test cost calculation
            tokens = 1000000  # 1M tokens
            cost = classifier._estimate_cost(tokens)
            
            # Should be approximately $75 for 1M tokens
            assert 70 <= cost <= 80
            
            # Test smaller amount
            tokens = 10000  # 10K tokens
            cost = classifier._estimate_cost(tokens)
            assert 0.7 <= cost <= 0.8
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_classification_result_dataclass(self, mock_llm, mock_api_key):
        """Test ClassificationResult dataclass"""
        result = ClassificationResult(
            ticket_id="T001",
            categories=[1, 2],
            confidence=0.95,
            processing_time=1.5,
            tokens_used=150
        )
        
        assert result.ticket_id == "T001"
        assert result.categories == [1, 2]
        assert result.confidence == 0.95
        assert result.processing_time == 1.5
        assert result.tokens_used == 150
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_batch_result_dataclass(self, mock_llm, mock_api_key):
        """Test BatchResult dataclass"""
        result1 = ClassificationResult("T001", [1], 0.9, 1.0, 100)
        result2 = ClassificationResult("T002", [2], 0.8, 1.2, 120)
        
        batch_result = BatchResult(
            batch_id=0,
            results=[result1, result2],
            total_tokens=220,
            total_time=2.5,
            success_rate=1.0
        )
        
        assert batch_result.batch_id == 0
        assert len(batch_result.results) == 2
        assert batch_result.total_tokens == 220
        assert batch_result.total_time == 2.5
        assert batch_result.success_rate == 1.0
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_get_classification_stats(self, mock_llm, mock_api_key):
        """Test classification statistics calculation"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            # Create sample results DataFrame
            results_data = {
                'ticket_id': ['T001', 'T002', 'T003'],
                'category_ids': ['1,2', '3', ''],
                'category_names': ['Payment,Booking', 'Technical', ''],
                'confidence': [0.95, 0.88, 0.0],
                'processing_time': [1.0, 1.2, 0.5],
                'tokens_used': [100, 120, 50]
            }
            results_df = pd.DataFrame(results_data)
            
            stats = classifier.get_classification_stats(results_df)
            
            # Check basic stats
            assert stats['total_tickets'] == 3
            assert stats['classified_tickets'] == 2  # T001 and T002 have categories
            assert stats['classification_rate'] == 2/3
            
            # Check confidence stats
            assert stats['avg_confidence'] == pytest.approx((0.95 + 0.88 + 0.0) / 3)
            assert stats['high_confidence_count'] == 1  # Only T001 >= 0.85
            
            # Check category distribution
            assert 1 in stats['category_distribution']
            assert 2 in stats['category_distribution']
            assert 3 in stats['category_distribution']
            
            # Check processing stats
            assert stats['total_processing_time'] == 2.7
            assert stats['total_tokens_used'] == 270
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_empty_dataframe_handling(self, mock_llm, mock_api_key):
        """Test handling of empty DataFrames"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            empty_df = pd.DataFrame()
            
            # Should raise error for empty classification
            with pytest.raises(ValueError, match="Cannot classify empty dataset"):
                classifier.classify_all_tickets(empty_df)
            
            # Should return empty stats for empty results
            stats = classifier.get_classification_stats(empty_df)
            assert stats == {}
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    def test_categories_not_loaded_error(self, mock_llm, mock_api_key, sample_tickets_df):
        """Test error when categories not loaded"""
        with patch('src.discovery.fast_classifier.Path'):
            classifier = FastClassifier(api_key=mock_api_key)
            
            # Should raise error when trying to classify without categories
            with pytest.raises(ValueError, match="Categories must be loaded before classification"):
                classifier.classify_all_tickets(sample_tickets_df)
            
            # Should raise error when trying to setup chain without categories
            with pytest.raises(ValueError, match="Categories must be loaded before setting up classification chain"):
                classifier._setup_classification_chain()
    
    @patch('src.discovery.fast_classifier.ChatGoogleGenerativeAI')
    @patch('src.discovery.fast_classifier.pd.read_csv')
    def test_load_and_classify_tickets_utility(self, mock_read_csv, mock_llm, 
                                             sample_tickets_df, sample_categories):
        """Test utility function for loading and classifying tickets"""
        # Mock file reading
        mock_read_csv.return_value = sample_tickets_df
        
        with patch('src.discovery.fast_classifier.Path'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                categories_path = Path(tmp_dir) / "categories.json"
                tickets_path = Path(tmp_dir) / "tickets.csv"
                output_path = Path(tmp_dir) / "results.csv"
                
                # Create categories file
                with open(categories_path, 'w') as f:
                    json.dump(sample_categories, f)
                
                # Mock the classification process
                with patch.object(FastClassifier, 'classify_all_tickets') as mock_classify:
                    mock_results = pd.DataFrame({'ticket_id': ['T001'], 'category_ids': ['1']})
                    mock_classify.return_value = mock_results
                    
                    result = load_and_classify_tickets(
                        categories_path=categories_path,
                        tickets_path=tickets_path,
                        output_path=output_path,
                        api_key="test_key"
                    )
                    
                    # Verify the utility function works
                    assert isinstance(result, pd.DataFrame)
                    mock_read_csv.assert_called_once()
                    mock_classify.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])