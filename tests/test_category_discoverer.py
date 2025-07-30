"""
Tests for CategoryDiscoverer module
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

# Mock the path operations globally for tests
with patch('src.discovery.category_discoverer.Path'):
    from discovery.category_discoverer import CategoryDiscoverer, validate_categories_schema


class TestCategoryDiscoverer:
    """Test cases for CategoryDiscoverer class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"
    
    @pytest.fixture
    def sample_tickets_df(self):
        """Create sample tickets DataFrame for testing"""
        data = {
            'ticket_id': ['T001', 'T001', 'T001', 'T002', 'T002', 'T003', 'T003'],
            'sender': ['USER', 'AGENT', 'USER', 'USER', 'AGENT', 'USER', 'AGENT'],
            'text': [
                'Meu cartão foi recusado na compra',
                'Vou verificar o problema com seu cartão',
                'Obrigado pela ajuda',
                'Como alterar minha reserva?',
                'Posso ajudar com a alteração',
                'Site não está carregando',
                'Vamos verificar o problema técnico'
            ],
            'message_sended_at': [
                '2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 10:10:00',
                '2024-01-02 11:00:00', '2024-01-02 11:05:00',
                '2024-01-03 12:00:00', '2024-01-03 12:05:00'
            ],
            'category': ['TEXT'] * 7
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_categories_response(self):
        """Sample categories JSON response"""
        return {
            "version": "1.0",
            "generated_at": "2024-01-01T10:00:00",
            "discovery_stats": {
                "total_patterns_analyzed": 10,
                "categories_created": 3,
                "confidence_level": 0.85
            },
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                    "description": "Falhas em transações e cartões recusados",
                    "keywords": ["cartão", "pagamento", "recusado"],
                    "examples": ["Meu cartão foi recusado"],
                    "subcategories": []
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                    "description": "Mudanças e cancelamentos de reservas",
                    "keywords": ["reserva", "alterar", "cancelar"],
                    "examples": ["Como alterar minha reserva?"],
                    "subcategories": []
                },
                {
                    "id": 3,
                    "technical_name": "technical_issues",
                    "display_name": "Problemas Técnicos",
                    "description": "Falhas no site e aplicativo",
                    "keywords": ["site", "carregando", "erro"],
                    "examples": ["Site não está carregando"],
                    "subcategories": []
                }
            ],
            "metadata": {
                "llm_model": "gemini-2.5-flash",
                "discovery_method": "map_reduce_pattern_analysis",
                "chunk_size": 800000,
                "overlap_tokens": 240000
            }
        }
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_discoverer_initialization(self, mock_llm, mock_api_key):
        """Test CategoryDiscoverer initialization"""
        with patch('src.discovery.category_discoverer.Path') as mock_path:
            mock_path.cwd.return_value = Path("/tmp")
            
            discoverer = CategoryDiscoverer(
                api_key=mock_api_key,
                model_name="gemini-2.5-flash",
                temperature=0.1
            )
            
            assert discoverer.model_name == "gemini-2.5-flash"
            assert discoverer.temperature == 0.1
            assert discoverer.chunk_size == 800_000
            assert discoverer.overlap == 240_000
            assert discoverer.min_categories == 5
            assert discoverer.max_categories == 25
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_prepare_tickets_text(self, mock_llm, mock_api_key, sample_tickets_df):
        """Test tickets text preparation"""
        with patch('src.discovery.category_discoverer.Path'):
            discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        result = discoverer._prepare_tickets_text(sample_tickets_df)
        
        # Check that all tickets are included
        assert "TICKET T001" in result
        assert "TICKET T002" in result
        assert "TICKET T003" in result
        
        # Check message format
        assert "[USER]: Meu cartão foi recusado na compra" in result
        assert "[AGENT]: Vou verificar o problema com seu cartão" in result
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_create_discovery_chunks(self, mock_llm, mock_api_key):
        """Test chunk creation for discovery"""
        with patch('src.discovery.category_discoverer.Path'):
            discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        # Create test text
        test_text = "A" * 1000000  # 1M characters should create multiple chunks
        
        chunks = discoverer._create_discovery_chunks(test_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_validate_and_enhance_categories(self, mock_llm, mock_api_key, 
                                           sample_tickets_df, sample_categories_response):
        """Test categories validation and enhancement"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        enhanced = discoverer._validate_and_enhance_categories(
            sample_categories_response, 
            sample_tickets_df
        )
        
        # Check enhancement
        assert "total_tickets_analyzed" in enhanced["discovery_stats"]
        assert "unique_tickets" in enhanced["discovery_stats"]
        assert "avg_keywords_per_category" in enhanced["discovery_stats"]
        
        # Verify values
        assert enhanced["discovery_stats"]["total_tickets_analyzed"] == len(sample_tickets_df)
        assert enhanced["discovery_stats"]["unique_tickets"] == 3  # T001, T002, T003
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_save_and_load_categories(self, mock_llm, mock_api_key, sample_categories_response):
        """Test saving and loading categories JSON"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_categories.json"
            
            # Test save
            discoverer._save_categories_json(sample_categories_response, tmp_path)
            assert tmp_path.exists()
            
            # Test load
            loaded = discoverer.load_categories(tmp_path)
            assert loaded == sample_categories_response
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_get_discovery_stats(self, mock_llm, mock_api_key, sample_categories_response):
        """Test discovery statistics calculation"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        stats = discoverer.get_discovery_stats(sample_categories_response)
        
        # Check required stats
        assert "total_categories" in stats
        assert "categories_with_subcategories" in stats
        assert "total_subcategories" in stats
        assert "avg_keywords_per_category" in stats
        assert "categories_by_complexity" in stats
        
        # Verify values
        assert stats["total_categories"] == 3
        assert stats["categories_with_subcategories"] == 0  # No subcategories in sample
        assert stats["total_subcategories"] == 0
    
    def test_extract_json_from_response(self):
        """Test JSON extraction from LLM response"""
        # Mock discoverer without API calls
        with patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI'):
            discoverer = CategoryDiscoverer(api_key="test")
            
            # Test clean JSON
            clean_json = '{"test": "value"}'
            result = discoverer._extract_json_from_response(clean_json)
            assert result == clean_json
            
            # Test JSON with extra text
            messy_response = 'Here is the result: {"test": "value"} and some more text'
            result = discoverer._extract_json_from_response(messy_response)
            assert result == '{"test": "value"}'
            
            # Test invalid response
            with pytest.raises(ValueError):
                discoverer._extract_json_from_response("No JSON here")
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_empty_dataframe_error(self, mock_llm, mock_api_key):
        """Test error handling for empty DataFrame"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot discover categories from empty dataset"):
            discoverer.discover_categories(empty_df)
    
    def test_validate_categories_schema(self, sample_categories_response):
        """Test categories schema validation"""
        # Valid schema
        assert validate_categories_schema(sample_categories_response) == True
        
        # Invalid schema - missing required key
        invalid_categories = sample_categories_response.copy()
        del invalid_categories["version"]
        assert validate_categories_schema(invalid_categories) == False
        
        # Invalid category structure
        invalid_cat_structure = sample_categories_response.copy()
        invalid_cat_structure["categories"][0] = {"incomplete": "category"}
        assert validate_categories_schema(invalid_cat_structure) == False
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    def test_category_count_warnings(self, mock_llm, mock_api_key, sample_tickets_df):
        """Test warnings for category count outside expected range"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key)
        
        # Test too few categories
        few_categories = {
            "categories": [{"id": 1, "technical_name": "test", "display_name": "Test", "description": "Test"}],
            "discovery_stats": {}
        }
        
        with patch.object(discoverer.logger, 'warning') as mock_warning:
            discoverer._validate_and_enhance_categories(few_categories, sample_tickets_df)
            mock_warning.assert_called()
    
    @patch('src.discovery.category_discoverer.ChatGoogleGenerativeAI')
    @patch('src.discovery.category_discoverer.ThreadPoolExecutor')
    def test_map_pattern_analysis_parallel_execution(self, mock_executor, mock_llm, 
                                                   mock_api_key, sample_tickets_df):
        """Test parallel execution in map phase"""
        discoverer = CategoryDiscoverer(api_key=mock_api_key, max_workers=2)
        
        # Mock executor behavior
        mock_future = Mock()
        mock_future.result.return_value = "pattern analysis result"
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value.__enter__.return_value.__iter__ = lambda x: iter([mock_future])
        
        # Mock text preparation and chunking
        with patch.object(discoverer, '_prepare_tickets_text', return_value="test text"):
            with patch.object(discoverer, '_create_discovery_chunks', return_value=["chunk1", "chunk2"]):
                with patch.object(discoverer, '_analyze_chunk_patterns', return_value="analysis"):
                    
                    result = discoverer._map_pattern_analysis(sample_tickets_df)
                    
                    assert len(result) > 0
                    mock_executor.assert_called_with(max_workers=2)


if __name__ == "__main__":
    pytest.main([__file__])