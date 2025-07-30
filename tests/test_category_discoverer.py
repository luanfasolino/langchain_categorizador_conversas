"""
Tests for CategoryDiscoverer module
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.category_discoverer import (
    CategoryDiscoverer,
    validate_categories_schema,
)


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
            "ticket_id": ["T001", "T001", "T001", "T002", "T002", "T003", "T003"],
            "sender": ["USER", "AGENT", "USER", "USER", "AGENT", "USER", "AGENT"],
            "text": [
                "Meu cartão foi recusado na compra",
                "Vou verificar o problema com seu cartão",
                "Obrigado pela ajuda",
                "Como alterar minha reserva?",
                "Posso ajudar com a alteração",
                "Site não está carregando",
                "Vamos verificar o problema técnico",
            ],
            "message_sended_at": [
                "2024-01-01 10:00:00",
                "2024-01-01 10:05:00",
                "2024-01-01 10:10:00",
                "2024-01-02 11:00:00",
                "2024-01-02 11:05:00",
                "2024-01-03 12:00:00",
                "2024-01-03 12:05:00",
            ],
            "category": ["TEXT"] * 7,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_categories(self):
        """Sample categories structure for testing"""
        return {
            "version": "1.0",
            "generated_at": "2024-01-01T10:00:00",
            "discovery_stats": {
                "total_patterns_analyzed": 3,
                "categories_created": 3,
                "confidence_level": 0.85,
            },
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                    "description": "Falhas em transações e cartões",
                    "keywords": ["cartão", "recusado", "pagamento"],
                    "examples": ["Meu cartão foi recusado"],
                    "subcategories": [],
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                    "description": "Modificações e cancelamentos",
                    "keywords": ["reserva", "alterar", "cancelar"],
                    "examples": ["Como alterar minha reserva?"],
                    "subcategories": [],
                },
            ],
            "metadata": {
                "llm_model": "gemini-2.5-flash",
                "discovery_method": "map_reduce_pattern_analysis",
                "chunk_size": 800000,
                "overlap_tokens": 240000,
            },
        }

    @pytest.fixture
    def mock_discoverer(self, mock_api_key):
        """Create a mocked CategoryDiscoverer instance"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("discovery.category_discoverer.ChatGoogleGenerativeAI"):
                discoverer = CategoryDiscoverer(
                    api_key=mock_api_key, database_dir=Path(tmp_dir)
                )
                yield discoverer

    def test_initialization(self, mock_api_key):
        """Test CategoryDiscoverer initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("discovery.category_discoverer.ChatGoogleGenerativeAI"):
                discoverer = CategoryDiscoverer(
                    api_key=mock_api_key, database_dir=Path(tmp_dir)
                )

                # API key is passed to LLM, not stored as attribute
                assert discoverer.database_dir == Path(tmp_dir)
                assert discoverer.chunk_size == 50_000
                assert discoverer.overlap == 15_000
                assert discoverer.min_categories == 5
                assert discoverer.max_categories == 25

    def test_prepare_tickets_text(self, mock_discoverer, sample_tickets_df):
        """Test ticket text preparation"""
        prepared_text = mock_discoverer._prepare_tickets_text(sample_tickets_df)

        assert isinstance(prepared_text, str)
        assert len(prepared_text) > 0
        assert "TICKET T001:" in prepared_text
        assert "TICKET T002:" in prepared_text
        assert "TICKET T003:" in prepared_text
        assert "[USER]:" in prepared_text
        assert "[AGENT]:" in prepared_text

    def test_create_discovery_chunks(self, mock_discoverer):
        """Test chunk creation for discovery"""
        sample_text = "Sample text for testing chunk creation. " * 100
        chunks = mock_discoverer._create_discovery_chunks(sample_text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_validate_categories_schema_valid(self, sample_categories):
        """Test validation with valid categories schema"""
        is_valid = validate_categories_schema(sample_categories)
        assert is_valid

    def test_validate_categories_schema_invalid_missing_keys(self):
        """Test validation with missing required keys"""
        invalid_categories = {
            "version": "1.0",
            # missing other required keys
        }
        is_valid = validate_categories_schema(invalid_categories)
        assert not is_valid

    def test_validate_categories_schema_invalid_category_structure(self):
        """Test validation with invalid category structure"""
        invalid_categories = {
            "version": "1.0",
            "generated_at": "2024-01-01T10:00:00",
            "discovery_stats": {},
            "categories": [
                {
                    "id": 1,
                    # missing required keys like technical_name, display_name
                }
            ],
            "metadata": {},
        }
        is_valid = validate_categories_schema(invalid_categories)
        assert not is_valid

    def test_get_discovery_stats(self, mock_discoverer, sample_categories):
        """Test discovery statistics generation"""
        stats = mock_discoverer.get_discovery_stats(sample_categories)

        assert isinstance(stats, dict)
        assert "total_categories" in stats
        assert "categories_with_subcategories" in stats
        assert "total_subcategories" in stats
        assert "avg_keywords_per_category" in stats
        assert "avg_examples_per_category" in stats
        assert "categories_by_complexity" in stats

        assert stats["total_categories"] == 2
        assert stats["categories_with_subcategories"] == 0
        assert stats["total_subcategories"] == 0

    def test_get_discovery_stats_empty_categories(self, mock_discoverer):
        """Test discovery stats with empty categories"""
        empty_categories = {}
        stats = mock_discoverer.get_discovery_stats(empty_categories)

        assert isinstance(stats, dict)
        assert stats == {}

    @patch("discovery.category_discoverer.ChatGoogleGenerativeAI")
    def test_discover_categories_with_mocked_llm(
        self, mock_llm, mock_api_key, sample_tickets_df
    ):
        """Test full discovery process with mocked LLM"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock LLM responses
            mock_chain = Mock()
            mock_chain.invoke.side_effect = [
                "Mock pattern analysis 1",
                "Mock consolidated analysis",
                '{"version": "1.0", "categories": []}',
            ]

            discoverer = CategoryDiscoverer(
                api_key=mock_api_key, database_dir=Path(tmp_dir)
            )
            discoverer.map_chain = mock_chain
            discoverer.combine_chain = mock_chain
            discoverer.extract_chain = mock_chain

            # Test discovery
            with patch.object(
                discoverer, "_validate_and_enhance_categories"
            ) as mock_validate:
                mock_validate.return_value = {
                    "categories": [],
                    "discovery_stats": {"categories_created": 0},
                }

                result = discoverer.discover_categories(
                    sample_tickets_df, force_rediscovery=True
                )

                assert isinstance(result, dict)
                mock_validate.assert_called_once()

    def test_load_categories(self, mock_discoverer, sample_categories):
        """Test loading categories from file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(sample_categories, f, indent=2)
            categories_path = Path(f.name)

        try:
            loaded_categories = mock_discoverer.load_categories(categories_path)
            assert loaded_categories == sample_categories
        finally:
            categories_path.unlink()

    def test_save_categories_json(self, mock_discoverer, sample_categories):
        """Test saving categories to JSON file"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_categories.json"

            mock_discoverer._save_categories_json(sample_categories, output_path)

            assert output_path.exists()

            # Verify content
            import json

            with open(output_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            assert saved_data == sample_categories

    def test_extract_json_from_response(self, mock_discoverer):
        """Test JSON extraction from LLM response"""
        response_with_json = 'Some text before {"key": "value"} some text after'
        extracted = mock_discoverer._extract_json_from_response(response_with_json)
        assert extracted == '{"key": "value"}'

    def test_extract_json_from_response_no_json(self, mock_discoverer):
        """Test JSON extraction when no JSON present"""
        response_without_json = "No JSON in this response"
        with pytest.raises(ValueError, match="No valid JSON found"):
            mock_discoverer._extract_json_from_response(response_without_json)

    def test_validate_and_enhance_categories(
        self, mock_discoverer, sample_categories, sample_tickets_df
    ):
        """Test category validation and enhancement"""
        enhanced = mock_discoverer._validate_and_enhance_categories(
            sample_categories, sample_tickets_df
        )

        assert "discovery_stats" in enhanced
        assert "total_tickets_analyzed" in enhanced["discovery_stats"]
        assert "unique_tickets" in enhanced["discovery_stats"]
        assert "categories_created" in enhanced["discovery_stats"]
        assert "avg_keywords_per_category" in enhanced["discovery_stats"]
        assert "confidence_level" in enhanced["discovery_stats"]

        assert enhanced["discovery_stats"]["total_tickets_analyzed"] == 7
        assert enhanced["discovery_stats"]["unique_tickets"] == 3
        assert enhanced["discovery_stats"]["categories_created"] == 2

    def test_validate_and_enhance_categories_invalid_structure(self, mock_discoverer):
        """Test validation with invalid categories structure"""
        invalid_categories = {"invalid": "structure"}
        sample_df = pd.DataFrame({"ticket_id": ["T001"], "text": ["test"]})

        with pytest.raises(ValueError, match="missing 'categories' key"):
            mock_discoverer._validate_and_enhance_categories(
                invalid_categories, sample_df
            )

    def test_discover_categories_empty_dataframe(self, mock_discoverer):
        """Test discovery with empty DataFrame"""
        empty_df = pd.DataFrame()

        with pytest.raises(
            ValueError, match="Cannot discover categories from empty dataset"
        ):
            mock_discoverer.discover_categories(empty_df)

    def test_chunk_size_and_overlap_configuration(self, mock_discoverer):
        """Test that chunk size and overlap are properly configured"""
        assert mock_discoverer.chunk_size == 50_000
        assert mock_discoverer.overlap == 15_000
        assert mock_discoverer.text_splitter._chunk_size == 50_000
        assert mock_discoverer.text_splitter._chunk_overlap == 15_000

    def test_discovery_chains_setup(self, mock_discoverer):
        """Test that discovery chains are properly set up"""
        assert hasattr(mock_discoverer, "map_chain")
        assert hasattr(mock_discoverer, "combine_chain")
        assert hasattr(mock_discoverer, "extract_chain")
        assert hasattr(mock_discoverer, "map_prompt")
        assert hasattr(mock_discoverer, "combine_prompt")
        assert hasattr(mock_discoverer, "extract_prompt")
