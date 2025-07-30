"""
Tests for FastClassifier module
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from discovery.fast_classifier import (
    FastClassifier, ClassificationResult, BatchResult, load_and_classify_tickets
)


class TestFastClassifier:
    """Test cases for FastClassifier class"""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"

    @pytest.fixture
    def sample_tickets_df(self):
        """Create sample tickets DataFrame for testing"""
        data = {
            'ticket_id': ['T001', 'T002', 'T003'],
            'sender': ['USER', 'USER', 'USER'],
            'text': [
                'Meu cartão foi recusado na compra',
                'Como alterar minha reserva?',
                'Site não está carregando'
            ],
            'category': ['TEXT', 'TEXT', 'TEXT']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_categories(self):
        """Sample categories for testing"""
        return {
            "categories": [
                {
                    "id": 1,
                    "technical_name": "payment_issues",
                    "display_name": "Problemas de Pagamento",
                    "keywords": ["cartão", "pagamento"]
                },
                {
                    "id": 2,
                    "technical_name": "booking_changes",
                    "display_name": "Alterações de Reserva",
                    "keywords": ["reserva", "alterar"]
                }
            ]
        }

    @pytest.fixture
    def mock_classifier(self, mock_api_key):
        """Create a mocked FastClassifier instance"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('discovery.fast_classifier.ChatGoogleGenerativeAI'):
                classifier = FastClassifier(
                    api_key=mock_api_key,
                    database_dir=Path(tmp_dir)
                )
                yield classifier

    def test_initialization(self, mock_api_key):
        """Test FastClassifier initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('discovery.fast_classifier.ChatGoogleGenerativeAI'):
                classifier = FastClassifier(
                    api_key=mock_api_key,
                    database_dir=Path(tmp_dir)
                )

                assert classifier.api_key == mock_api_key
                assert classifier.database_dir == Path(tmp_dir)
                assert classifier.batch_size == 100
                assert classifier.max_workers == 4
                assert classifier.confidence_threshold == 0.7

    def test_load_categories(self, mock_classifier, sample_categories):
        """Test loading categories from dictionary"""
        loaded_categories = mock_classifier._load_categories_from_dict(
            sample_categories
        )

        assert isinstance(loaded_categories, dict)
        assert len(loaded_categories["categories"]) == 2

    def test_prepare_tickets_for_classification(
        self, mock_classifier, sample_tickets_df
    ):
        """Test ticket preparation for classification"""
        prepared = mock_classifier._prepare_tickets_for_classification(
            sample_tickets_df
        )

        assert isinstance(prepared, list)
        assert len(prepared) == 3
        assert all("ticket_id" in ticket for ticket in prepared)
        assert all("text" in ticket for ticket in prepared)

    def test_parse_classification_response_valid(self, mock_classifier):
        """Test parsing valid classification response"""
        response = '{"categories": [1, 2], "confidence": 0.95}'
        parsed = mock_classifier._parse_classification_response(response)

        assert parsed["categories"] == [1, 2]
        assert parsed["confidence"] == 0.95

    def test_parse_classification_response_invalid(self, mock_classifier):
        """Test parsing invalid classification response"""
        response = 'invalid json'
        parsed = mock_classifier._parse_classification_response(response)

        assert parsed["categories"] == []
        assert parsed["confidence"] == 0.0

    def test_get_category_name(self, mock_classifier, sample_categories):
        """Test category name lookup"""
        mock_classifier.categories = sample_categories

        name = mock_classifier._get_category_name(1)
        assert name == "Problemas de Pagamento"

        name = mock_classifier._get_category_name(999)
        assert name == "Unknown"

    def test_estimate_cost(self, mock_classifier):
        """Test cost estimation"""
        cost = mock_classifier._estimate_cost(10000)  # 10K tokens
        assert isinstance(cost, float)
        assert cost > 0

    def test_get_classification_stats(self, mock_classifier):
        """Test classification statistics"""
        results_df = pd.DataFrame({
            'ticket_id': ['T001', 'T002'],
            'category_ids': ['[1]', '[2]'],
            'confidence': [0.95, 0.87]
        })

        stats = mock_classifier.get_classification_stats(results_df)

        assert isinstance(stats, dict)
        assert "total_classified" in stats
        assert "avg_confidence" in stats
        assert stats["total_classified"] == 2

    def test_format_categories_for_prompt(self, mock_classifier, sample_categories):
        """Test category formatting for prompt"""
        formatted = mock_classifier._format_categories_for_prompt(sample_categories)

        assert isinstance(formatted, str)
        assert "Problemas de Pagamento" in formatted
        assert "Alterações de Reserva" in formatted

    def test_classification_result_dataclass(self):
        """Test ClassificationResult dataclass"""
        result = ClassificationResult(
            ticket_id="T001",
            categories=[1, 2],
            category_names=["Payment", "Booking"],
            confidence=0.95,
            processing_time=1.2,
            tokens_used=150
        )

        assert result.ticket_id == "T001"
        assert result.categories == [1, 2]
        assert result.confidence == 0.95

    def test_batch_result_dataclass(self):
        """Test BatchResult dataclass"""
        results = [
            ClassificationResult("T001", [1], ["Payment"], 0.95, 1.2, 150),
            ClassificationResult("T002", [2], ["Booking"], 0.87, 1.5, 160)
        ]
        batch_result = BatchResult(
            results=results,
            total_processing_time=2.7,
            total_tokens_used=310,
            batch_size=2,
            success_count=2,
            error_count=0
        )

        assert len(batch_result.results) == 2
        assert batch_result.success_count == 2
        assert batch_result.error_count == 0

    def test_load_and_classify_tickets_function(self):
        """Test the load_and_classify_tickets utility function"""
        # This is a placeholder test since the function requires complex setup
        assert callable(load_and_classify_tickets)
