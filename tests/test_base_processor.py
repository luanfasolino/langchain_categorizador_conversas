"""Testes para o BaseProcessor."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd

from src.base_processor import BaseProcessor


class TestBaseProcessor:
    """Testes para a classe BaseProcessor."""

    def test_init_basic(self, mock_api_key, temp_dir):
        """Testa inicialização básica do BaseProcessor."""
        processor = BaseProcessor(mock_api_key, temp_dir)

        # Verifica atributos básicos baseados no código real
        assert hasattr(processor, "llm")
        assert processor.database_dir == temp_dir
        assert hasattr(processor, "max_workers")
        assert hasattr(processor, "use_cache")

    def test_clean_text(self, mock_api_key, temp_dir):
        """Testa limpeza de texto."""
        processor = BaseProcessor(mock_api_key, temp_dir)

        # Texto com caracteres especiais
        dirty_text = "Olá 📱 como você está?"
        clean_text = processor.clean_text(dirty_text)

        assert isinstance(clean_text, str)
        assert len(clean_text) > 0
        # Verifica que caracteres especiais foram tratados
        assert "📱" not in clean_text

    def test_extract_json_valid(self, mock_api_key, temp_dir):
        """Testa extração de JSON válido."""
        processor = BaseProcessor(mock_api_key, temp_dir)

        # JSON válido
        json_response = '{"categorias": ["Login"], "resumo": "Problema com login"}'
        result = processor.extract_json(json_response)

        assert isinstance(result, str)
        assert "categorias" in result

    def test_extract_json_with_markdown(self, mock_api_key, temp_dir):
        """Testa extração de JSON com markdown."""
        processor = BaseProcessor(mock_api_key, temp_dir)

        # JSON dentro de markdown
        markdown_response = """```json
{"categorias": ["Login"], "resumo": "Problema com login"}
```"""
        result = processor.extract_json(markdown_response)

        assert isinstance(result, str)
        assert "categorias" in result

    def test_estimate_tokens(self, mock_api_key, temp_dir):
        """Testa estimativa de tokens."""
        processor = BaseProcessor(mock_api_key, temp_dir)

        text = "Este é um texto de exemplo para testar a estimativa de tokens."
        tokens = processor.estimate_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_cache_functionality(self, mock_api_key, temp_dir):
        """Testa funcionalidade básica de cache."""
        processor = BaseProcessor(mock_api_key, temp_dir, use_cache=True)

        # Testa geração de chave de cache
        data = {"test": "data"}
        cache_key = processor._generate_cache_key(data)

        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
