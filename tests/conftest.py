"""Configurações compartilhadas para todos os testes."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_data():
    """Dados de exemplo para testes."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "conversa": [
                "Cliente tem problema com login",
                "Produto não funciona corretamente",
                "Preciso cancelar minha conta",
            ],
            "categoria": ["Login", "Produto", "Cancelamento"],
        }
    )


@pytest.fixture
def temp_dir():
    """Diretório temporário para testes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_api_key():
    """API key falsa para testes."""
    return "test_api_key_fake"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock das variáveis de ambiente."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    yield
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
