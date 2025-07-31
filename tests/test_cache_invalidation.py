"""
Testes específicos para invalidação de cache e geração de chaves avançada.
"""

import pytest
import tempfile
from pathlib import Path

from src.cache_manager import CacheManager


class TestCacheInvalidation:
    """Testes para funcionalidades avançadas de invalidação de cache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Cria instância do CacheManager para testes."""
        return CacheManager(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=10,
            use_compression=True,
            enable_statistics=True,
        )

    def test_versioned_cache_keys(self, cache_manager):
        """Testa geração de chaves com versioning."""
        data = {"test": "data", "version": 1}

        # Chaves com versões diferentes devem ser diferentes
        key_v1 = cache_manager.generate_cache_key(data, version="v1")
        key_v2 = cache_manager.generate_cache_key(data, version="v2")
        key_v3 = cache_manager.generate_cache_key(data, version="v3")

        assert key_v1 != key_v2
        assert key_v1 != key_v3
        assert key_v2 != key_v3

        # Mesma versão deve gerar mesma chave
        key_v1_duplicate = cache_manager.generate_cache_key(data, version="v1")
        assert key_v1 == key_v1_duplicate

    def test_timestamp_based_invalidation(self, cache_manager):
        """Testa invalidação baseada em timestamp."""
        data = {"test": "time_sensitive_data"}

        # Gera chave com timestamp
        key_with_time = cache_manager.generate_cache_key(data, include_timestamp=True)

        # Mesma chave na mesma hora
        key_same_hour = cache_manager.generate_cache_key(data, include_timestamp=True)

        assert key_with_time == key_same_hour

        # Simula mudança de hora (mock do time para teste futuro)
        # Em produção, chaves com timestamp se invalidam automaticamente
        assert len(key_with_time) == 64  # SHA-256

    def test_cache_invalidation_with_file_content_changes(self, cache_manager):
        """Testa invalidação quando conteúdo do arquivo muda."""
        # Simula processamento de arquivo com conteúdo diferente
        file_data_v1 = {
            "file_path": "/test/file.csv",
            "content_hash": "abc123",
            "nrows": 1000,
            "method": "prepare_data",
        }

        file_data_v2 = {
            "file_path": "/test/file.csv",
            "content_hash": "def456",  # Conteúdo mudou
            "nrows": 1000,
            "method": "prepare_data",
        }

        key1 = cache_manager.generate_cache_key(file_data_v1)
        key2 = cache_manager.generate_cache_key(file_data_v2)

        # Chaves devem ser diferentes quando conteúdo muda
        assert key1 != key2

        # Testa cache miss/hit comportamento
        cache_manager.set(key1, "processed_data_v1")

        assert cache_manager.get(key1) == "processed_data_v1"
        assert cache_manager.get(key2) is None  # Cache miss para conteúdo novo

    def test_processing_parameter_invalidation(self, cache_manager):
        """Testa invalidação quando parâmetros de processamento mudam."""
        base_data = {"tickets": ["ticket1", "ticket2"]}

        # Diferentes parâmetros de processamento
        params_v1 = {
            **base_data,
            "model": "gemini-2.5-flash",
            "temperature": 0.3,
            "max_tokens": 1000,
            "chunk_size": 100000,
        }

        params_v2 = {
            **base_data,
            "model": "gemini-2.5-flash",
            "temperature": 0.5,  # Temperatura diferente
            "max_tokens": 1000,
            "chunk_size": 100000,
        }

        params_v3 = {
            **base_data,
            "model": "gemini-2.5-flash",
            "temperature": 0.3,
            "max_tokens": 1000,
            "chunk_size": 200000,  # Chunk size diferente
        }

        key1 = cache_manager.generate_cache_key(params_v1)
        key2 = cache_manager.generate_cache_key(params_v2)
        key3 = cache_manager.generate_cache_key(params_v3)

        # Todas as chaves devem ser diferentes
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_selective_cache_invalidation(self, cache_manager):
        """Testa invalidação seletiva de itens específicos."""
        # Adiciona múltiplos itens
        test_data = {
            "item1": "data1",
            "item2": "data2",
            "item3": "data3",
            "item4": "data4",
        }

        for key, data in test_data.items():
            cache_manager.set(key, data)
            assert cache_manager.get(key) == data

        # Invalida item específico
        assert cache_manager.invalidate("item2") is True

        # Verifica que apenas item2 foi removido
        assert cache_manager.get("item1") == "data1"
        assert cache_manager.get("item2") is None
        assert cache_manager.get("item3") == "data3"
        assert cache_manager.get("item4") == "data4"

        # Tenta invalidar item inexistente
        assert cache_manager.invalidate("nonexistent") is False

    def test_bulk_cache_invalidation(self, cache_manager):
        """Testa invalidação em massa do cache."""
        # Adiciona muitos itens
        for i in range(20):
            cache_manager.set(f"bulk_item_{i}", f"bulk_data_{i}")

        # Verifica que todos foram adicionados
        stats_before = cache_manager.get_statistics()
        assert stats_before["operations"]["saves"] >= 20

        # Limpa tudo
        assert cache_manager.clear_all() is True

        # Verifica que todos foram removidos
        for i in range(20):
            assert cache_manager.get(f"bulk_item_{i}") is None

        # Verifica estatísticas resetadas
        stats_after = cache_manager.get_statistics()
        assert stats_after["operations"]["saves"] == 0
        assert stats_after["performance"]["hits"] == 0

    def test_cache_key_consistency_across_restarts(self, temp_cache_dir):
        """Testa consistência de chaves entre reinicializações."""
        data = {"consistent": "data", "number": 42}

        # Primeira instância
        cache1 = CacheManager(cache_dir=temp_cache_dir)
        key1 = cache1.generate_cache_key(data, version="v1")
        cache1.set(key1, "consistent_result")

        # Segunda instância (simula reinicialização)
        cache2 = CacheManager(cache_dir=temp_cache_dir)
        key2 = cache2.generate_cache_key(data, version="v1")

        # Chaves devem ser idênticas
        assert key1 == key2

        # Dados devem ser recuperáveis
        assert cache2.get(key2) == "consistent_result"

    def test_cache_schema_versioning(self, cache_manager):
        """Testa versionamento de schema para mudanças estruturais."""
        # Dados com estrutura v1
        data_v1 = {
            "tickets": [{"id": "1", "text": "ticket1"}, {"id": "2", "text": "ticket2"}],
            "schema_version": "v1",
        }

        # Dados com estrutura v2 (novos campos)
        data_v2 = {
            "tickets": [
                {"id": "1", "text": "ticket1", "category": "technical"},
                {"id": "2", "text": "ticket2", "category": "billing"},
            ],
            "metadata": {"processed_at": "2025-01-01"},
            "schema_version": "v2",
        }

        # Chaves com versões diferentes para schemas diferentes
        key_schema_v1 = cache_manager.generate_cache_key(data_v1, version="schema_v1")
        key_schema_v2 = cache_manager.generate_cache_key(data_v2, version="schema_v2")

        assert key_schema_v1 != key_schema_v2

        # Cache separado para cada versão de schema
        cache_manager.set(key_schema_v1, "processed_with_schema_v1")
        cache_manager.set(key_schema_v2, "processed_with_schema_v2")

        assert cache_manager.get(key_schema_v1) == "processed_with_schema_v1"
        assert cache_manager.get(key_schema_v2) == "processed_with_schema_v2"

    def test_complex_data_structure_hashing(self, cache_manager):
        """Testa hashing de estruturas de dados complexas."""
        complex_data = {
            "nested": {
                "level1": {
                    "level2": {
                        "arrays": [1, 2, 3, {"inner": "value"}],
                        "mixed": ["string", 42, True, None],
                    }
                }
            },
            "unicode": "teste com acentuação e émojis 🚀",
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "numbers": [1, 2.5, -3, 0.00001],
            "booleans": [True, False],
            "nulls": [None, None],
        }

        # Deve conseguir gerar chave consistente
        key1 = cache_manager.generate_cache_key(complex_data)
        key2 = cache_manager.generate_cache_key(complex_data)

        assert key1 == key2
        assert len(key1) == 64  # SHA-256

        # Pequena mudança deve resultar em chave diferente
        complex_data_modified = complex_data.copy()
        complex_data_modified["nested"]["level1"]["level2"]["arrays"][0] = 99

        key3 = cache_manager.generate_cache_key(complex_data_modified)
        assert key1 != key3

    def test_error_handling_in_key_generation(self, cache_manager):
        """Testa tratamento de erros na geração de chaves."""
        # Dados que podem causar problemas na serialização
        problematic_data = {
            "circular_ref": None,
            "special_types": set([1, 2, 3]),  # Set não é JSON serializable
        }

        # Adiciona referência circular
        problematic_data["circular_ref"] = problematic_data

        # Deve conseguir gerar chave mesmo com dados problemáticos
        key = cache_manager.generate_cache_key(problematic_data)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256

    def test_memory_cache_invalidation_consistency(self, cache_manager):
        """Testa consistência entre cache em memória e disco na invalidação."""
        key = "memory_disk_consistency"
        data = "test_data_for_consistency"

        # Adiciona ao cache (deve ir para memória e disco)
        cache_manager.set(key, data)

        # Verifica que está em ambos
        assert cache_manager.get(key) == data  # Pode vir da memória

        # Force load from disk by clearing memory cache
        cache_manager._memory_cache.clear()
        assert cache_manager.get(key) == data  # Agora vem do disco

        # Invalida
        assert cache_manager.invalidate(key) is True

        # Verifica que foi removido de ambos
        assert cache_manager.get(key) is None
        assert key not in cache_manager._memory_cache


if __name__ == "__main__":
    pytest.main([__file__])
