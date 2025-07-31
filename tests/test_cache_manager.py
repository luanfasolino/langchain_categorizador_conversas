"""
Testes para o CacheManager - Sistema de cache otimizado.
"""

import pytest
import tempfile
import time
import threading
import os
from pathlib import Path

from src.cache_manager import CacheManager


class TestCacheManager:
    """Testes para a classe CacheManager."""

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
            max_cache_size_mb=10,  # 10MB para testes
            max_file_size_mb=1,  # 1MB para forçar compressão
            use_compression=True,
            enable_statistics=True,
        )

    def test_cache_initialization(self, temp_cache_dir):
        """Testa inicialização do cache."""
        cache = CacheManager(cache_dir=temp_cache_dir)

        assert cache.cache_dir.exists()
        assert cache.max_cache_size_bytes > 0
        assert cache.use_compression is True
        assert cache.enable_statistics is True

    def test_cache_key_generation(self, cache_manager):
        """Testa geração de chaves de cache."""
        # Teste com diferentes tipos de dados
        key1 = cache_manager.generate_cache_key({"test": "data"})
        key2 = cache_manager.generate_cache_key({"test": "data"})
        key3 = cache_manager.generate_cache_key({"different": "data"})

        assert key1 == key2  # Mesmos dados devem gerar mesma chave
        assert key1 != key3  # Dados diferentes devem gerar chaves diferentes
        assert len(key1) == 64  # SHA-256 tem 64 caracteres hex

    def test_cache_key_with_version(self, cache_manager):
        """Testa geração de chaves com versão."""
        key1 = cache_manager.generate_cache_key({"test": "data"}, version="v1")
        key2 = cache_manager.generate_cache_key({"test": "data"}, version="v2")

        assert key1 != key2  # Versões diferentes devem gerar chaves diferentes

    def test_cache_key_with_timestamp(self, cache_manager):
        """Testa geração de chaves com timestamp."""
        key1 = cache_manager.generate_cache_key(
            {"test": "data"}, include_timestamp=True
        )

        # Pequeno delay para garantir timestamp diferente
        time.sleep(0.1)

        key2 = cache_manager.generate_cache_key(
            {"test": "data"}, include_timestamp=True
        )

        # Devem ser iguais se estiverem na mesma hora
        # (timestamp é em horas)
        assert key1 == key2

    def test_basic_cache_operations(self, cache_manager):
        """Testa operações básicas de cache (get/set)."""
        key = "test_key"
        data = {"test": "data", "number": 42}

        # Cache miss inicial
        assert cache_manager.get(key) is None

        # Salva no cache
        assert cache_manager.set(key, data) is True

        # Cache hit
        retrieved_data = cache_manager.get(key)
        assert retrieved_data == data

    def test_memory_cache_lru(self, cache_manager):
        """Testa política LRU do cache em memória."""
        # Adiciona muitos itens para forçar LRU
        for i in range(1050):  # Mais que o limite de 1000
            cache_manager.set(f"key_{i}", f"data_{i}")

        # Os primeiros itens devem ter sido removidos do cache em memória
        stats = cache_manager.get_statistics()
        assert stats["memory_cache"]["items"] <= 1000

    def test_cache_compression(self, cache_manager):
        """Testa compressão automática de arquivos grandes."""
        key = "large_data"
        # Cria dados grandes (>1MB para forçar compressão)
        large_data = "x" * (1024 * 1024 + 1)  # ~1MB

        assert cache_manager.set(key, large_data) is True

        # Verifica se arquivo comprimido foi criado
        compressed_file = cache_manager._get_compressed_cache_file_path(key)
        assert compressed_file.exists()

        # Verifica se dados podem ser recuperados
        retrieved_data = cache_manager.get(key)
        assert retrieved_data == large_data

    def test_cache_invalidation(self, cache_manager):
        """Testa invalidação de cache."""
        key = "test_invalidate"
        data = {"test": "data"}

        # Salva e verifica
        cache_manager.set(key, data)
        assert cache_manager.get(key) == data

        # Invalida
        assert cache_manager.invalidate(key) is True

        # Verifica que foi removido
        assert cache_manager.get(key) is None

        # Tentativa de invalidar item inexistente
        assert cache_manager.invalidate("nonexistent") is False

    def test_cache_clear_all(self, cache_manager):
        """Testa limpeza completa do cache."""
        # Adiciona vários itens
        for i in range(5):
            cache_manager.set(f"key_{i}", f"data_{i}")

        # Verifica que existem
        assert cache_manager.get("key_0") is not None

        # Limpa tudo
        assert cache_manager.clear_all() is True

        # Verifica que foram removidos
        for i in range(5):
            assert cache_manager.get(f"key_{i}") is None

        # Verifica estatísticas zeradas (hits e saves são zerados, mas misses podem ter ocorrido)
        stats = cache_manager.get_statistics()
        assert stats["performance"]["hits"] == 0
        assert stats["operations"]["saves"] == 0

    def test_cache_statistics(self, cache_manager):
        """Testa coleta de estatísticas."""
        # Operações para gerar estatísticas
        cache_manager.set("key1", "data1")
        cache_manager.get("key1")  # hit
        cache_manager.get("nonexistent")  # miss

        stats = cache_manager.get_statistics()

        # Verifica estrutura das estatísticas
        assert "performance" in stats
        assert "storage" in stats
        assert "memory_cache" in stats
        assert "operations" in stats
        assert "configuration" in stats

        # Verifica valores
        assert stats["performance"]["hits"] >= 1
        assert stats["performance"]["misses"] >= 1
        assert stats["performance"]["hit_rate_percent"] >= 0
        assert stats["operations"]["saves"] >= 1

    def test_cache_cleanup_old_files(self, cache_manager):
        """Testa limpeza de arquivos antigos."""
        key = "old_file"
        cache_manager.set(key, "data")

        # Simula arquivo antigo modificando timestamp
        cache_file = cache_manager._get_cache_file_path(key)
        old_time = time.time() - (74 * 3600)  # 74 horas atrás
        os.utime(cache_file, (old_time, old_time))

        # Executa cleanup
        removed = cache_manager.cleanup_old_files(max_age_hours=72)

        assert removed >= 1
        assert not cache_file.exists()

    def test_cache_optimization(self, cache_manager):
        """Testa otimização do cache."""
        # Cria arquivo grande não comprimido
        key = "large_uncompressed"
        large_data = "x" * (1024 * 1024 + 1)  # >1MB

        # Força salvamento sem compressão
        cache_manager.use_compression = False
        cache_manager.set(key, large_data)
        cache_manager.use_compression = True

        # Executa otimização
        results = cache_manager.optimize_cache()

        # Verifica se arquivo foi comprimido
        assert results["files_compressed"] >= 1

        # Verifica se arquivo comprimido existe
        compressed_file = cache_manager._get_compressed_cache_file_path(key)
        assert compressed_file.exists()

    def test_thread_safety(self, cache_manager):
        """Testa thread safety do cache."""
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    data = f"data_{worker_id}_{i}"

                    # Set e get em loop
                    cache_manager.set(key, data)
                    retrieved = cache_manager.get(key)

                    if retrieved == data:
                        results.append(f"worker_{worker_id}_success")
                    else:
                        errors.append(f"worker_{worker_id}_data_mismatch")

            except Exception as e:
                errors.append(f"worker_{worker_id}_exception: {str(e)}")

        # Cria múltiplas threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Aguarda conclusão
        for thread in threads:
            thread.join()

        # Verifica resultados
        assert len(errors) == 0, f"Erros encontrados: {errors}"
        assert len(results) == 50  # 5 workers * 10 items cada

    def test_cache_size_limit_enforcement(self, cache_manager):
        """Testa enforcement do limite de tamanho do cache."""
        # Cache pequeno para facilitar teste (10MB)
        large_data = "x" * (2 * 1024 * 1024)  # 2MB cada

        # Adiciona dados até ultrapassar limite
        keys = []
        for i in range(8):  # 8 * 2MB = 16MB > 10MB limit
            key = f"large_item_{i}"
            keys.append(key)
            cache_manager.set(key, large_data)

        # Força otimização para aplicar limpeza
        cache_manager.optimize_cache()

        # Verifica que alguns arquivos foram removidos
        cache_size = cache_manager._get_cache_size()
        max_size = cache_manager.max_cache_size_bytes

        # O cache deve estar dentro do limite ou próximo
        assert cache_size <= max_size * 1.1  # 10% de tolerância

    def test_error_handling(self, cache_manager):
        """Testa tratamento de erros."""

        # Testa com dados que não podem ser serializados
        class UnpicklableClass:
            def __reduce__(self):
                raise TypeError("Cannot pickle this object")

        unpicklable_data = UnpicklableClass()

        # Set deve falhar graciosamente
        result = cache_manager.set("bad_key", unpicklable_data)
        assert result is False

        # Estatísticas devem mostrar erro
        stats = cache_manager.get_statistics()
        assert stats["operations"]["errors"] > 0

    def test_cache_key_collision_resistance(self, cache_manager):
        """Testa resistência a colisões de chave."""
        # Dados similares mas diferentes
        data1 = {"order": ["a", "b"], "value": 1}
        data2 = {"order": ["b", "a"], "value": 1}
        data3 = {"order": ["a", "b"], "value": 2}

        key1 = cache_manager.generate_cache_key(data1)
        key2 = cache_manager.generate_cache_key(data2)
        key3 = cache_manager.generate_cache_key(data3)

        # Todas as chaves devem ser diferentes
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_persistence_across_instances(self, temp_cache_dir):
        """Testa persistência do cache entre instâncias."""
        # Primeira instância
        cache1 = CacheManager(cache_dir=temp_cache_dir)
        cache1.set("persistent_key", "persistent_data")

        # Segunda instância (mesmo diretório)
        cache2 = CacheManager(cache_dir=temp_cache_dir)

        # Deve recuperar dados da primeira instância
        data = cache2.get("persistent_key")
        assert data == "persistent_data"


if __name__ == "__main__":
    pytest.main([__file__])
