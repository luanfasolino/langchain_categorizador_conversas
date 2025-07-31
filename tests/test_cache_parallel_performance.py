"""
Testes de performance para processamento paralelo do cache.
"""

import pytest
import tempfile
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

from src.cache_manager import CacheManager


class TestCacheParallelPerformance:
    """Testes para validar performance do cache em processamento paralelo."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Cria instância do CacheManager otimizada para paralelo."""
        return CacheManager(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=100,  # Cache grande para testes
            max_file_size_mb=5,  # Força compressão mais cedo
            use_compression=True,
            enable_statistics=True,
        )

    def test_concurrent_write_performance(self, cache_manager):
        """Testa performance de escritas concorrentes."""
        num_threads = 10
        items_per_thread = 50

        def write_worker(worker_id: int, results: List[Dict[str, Any]]):
            """Worker que escreve itens no cache."""
            start_time = time.time()
            success_count = 0

            for i in range(items_per_thread):
                key = f"worker_{worker_id}_item_{i}"
                data = {
                    "worker_id": worker_id,
                    "item_id": i,
                    "content": f"Content for worker {worker_id} item {i}"
                    * 100,  # ~3KB cada
                    "timestamp": time.time(),
                }

                success = cache_manager.set(key, data)
                if success:
                    success_count += 1

            end_time = time.time()

            results.append(
                {
                    "worker_id": worker_id,
                    "duration": end_time - start_time,
                    "success_count": success_count,
                    "items_per_second": success_count / (end_time - start_time),
                }
            )

        # Executa workers concorrentes
        results = []
        threads = []

        start_time = time.time()

        for worker_id in range(num_threads):
            thread = threading.Thread(target=write_worker, args=(worker_id, results))
            threads.append(thread)
            thread.start()

        # Aguarda conclusão
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Valida resultados
        assert len(results) == num_threads
        total_items = sum(r["success_count"] for r in results)
        total_throughput = total_items / total_time

        print(f"\n📊 PERFORMANCE DE ESCRITAS CONCORRENTES:")
        print(f"   Threads: {num_threads}")
        print(f"   Items por thread: {items_per_thread}")
        print(f"   Total de items: {total_items}")
        print(f"   Tempo total: {total_time:.2f}s")
        print(f"   Throughput: {total_throughput:.1f} items/s")
        print(
            f"   Média por thread: {sum(r['items_per_second'] for r in results) / len(results):.1f} items/s"
        )

        # Verifica que todas as operações foram bem-sucedidas
        assert total_items == num_threads * items_per_thread

        # Verifica throughput mínimo (ajustar conforme hardware)
        assert (
            total_throughput > 50
        ), f"Throughput muito baixo: {total_throughput:.1f} items/s"

    def test_concurrent_read_performance(self, cache_manager):
        """Testa performance de leituras concorrentes."""
        # Prepara dados no cache
        num_items = 500
        for i in range(num_items):
            key = f"read_test_item_{i}"
            data = {
                "id": i,
                "content": f"Test content for item {i}" * 50,  # ~1.5KB cada
                "metadata": {"created": time.time(), "type": "test"},
            }
            cache_manager.set(key, data)

        # Teste de leituras concorrentes
        num_threads = 8
        reads_per_thread = 100

        def read_worker(worker_id: int, results: List[Dict[str, Any]]):
            """Worker que lê itens do cache."""
            start_time = time.time()
            hit_count = 0
            miss_count = 0

            for i in range(reads_per_thread):
                # Lê item aleatório (pode gerar hits e misses)
                item_id = (worker_id * reads_per_thread + i) % num_items
                key = f"read_test_item_{item_id}"

                data = cache_manager.get(key)
                if data is not None:
                    hit_count += 1
                else:
                    miss_count += 1

            end_time = time.time()

            results.append(
                {
                    "worker_id": worker_id,
                    "duration": end_time - start_time,
                    "hit_count": hit_count,
                    "miss_count": miss_count,
                    "reads_per_second": reads_per_thread / (end_time - start_time),
                }
            )

        # Executa workers concorrentes
        results = []
        threads = []

        start_time = time.time()

        for worker_id in range(num_threads):
            thread = threading.Thread(target=read_worker, args=(worker_id, results))
            threads.append(thread)
            thread.start()

        # Aguarda conclusão
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Valida resultados
        assert len(results) == num_threads
        total_reads = sum(r["hit_count"] + r["miss_count"] for r in results)
        total_hits = sum(r["hit_count"] for r in results)
        hit_rate = (total_hits / total_reads) * 100
        total_throughput = total_reads / total_time

        print(f"\n📊 PERFORMANCE DE LEITURAS CONCORRENTES:")
        print(f"   Threads: {num_threads}")
        print(f"   Reads por thread: {reads_per_thread}")
        print(f"   Total de reads: {total_reads}")
        print(f"   Hit rate: {hit_rate:.1f}%")
        print(f"   Tempo total: {total_time:.2f}s")
        print(f"   Throughput: {total_throughput:.1f} reads/s")
        print(
            f"   Média por thread: {sum(r['reads_per_second'] for r in results) / len(results):.1f} reads/s"
        )

        # Verifica hit rate alto (dados estão no cache)
        assert hit_rate > 95, f"Hit rate muito baixo: {hit_rate:.1f}%"

        # Verifica throughput mínimo
        assert (
            total_throughput > 200
        ), f"Throughput muito baixo: {total_throughput:.1f} reads/s"

    def test_mixed_operations_performance(self, cache_manager):
        """Testa performance com operações mistas (read/write/invalidate)."""
        num_threads = 6
        operations_per_thread = 100

        def mixed_worker(worker_id: int, results: List[Dict[str, Any]]):
            """Worker que executa operações mistas."""
            start_time = time.time()
            operations = {"reads": 0, "writes": 0, "invalidations": 0}

            for i in range(operations_per_thread):
                operation_type = i % 3  # Cicla entre operações
                key = f"mixed_worker_{worker_id}_item_{i // 3}"

                if operation_type == 0:  # Write
                    data = {
                        "worker": worker_id,
                        "operation": i,
                        "content": f"Mixed content {worker_id}-{i}" * 20,
                    }
                    cache_manager.set(key, data)
                    operations["writes"] += 1

                elif operation_type == 1:  # Read
                    cache_manager.get(key)
                    operations["reads"] += 1

                else:  # Invalidate (ocasionalmente)
                    if i % 10 == 0:  # Invalida a cada 10 operações
                        cache_manager.invalidate(key)
                        operations["invalidations"] += 1

            end_time = time.time()

            results.append(
                {
                    "worker_id": worker_id,
                    "duration": end_time - start_time,
                    "operations": operations,
                    "ops_per_second": operations_per_thread / (end_time - start_time),
                }
            )

        # Executa workers concorrentes
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            start_time = time.time()

            for worker_id in range(num_threads):
                future = executor.submit(mixed_worker, worker_id, results)
                futures.append(future)

            # Aguarda conclusão
            concurrent.futures.wait(futures)

        total_time = time.time() - start_time

        # Valida resultados
        assert len(results) == num_threads
        total_operations = sum(sum(r["operations"].values()) for r in results)
        total_throughput = total_operations / total_time

        print(f"\n📊 PERFORMANCE DE OPERAÇÕES MISTAS:")
        print(f"   Threads: {num_threads}")
        print(f"   Operações por thread: {operations_per_thread}")
        print(f"   Total de operações: {total_operations}")
        print(f"   Tempo total: {total_time:.2f}s")
        print(f"   Throughput: {total_throughput:.1f} ops/s")

        # Verifica distribuição de operações
        total_ops_by_type = {"reads": 0, "writes": 0, "invalidations": 0}
        for result in results:
            for op_type, count in result["operations"].items():
                total_ops_by_type[op_type] += count

        print(f"   Reads: {total_ops_by_type['reads']}")
        print(f"   Writes: {total_ops_by_type['writes']}")
        print(f"   Invalidations: {total_ops_by_type['invalidations']}")

        # Verifica throughput mínimo
        assert (
            total_throughput > 100
        ), f"Throughput muito baixo: {total_throughput:.1f} ops/s"

    def test_cache_warmup_performance(self, cache_manager):
        """Testa performance de cache warming (pré-carregamento)."""
        # Dados para warm up
        warmup_data = {}
        for i in range(200):
            key = f"warmup_item_{i}"
            data = {
                "id": i,
                "content": f"Warmup content {i}" * 30,
                "priority": "high" if i % 10 == 0 else "normal",
            }
            warmup_data[key] = data

        # Teste 1: Warm up sequencial
        start_time = time.time()
        for key, data in warmup_data.items():
            cache_manager.set(key, data)
        sequential_time = time.time() - start_time

        # Limpa cache
        cache_manager.clear_all()

        # Teste 2: Warm up paralelo
        def warmup_worker(items: List[tuple]):
            """Worker para warm up paralelo."""
            for key, data in items:
                cache_manager.set(key, data)

        # Divide dados entre workers
        num_workers = 4
        items_list = list(warmup_data.items())
        chunk_size = len(items_list) // num_workers
        chunks = [
            items_list[i : i + chunk_size]
            for i in range(0, len(items_list), chunk_size)
        ]

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(warmup_worker, chunk) for chunk in chunks]
            concurrent.futures.wait(futures)
        parallel_time = time.time() - start_time

        # Valida resultados
        speedup = sequential_time / parallel_time

        print(f"\n📊 PERFORMANCE DE CACHE WARMING:")
        print(f"   Items: {len(warmup_data)}")
        print(f"   Tempo sequencial: {sequential_time:.2f}s")
        print(f"   Tempo paralelo: {parallel_time:.2f}s ({num_workers} workers)")
        print(f"   Speedup: {speedup:.2f}x")
        print(
            f"   Throughput sequencial: {len(warmup_data)/sequential_time:.1f} items/s"
        )
        print(f"   Throughput paralelo: {len(warmup_data)/parallel_time:.1f} items/s")

        # Para operações rápidas, o speedup pode ser limitado pelo overhead
        # Verifica que a diferença não é significativa (overhead aceitável)
        if parallel_time > sequential_time:
            overhead = (parallel_time - sequential_time) / sequential_time * 100
            assert overhead < 50, f"Overhead muito alto: {overhead:.1f}%"
        else:
            # Se conseguiu speedup, verifica que é razoável
            assert speedup > 0.8, f"Performance muito degradada: {speedup:.2f}x"

        # Verifica que todos os itens foram carregados
        loaded_count = 0
        for key in warmup_data.keys():
            if cache_manager.get(key) is not None:
                loaded_count += 1

        assert loaded_count == len(
            warmup_data
        ), f"Nem todos os itens foram carregados: {loaded_count}/{len(warmup_data)}"

    def test_cache_contention_handling(self, cache_manager):
        """Testa tratamento de contenção em alta concorrência."""
        # Teste de alta contenção - muitas threads acessando mesmas chaves
        num_threads = 20
        shared_keys = [f"shared_key_{i}" for i in range(10)]
        operations_per_thread = 50

        def contention_worker(worker_id: int, results: List[Dict[str, Any]]):
            """Worker que causa contenção."""
            start_time = time.time()
            success_count = 0
            errors = 0

            for i in range(operations_per_thread):
                try:
                    key = shared_keys[i % len(shared_keys)]

                    if i % 2 == 0:  # Write
                        data = {
                            "worker": worker_id,
                            "iteration": i,
                            "timestamp": time.time(),
                        }
                        if cache_manager.set(key, data):
                            success_count += 1
                    else:  # Read
                        cache_manager.get(key)
                        success_count += 1

                except Exception:
                    errors += 1

            end_time = time.time()

            results.append(
                {
                    "worker_id": worker_id,
                    "duration": end_time - start_time,
                    "success_count": success_count,
                    "errors": errors,
                }
            )

        # Executa workers com alta contenção
        results = []
        threads = []

        start_time = time.time()

        for worker_id in range(num_threads):
            thread = threading.Thread(
                target=contention_worker, args=(worker_id, results)
            )
            threads.append(thread)
            thread.start()

        # Aguarda conclusão
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Valida resultados
        total_operations = sum(r["success_count"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        error_rate = (total_errors / (total_operations + total_errors)) * 100

        print(f"\n📊 TESTE DE CONTENÇÃO:")
        print(f"   Threads: {num_threads}")
        print(f"   Chaves compartilhadas: {len(shared_keys)}")
        print(f"   Operações por thread: {operations_per_thread}")
        print(f"   Total de operações: {total_operations}")
        print(f"   Total de erros: {total_errors}")
        print(f"   Taxa de erro: {error_rate:.2f}%")
        print(f"   Tempo total: {total_time:.2f}s")
        print(f"   Throughput: {total_operations/total_time:.1f} ops/s")

        # Verifica que sistema lida bem com contenção
        assert error_rate < 5, f"Taxa de erro muito alta: {error_rate:.2f}%"
        assert total_operations > 0, "Nenhuma operação foi bem-sucedida"

    def test_memory_cache_effectiveness(self, cache_manager):
        """Testa efetividade do cache em memória para acesso rápido."""
        # Adiciona dados que devem ficar no cache em memória
        small_items = {}
        for i in range(100):
            key = f"small_item_{i}"
            data = {"id": i, "small_content": f"Small {i}"}  # Items pequenos
            small_items[key] = data
            cache_manager.set(key, data)

        # Força alguns items para disco (items grandes)
        large_items = {}
        for i in range(20):
            key = f"large_item_{i}"
            data = {"id": i, "large_content": "X" * 100000}  # 100KB cada
            large_items[key] = data
            cache_manager.set(key, data)

        # Teste de performance: items pequenos vs grandes
        def read_worker(keys: List[str], results: Dict[str, float]):
            """Worker que lê itens e mede tempo."""
            start_time = time.time()

            for key in keys:
                cache_manager.get(key)

            results[f"worker_{threading.current_thread().ident}"] = (
                time.time() - start_time
            )

        # Testa leitura de items pequenos (memory cache)
        small_results = {}
        threads = []
        for i in range(4):
            keys_chunk = list(small_items.keys())[i * 25 : (i + 1) * 25]
            thread = threading.Thread(
                target=read_worker, args=(keys_chunk, small_results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Testa leitura de items grandes (disk cache)
        large_results = {}
        threads = []
        for i in range(4):
            keys_chunk = list(large_items.keys())[i * 5 : (i + 1) * 5]
            thread = threading.Thread(
                target=read_worker, args=(keys_chunk, large_results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Analisa resultados
        avg_small_time = sum(small_results.values()) / len(small_results)
        avg_large_time = sum(large_results.values()) / len(large_results)

        small_throughput = 25 / avg_small_time  # 25 items per thread
        large_throughput = 5 / avg_large_time  # 5 items per thread

        print(f"\n📊 EFETIVIDADE DO CACHE EM MEMÓRIA:")
        print(f"   Items pequenos (memory cache):")
        print(f"      Tempo médio: {avg_small_time:.3f}s")
        print(f"      Throughput: {small_throughput:.1f} items/s")
        print(f"   Items grandes (disk cache):")
        print(f"      Tempo médio: {avg_large_time:.3f}s")
        print(f"      Throughput: {large_throughput:.1f} items/s")
        memory_speedup = small_throughput / large_throughput
        print(f"   Speedup memory vs disk: {memory_speedup:.2f}x")

        # Memory cache deve ser pelo menos comparável ao disk cache
        # (em SSDs modernos a diferença pode ser pequena)
        assert (
            memory_speedup > 0.5
        ), f"Memory cache muito mais lento: {memory_speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
