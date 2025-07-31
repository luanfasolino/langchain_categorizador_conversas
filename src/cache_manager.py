"""
Cache Manager - Sistema de cache otimizado para o Categorizador de Conversas.

Este módulo implementa um sistema de cache avançado com:
- Gerenciamento de tamanho e memória
- Compressão de dados grandes
- Política LRU (Least Recently Used)
- Estatísticas e monitoramento
- Cache threadsafe para processamento paralelo
- Limpeza automática e manutenção
"""

import pickle
import hashlib
import time
import json
import threading
import gzip
from pathlib import Path
from typing import Any, Dict, Optional
from collections import OrderedDict
import logging


class CacheManager:
    """
    Gerenciador de cache otimizado com recursos avançados.

    Features:
    - LRU eviction policy
    - Compressão automática para arquivos grandes
    - Thread safety para processamento paralelo
    - Estatísticas detalhadas
    - Limpeza automática
    - Invalidação inteligente
    """

    def __init__(
        self,
        cache_dir: Path,
        max_cache_size_mb: int = 1024,  # 1GB limite padrão
        max_file_size_mb: int = 50,  # 50MB por arquivo antes de comprimir
        use_compression: bool = True,
        enable_statistics: bool = True,
        cleanup_interval_hours: int = 24,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Configurações
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.use_compression = use_compression
        self.enable_statistics = enable_statistics
        self.cleanup_interval = cleanup_interval_hours * 3600

        # LRU cache em memória para acesso rápido
        self._memory_cache = OrderedDict()
        self._cache_lock = threading.RLock()

        # Estatísticas
        self._stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "errors": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "compressed_files": 0,
            "total_size_bytes": 0,
            "last_cleanup": time.time(),
        }

        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Inicialização
        self._load_existing_cache_info()
        self._schedule_cleanup()

    def generate_cache_key(
        self, data: Any, include_timestamp: bool = False, version: str = "v1"
    ) -> str:
        """
        Gera chave de cache SHA-256 melhorada.

        Args:
            data: Dados para gerar a chave
            include_timestamp: Se deve incluir timestamp para invalidação automática
            version: Versão do cache para esquemas diferentes
        """
        try:
            # Prepara dados para hash
            if isinstance(data, dict):
                # Ordenar chaves para consistência
                data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, (list, tuple)):
                # Para listas/tuplas, converte cada item
                data_str = json.dumps(
                    [str(item) for item in data], sort_keys=True, ensure_ascii=False
                )
            else:
                data_str = str(data)

            # Adiciona versão e timestamp se necessário
            if include_timestamp:
                # Timestamp em horas para invalidação diária
                timestamp = int(time.time() // 3600)
                data_str = f"{version}:{timestamp}:{data_str}"
            else:
                data_str = f"{version}:{data_str}"

            # Gera hash SHA-256
            return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

        except Exception as e:
            self.logger.error(f"Erro ao gerar chave de cache: {e}")
            # Fallback para hash simples
            return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Recupera item do cache com suporte a LRU e compressão.
        """
        with self._cache_lock:
            # Verifica cache em memória primeiro (mais rápido)
            if cache_key in self._memory_cache:
                # Move para o final (mais recente)
                value = self._memory_cache.pop(cache_key)
                self._memory_cache[cache_key] = value
                self._stats["hits"] += 1
                self._stats["memory_hits"] += 1

                if self.enable_statistics:
                    self.logger.debug(f"Cache hit (memory): {cache_key[:16]}...")

                return value

            # Verifica cache em disco
            cache_file = self._get_cache_file_path(cache_key)
            compressed_file = self._get_compressed_cache_file_path(cache_key)

            # Tenta primeiro arquivo comprimido, depois normal
            for file_path, is_compressed in [
                (compressed_file, True),
                (cache_file, False),
            ]:
                if file_path.exists():
                    try:
                        data = self._load_from_disk(file_path, is_compressed)

                        # Adiciona ao cache em memória se não for muito grande
                        data_size = len(pickle.dumps(data))
                        if data_size < self.max_file_size_bytes // 10:  # 10% do limite
                            self._add_to_memory_cache(cache_key, data)

                        # Atualiza timestamp de acesso
                        file_path.touch()

                        self._stats["hits"] += 1
                        self._stats["disk_hits"] += 1

                        if self.enable_statistics:
                            self.logger.debug(
                                f"Cache hit (disk, {'compressed' if is_compressed else 'normal'}): {cache_key[:16]}..."
                            )

                        return data

                    except Exception as e:
                        self.logger.error(f"Erro ao carregar cache {cache_key}: {e}")
                        # Remove arquivo corrompido
                        try:
                            file_path.unlink()
                        except Exception:
                            pass

            # Cache miss
            self._stats["misses"] += 1
            if self.enable_statistics:
                self.logger.debug(f"Cache miss: {cache_key[:16]}...")

            return None

    def set(self, cache_key: str, data: Any, force_compression: bool = False) -> bool:
        """
        Salva item no cache com compressão automática e gestão de tamanho.
        """
        try:
            with self._cache_lock:
                # Serializa dados
                serialized_data = pickle.dumps(data)
                data_size = len(serialized_data)

                # Decide se usa compressão
                use_compression = self.use_compression and (
                    data_size > self.max_file_size_bytes or force_compression
                )

                # Verifica se deve limpar cache antes de salvar
                if self._should_cleanup_cache(data_size):
                    self._cleanup_old_files()

                # Salva em disco
                if use_compression:
                    file_path = self._get_compressed_cache_file_path(cache_key)
                    self._save_to_disk(file_path, data, True)
                    self._stats["compressed_files"] += 1
                else:
                    file_path = self._get_cache_file_path(cache_key)
                    self._save_to_disk(file_path, data, False)

                # Adiciona ao cache em memória se não for muito grande
                if data_size < self.max_file_size_bytes // 10:
                    self._add_to_memory_cache(cache_key, data)

                # Atualiza estatísticas
                self._stats["saves"] += 1
                self._stats["total_size_bytes"] += data_size

                if self.enable_statistics:
                    self.logger.debug(
                        f"Cache saved ({'compressed' if use_compression else 'normal'}): "
                        f"{cache_key[:16]}... ({data_size:,} bytes)"
                    )

                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar cache {cache_key}: {e}")
            self._stats["errors"] += 1
            return False

    def invalidate(self, cache_key: str) -> bool:
        """Remove item específico do cache."""
        with self._cache_lock:
            try:
                # Remove do cache em memória
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]

                # Remove arquivos do disco
                files_removed = 0
                for file_path in [
                    self._get_cache_file_path(cache_key),
                    self._get_compressed_cache_file_path(cache_key),
                ]:
                    if file_path.exists():
                        file_path.unlink()
                        files_removed += 1

                if files_removed > 0:
                    self.logger.info(f"Cache invalidated: {cache_key[:16]}...")
                    return True

                return False

            except Exception as e:
                self.logger.error(f"Erro ao invalidar cache {cache_key}: {e}")
                return False

    def clear_all(self) -> bool:
        """Remove todos os itens do cache."""
        with self._cache_lock:
            try:
                # Limpa cache em memória
                self._memory_cache.clear()

                # Remove todos os arquivos de cache
                files_removed = 0
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                    files_removed += 1

                for file_path in self.cache_dir.glob("*.pkl.gz"):
                    file_path.unlink()
                    files_removed += 1

                # Reset estatísticas
                self._stats.update(
                    {
                        "hits": 0,
                        "misses": 0,
                        "saves": 0,
                        "errors": 0,
                        "memory_hits": 0,
                        "disk_hits": 0,
                        "compressed_files": 0,
                        "total_size_bytes": 0,
                    }
                )

                self.logger.info(f"Cache cleared: {files_removed} files removed")
                return True

            except Exception as e:
                self.logger.error(f"Erro ao limpar cache: {e}")
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas do cache."""
        with self._cache_lock:
            # Calcula estatísticas em tempo real
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_requests * 100)
                if total_requests > 0
                else 0
            )

            # Estatísticas de arquivos
            cache_files = list(self.cache_dir.glob("*.pkl")) + list(
                self.cache_dir.glob("*.pkl.gz")
            )
            total_files = len(cache_files)

            # Calcula tamanho real do cache
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())

            # Estatísticas de memória
            memory_cache_size = len(self._memory_cache)
            memory_usage_bytes = sum(
                len(pickle.dumps(value)) for value in self._memory_cache.values()
            )

            return {
                "performance": {
                    "total_requests": total_requests,
                    "hits": self._stats["hits"],
                    "misses": self._stats["misses"],
                    "hit_rate_percent": round(hit_rate, 2),
                    "memory_hits": self._stats["memory_hits"],
                    "disk_hits": self._stats["disk_hits"],
                },
                "storage": {
                    "total_files": total_files,
                    "compressed_files": self._stats["compressed_files"],
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "max_size_mb": round(self.max_cache_size_bytes / (1024 * 1024), 2),
                    "usage_percent": round(
                        total_size / self.max_cache_size_bytes * 100, 2
                    ),
                },
                "memory_cache": {
                    "items": memory_cache_size,
                    "size_bytes": memory_usage_bytes,
                    "size_mb": round(memory_usage_bytes / (1024 * 1024), 2),
                },
                "operations": {
                    "saves": self._stats["saves"],
                    "errors": self._stats["errors"],
                    "last_cleanup": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(self._stats["last_cleanup"])
                    ),
                },
                "configuration": {
                    "max_cache_size_mb": round(
                        self.max_cache_size_bytes / (1024 * 1024), 2
                    ),
                    "max_file_size_mb": round(
                        self.max_file_size_bytes / (1024 * 1024), 2
                    ),
                    "use_compression": self.use_compression,
                    "cleanup_interval_hours": round(self.cleanup_interval / 3600, 2),
                },
            }

    def cleanup_old_files(self, max_age_hours: int = 72) -> int:
        """
        Remove arquivos de cache antigos.

        Args:
            max_age_hours: Idade máxima em horas para manter arquivos

        Returns:
            Número de arquivos removidos
        """
        with self._cache_lock:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            files_removed = 0

            try:
                cache_files = list(self.cache_dir.glob("*.pkl")) + list(
                    self.cache_dir.glob("*.pkl.gz")
                )

                for file_path in cache_files:
                    if file_path.exists():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            files_removed += 1

                self._stats["last_cleanup"] = current_time

                if files_removed > 0:
                    self.logger.info(f"Cleanup: {files_removed} old files removed")

                return files_removed

            except Exception as e:
                self.logger.error(f"Erro durante cleanup: {e}")
                return 0

    def optimize_cache(self) -> Dict[str, int]:
        """
        Otimiza o cache removendo itens menos usados e comprimindo arquivos grandes.
        """
        with self._cache_lock:
            results = {
                "files_compressed": 0,
                "files_removed": 0,
                "space_saved_bytes": 0,
            }

            try:
                # 1. Comprime arquivos grandes não comprimidos
                normal_files = list(self.cache_dir.glob("*.pkl"))
                for file_path in normal_files:
                    if file_path.stat().st_size > self.max_file_size_bytes:
                        # Carrega, comprime e salva
                        try:
                            data = self._load_from_disk(file_path, False)
                            compressed_path = self._get_compressed_cache_file_path(
                                file_path.stem
                            )

                            original_size = file_path.stat().st_size
                            self._save_to_disk(compressed_path, data, True)
                            new_size = compressed_path.stat().st_size

                            # Remove arquivo original
                            file_path.unlink()

                            results["files_compressed"] += 1
                            results["space_saved_bytes"] += original_size - new_size

                        except Exception as e:
                            self.logger.error(f"Erro ao comprimir {file_path}: {e}")

                # 2. Remove arquivos LRU se o cache estiver muito grande
                if self._get_cache_size() > self.max_cache_size_bytes:
                    files_by_access = sorted(
                        self.cache_dir.glob("*.pkl*"), key=lambda f: f.stat().st_atime
                    )

                    current_size = self._get_cache_size()
                    target_size = (
                        self.max_cache_size_bytes * 0.8
                    )  # Remove até 80% do limite

                    for file_path in files_by_access:
                        if current_size <= target_size:
                            break

                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        current_size -= file_size
                        results["files_removed"] += 1
                        results["space_saved_bytes"] += file_size

                if any(results.values()):
                    self.logger.info(
                        f"Cache optimized: {results['files_compressed']} compressed, "
                        f"{results['files_removed']} removed, "
                        f"{results['space_saved_bytes']:,} bytes saved"
                    )

                return results

            except Exception as e:
                self.logger.error(f"Erro durante otimização: {e}")
                return results

    # Métodos privados

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Retorna caminho do arquivo de cache normal."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _get_compressed_cache_file_path(self, cache_key: str) -> Path:
        """Retorna caminho do arquivo de cache comprimido."""
        return self.cache_dir / f"{cache_key}.pkl.gz"

    def _load_from_disk(self, file_path: Path, is_compressed: bool) -> Any:
        """Carrega dados do disco com suporte a compressão."""
        if is_compressed:
            with gzip.open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def _save_to_disk(self, file_path: Path, data: Any, use_compression: bool) -> None:
        """Salva dados no disco com suporte a compressão."""
        if use_compression:
            with gzip.open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_to_memory_cache(self, cache_key: str, data: Any) -> None:
        """Adiciona item ao cache em memória com política LRU."""
        # Remove se já existe (para reordenar)
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        # Adiciona no final (mais recente)
        self._memory_cache[cache_key] = data

        # Limita tamanho do cache em memória (máximo 1000 itens)
        while len(self._memory_cache) > 1000:
            # Remove o mais antigo (primeiro item)
            self._memory_cache.popitem(last=False)

    def _should_cleanup_cache(self, new_data_size: int) -> bool:
        """Verifica se deve fazer cleanup antes de adicionar novo item."""
        current_size = self._get_cache_size()
        return (current_size + new_data_size) > self.max_cache_size_bytes

    def _get_cache_size(self) -> int:
        """Calcula tamanho total do cache em disco."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl")) + list(
                self.cache_dir.glob("*.pkl.gz")
            )
            return sum(f.stat().st_size for f in cache_files if f.exists())
        except Exception:
            return 0

    def _cleanup_old_files(self) -> None:
        """Cleanup automático de arquivos antigos."""
        self.cleanup_old_files(max_age_hours=72)  # 3 dias

    def _schedule_cleanup(self) -> None:
        """Agenda limpeza automática."""
        # Cleanup inicial se necessário
        if time.time() - self._stats["last_cleanup"] > self.cleanup_interval:
            self._cleanup_old_files()

    def _load_existing_cache_info(self) -> None:
        """Carrega informações de cache existente na inicialização."""
        try:
            # Conta arquivos existentes
            cache_files = list(self.cache_dir.glob("*.pkl"))
            compressed_files = list(self.cache_dir.glob("*.pkl.gz"))

            self._stats["compressed_files"] = len(compressed_files)

            if self.enable_statistics:
                self.logger.info(
                    f"Cache initialized: {len(cache_files)} normal files, "
                    f"{len(compressed_files)} compressed files"
                )

        except Exception as e:
            self.logger.error(f"Erro ao carregar informações do cache: {e}")
