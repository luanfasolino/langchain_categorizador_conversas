"""
Testes para o CacheReporter - Sistema de relatórios e monitoramento de cache.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

from src.cache_manager import CacheManager
from src.cache_reporter import CacheReporter


class TestCacheReporter:
    """Testes para a classe CacheReporter."""

    @pytest.fixture
    def temp_dirs(self):
        """Cria diretórios temporários para testes."""
        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as reports_dir:
            yield Path(cache_dir), Path(reports_dir)

    @pytest.fixture
    def cache_manager(self, temp_dirs):
        """Cria instância do CacheManager para testes."""
        cache_dir, _ = temp_dirs
        return CacheManager(
            cache_dir=cache_dir,
            max_cache_size_mb=10,
            use_compression=True,
            enable_statistics=True,
        )

    @pytest.fixture
    def cache_reporter(self, cache_manager, temp_dirs):
        """Cria instância do CacheReporter para testes."""
        _, reports_dir = temp_dirs
        return CacheReporter(
            cache_manager=cache_manager,
            reports_dir=reports_dir,
            monitoring_interval=1,  # 1 segundo para testes rápidos
            enable_continuous_monitoring=False,  # Desabilitado por padrão nos testes
        )

    def test_cache_reporter_initialization(self, cache_reporter, temp_dirs):
        """Testa inicialização do CacheReporter."""
        _, reports_dir = temp_dirs

        assert cache_reporter.reports_dir == reports_dir
        assert cache_reporter.monitoring_interval == 1
        assert cache_reporter.enable_continuous_monitoring is False
        assert cache_reporter._monitoring_active is False
        assert isinstance(cache_reporter.alert_thresholds, dict)

    def test_metrics_collection(self, cache_reporter, cache_manager):
        """Testa coleta de métricas."""
        # Adiciona alguns dados para gerar métricas
        for i in range(5):
            cache_manager.set(f"key_{i}", f"data_{i}")

        # Acessa alguns itens para gerar hits
        for i in range(3):
            cache_manager.get(f"key_{i}")

        # Coleta métricas
        metrics = cache_reporter._collect_metrics()

        # Verifica estrutura das métricas
        assert "timestamp" in metrics
        assert "datetime" in metrics
        assert "performance" in metrics
        assert "storage" in metrics
        assert "memory_cache" in metrics
        assert "operations" in metrics
        assert "system" in metrics

        # Verifica valores
        assert metrics["performance"]["hits"] >= 3
        assert metrics["operations"]["saves"] >= 5
        assert metrics["storage"]["total_files"] >= 5

    def test_continuous_monitoring(self, cache_manager, temp_dirs):
        """Testa monitoramento contínuo."""
        _, reports_dir = temp_dirs

        # Cria reporter com monitoramento habilitado
        reporter = CacheReporter(
            cache_manager=cache_manager,
            reports_dir=reports_dir,
            monitoring_interval=0.1,  # 100ms para teste rápido
            enable_continuous_monitoring=True,
        )

        try:
            # Aguarda algumas coletas
            time.sleep(0.5)

            assert reporter._monitoring_active is True
            assert len(reporter._metrics_history) > 0

            # Para monitoramento
            reporter.stop_monitoring()
            assert reporter._monitoring_active is False

        finally:
            reporter.stop_monitoring()

    def test_performance_report_generation(self, cache_reporter, cache_manager):
        """Testa geração de relatórios de performance."""
        # Simula algumas operações de cache
        for i in range(10):
            cache_manager.set(f"perf_key_{i}", f"perf_data_{i}")
            cache_manager.get(f"perf_key_{i}")

        # Adiciona métricas simuladas ao histórico
        current_time = time.time()
        for i in range(5):
            metrics = {
                "timestamp": current_time - (i * 3600),  # 1 hora atrás cada
                "datetime": f"2025-01-01T{12+i}:00:00",
                "performance": {
                    "hit_rate_percent": 75.0 + i,
                    "hits": 100 + i * 10,
                    "misses": 25 - i,
                },
                "storage": {
                    "usage_percent": 50.0 + i * 5,
                    "total_files": 100 + i * 20,
                    "compressed_files": 10 + i * 2,
                },
                "memory_cache": {"items": 50 + i * 5},
                "operations": {"saves": 150 + i * 10},
            }
            cache_reporter._metrics_history.append(metrics)

        # Gera relatório
        report = cache_reporter.generate_performance_report(hours_back=6)

        assert "period" in report
        assert "performance_summary" in report
        assert "trends" in report
        assert "recommendations" in report
        assert "current_status" in report

        # Verifica dados do período
        assert report["period"]["hours"] == 6
        assert report["period"]["data_points"] == 5

        # Verifica resumo de performance
        assert "hit_rate" in report["performance_summary"]
        assert "cache_usage" in report["performance_summary"]
        assert "file_count" in report["performance_summary"]

    def test_trend_analysis(self, cache_reporter):
        """Testa análise de tendências."""
        # Métricas simuladas com tendência de melhoria
        improving_metrics = []
        for i in range(6):
            metrics = {
                "performance": {"hit_rate_percent": 50.0 + i * 5},
                "storage": {"usage_percent": 80.0 - i * 2},
            }
            improving_metrics.append(metrics)

        trends = cache_reporter._analyze_trends(improving_metrics)

        assert "hit_rate" in trends
        assert "cache_usage" in trends
        assert "overall_health" in trends

    def test_recommendations_generation(self, cache_reporter):
        """Testa geração de recomendações."""
        # Métricas com problemas para gerar recomendações
        problematic_metrics = [
            {
                "performance": {"hit_rate_percent": 40.0},  # Baixo hit rate
                "storage": {
                    "usage_percent": 90.0,  # Alto uso
                    "total_files": 6000,  # Muitos arquivos
                    "compressed_files": 100,  # Baixa compressão
                },
            }
        ]

        recommendations = cache_reporter._generate_recommendations(problematic_metrics)

        assert len(recommendations) > 0
        assert any("hit rate" in rec.lower() for rec in recommendations)
        assert any("cache" in rec.lower() for rec in recommendations)

    def test_alert_checking(self, cache_reporter, temp_dirs):
        """Testa verificação de alertas."""
        _, reports_dir = temp_dirs

        # Métricas que devem gerar alertas (usa timestamp atual)
        from datetime import datetime

        current_time = datetime.now().isoformat()

        problematic_metrics = {
            "datetime": current_time,
            "performance": {"hit_rate_percent": 30.0},  # Abaixo do threshold
            "storage": {
                "usage_percent": 95.0,  # Acima do threshold
                "total_files": 15000,  # Acima do threshold
            },
        }

        # Verifica alertas
        cache_reporter._check_alerts(problematic_metrics)

        # Verifica se arquivo de alertas foi criado
        alerts_file = reports_dir / "cache_alerts.json"
        assert (
            alerts_file.exists()
        ), f"Arquivo de alertas não foi criado em {alerts_file}"

        # Lê alertas
        import json

        with open(alerts_file, "r", encoding="utf-8") as f:
            alerts = json.load(f)

        assert (
            len(alerts) >= 2
        ), f"Esperados pelo menos 2 alertas, encontrados: {len(alerts)}"
        alert_types = [alert["type"] for alert in alerts]
        assert "LOW_HIT_RATE" in alert_types
        assert "HIGH_CACHE_USAGE" in alert_types

    def test_csv_export(self, cache_reporter, temp_dirs):
        """Testa exportação para CSV."""
        _, reports_dir = temp_dirs

        # Adiciona métricas simuladas
        current_time = time.time()
        for i in range(3):
            metrics = {
                "timestamp": current_time - (i * 3600),
                "datetime": f"2025-01-01T{12+i}:00:00",
                "performance": {
                    "hit_rate_percent": 70.0 + i,
                    "hits": 80 + i * 10,
                    "misses": 20 - i,
                },
                "storage": {
                    "usage_percent": 60.0 + i * 5,
                    "total_files": 200 + i * 50,
                    "compressed_files": 20 + i * 5,
                },
                "memory_cache": {"items": 100 + i * 10, "size_mb": 5.0 + i},
            }
            cache_reporter._metrics_history.append(metrics)

        # Exporta para CSV
        csv_file = cache_reporter.export_metrics_csv(
            filename="test_export.csv", hours_back=4
        )

        assert csv_file.exists()
        assert csv_file.suffix == ".csv"

        # Verifica conteúdo do CSV
        import csv

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert "datetime" in rows[0]
        assert "hit_rate_percent" in rows[0]
        assert "cache_usage_percent" in rows[0]

    def test_cache_health_score(self, cache_reporter, cache_manager):
        """Testa cálculo do score de saúde do cache."""
        # Adiciona métricas simuladas
        metrics = {
            "datetime": "2025-01-01T12:00:00",
            "performance": {"hit_rate_percent": 85.0},
            "storage": {
                "usage_percent": 60.0,
                "total_files": 1000,
                "compressed_files": 300,
            },
            "operations": {"saves": 1000, "errors": 10},
        }
        cache_reporter._metrics_history.append(metrics)

        # Calcula score de saúde
        health = cache_reporter.get_cache_health_score()

        assert "score" in health
        assert "status" in health
        assert "factors" in health
        assert "timestamp" in health

        assert 0 <= health["score"] <= 100
        assert health["status"] in ["excellent", "good", "fair", "poor"]
        assert len(health["factors"]) == 5  # 5 fatores avaliados

        # Verifica fatores
        factor_names = [f["name"] for f in health["factors"]]
        assert "Hit Rate" in factor_names
        assert "Cache Usage" in factor_names
        assert "Error Rate" in factor_names
        assert "Organization" in factor_names
        assert "Stability" in factor_names

    def test_metrics_history_cleanup(self, cache_reporter):
        """Testa limpeza do histórico de métricas."""
        # Adiciona métricas antigas e recentes
        current_time = time.time()

        # Métricas antigas (mais de 24 horas)
        for i in range(3):
            old_metrics = {
                "timestamp": current_time - (25 + i) * 3600,  # 25+ horas atrás
                "datetime": f"2025-01-01T{i}:00:00",
            }
            cache_reporter._metrics_history.append(old_metrics)

        # Métricas recentes
        for i in range(3):
            recent_metrics = {
                "timestamp": current_time - i * 3600,  # Últimas 3 horas
                "datetime": f"2025-01-02T{20+i}:00:00",
            }
            cache_reporter._metrics_history.append(recent_metrics)

        assert len(cache_reporter._metrics_history) == 6

        # Executa limpeza
        cache_reporter._cleanup_old_metrics(max_age_hours=24)

        # Deve manter apenas métricas recentes
        assert len(cache_reporter._metrics_history) == 3

    def test_error_handling_in_monitoring(self, cache_manager, temp_dirs):
        """Testa tratamento de erros no monitoramento."""
        _, reports_dir = temp_dirs

        # Mock do cache_manager para gerar erro
        cache_manager_mock = Mock()
        cache_manager_mock.get_statistics.side_effect = Exception("Test error")

        reporter = CacheReporter(
            cache_manager=cache_manager_mock,
            reports_dir=reports_dir,
            monitoring_interval=0.1,
            enable_continuous_monitoring=True,
        )

        try:
            # Aguarda algumas tentativas de coleta
            time.sleep(0.3)

            # Monitoramento deve continuar funcionando mesmo com erros
            assert reporter._monitoring_active is True

        finally:
            reporter.stop_monitoring()

    def test_custom_alert_thresholds(self, cache_manager, temp_dirs):
        """Testa configuração de thresholds customizados."""
        _, reports_dir = temp_dirs

        reporter = CacheReporter(
            cache_manager=cache_manager,
            reports_dir=reports_dir,
            enable_continuous_monitoring=False,
        )

        # Modifica thresholds
        reporter.alert_thresholds["hit_rate_min"] = 80.0
        reporter.alert_thresholds["cache_usage_max"] = 70.0

        # Métricas que devem gerar alerta com novos thresholds
        metrics = {
            "datetime": "2025-01-01T12:00:00",
            "performance": {"hit_rate_percent": 75.0},  # Abaixo do novo threshold
            "storage": {
                "usage_percent": 75.0,  # Acima do novo threshold
                "total_files": 1000,
            },
        }

        reporter._check_alerts(metrics)

        # Verifica se alertas foram gerados
        alerts_file = reports_dir / "cache_alerts.json"
        assert alerts_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])
