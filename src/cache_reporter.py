"""
Cache Reporter - Sistema de relatórios e monitoramento avançado para o cache.

Este módulo implementa funcionalidades avançadas de relatórios incluindo:
- Dashboard em tempo real de estatísticas
- Relatórios de performance históricos
- Alertas de uso de cache
- Análise de padrões de acesso
- Recomendações de otimização
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import csv
import shutil


class CacheReporter:
    """
    Sistema de relatórios e monitoramento avançado para cache.

    Features:
    - Coleta contínua de métricas
    - Relatórios detalhados de performance
    - Alertas automáticos
    - Análise de tendências
    - Recomendações de otimização
    """

    def __init__(
        self,
        cache_manager,
        reports_dir: Path,
        monitoring_interval: int = 300,  # 5 minutos
        enable_continuous_monitoring: bool = True,
    ):
        self.cache_manager = cache_manager
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.monitoring_interval = monitoring_interval
        self.enable_continuous_monitoring = enable_continuous_monitoring

        # Histórico de métricas
        self._metrics_history = []
        self._monitoring_thread = None
        self._monitoring_active = False

        # Configurações de alertas
        self.alert_thresholds = {
            "hit_rate_min": 60.0,  # Hit rate mínimo aceitável (%)
            "cache_usage_max": 90.0,  # Uso máximo do cache (%)
            "error_rate_max": 5.0,  # Taxa máxima de erros (%)
            "memory_usage_max": 80.0,  # Uso máximo de memória (%)
            "files_count_max": 10000,  # Número máximo de arquivos
        }

        # Inicia monitoramento se habilitado
        if self.enable_continuous_monitoring:
            self.start_monitoring()

    def start_monitoring(self) -> None:
        """Inicia monitoramento contínuo do cache."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        print(f"Cache monitoring started (interval: {self.monitoring_interval}s)")

    def stop_monitoring(self) -> None:
        """Para monitoramento contínuo do cache."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        print("Cache monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Loop principal de monitoramento."""
        while self._monitoring_active:
            try:
                # Coleta métricas
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)

                # Mantém apenas as últimas 24 horas de métricas
                self._cleanup_old_metrics()

                # Verifica alertas
                self._check_alerts(metrics)

                # Aguarda próximo ciclo
                time.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"Erro no monitoramento de cache: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Coleta métricas atuais do cache."""
        stats = self.cache_manager.get_statistics()

        return {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "performance": stats["performance"],
            "storage": stats["storage"],
            "memory_cache": stats["memory_cache"],
            "operations": stats["operations"],
            "system": {
                "cache_dir_size": self._get_directory_size(
                    self.cache_manager.cache_dir
                ),
                "available_space": self._get_available_space(
                    self.cache_manager.cache_dir
                ),
            },
        }

    def _cleanup_old_metrics(self, max_age_hours: int = 24) -> None:
        """Remove métricas antigas do histórico."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        self._metrics_history = [
            m for m in self._metrics_history if m["timestamp"] > cutoff_time
        ]

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Verifica condições de alerta."""
        alerts = []

        # Verifica hit rate
        hit_rate = metrics["performance"]["hit_rate_percent"]
        if hit_rate < self.alert_thresholds["hit_rate_min"]:
            alerts.append(
                {
                    "type": "LOW_HIT_RATE",
                    "message": f"Hit rate baixo: {hit_rate:.1f}% (mínimo: {self.alert_thresholds['hit_rate_min']:.1f}%)",
                    "severity": "warning",
                    "timestamp": metrics["datetime"],
                }
            )

        # Verifica uso do cache
        cache_usage = metrics["storage"]["usage_percent"]
        if cache_usage > self.alert_thresholds["cache_usage_max"]:
            alerts.append(
                {
                    "type": "HIGH_CACHE_USAGE",
                    "message": f"Uso alto do cache: {cache_usage:.1f}% (máximo: {self.alert_thresholds['cache_usage_max']:.1f}%)",
                    "severity": "critical",
                    "timestamp": metrics["datetime"],
                }
            )

        # Verifica número de arquivos
        files_count = metrics["storage"]["total_files"]
        if files_count > self.alert_thresholds["files_count_max"]:
            alerts.append(
                {
                    "type": "TOO_MANY_FILES",
                    "message": f"Muitos arquivos no cache: {files_count:,} (máximo: {self.alert_thresholds['files_count_max']:,})",
                    "severity": "warning",
                    "timestamp": metrics["datetime"],
                }
            )

        # Salva alertas se houver
        if alerts:
            self._save_alerts(alerts)

    def _save_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Salva alertas em arquivo."""
        alerts_file = self.reports_dir / "cache_alerts.json"

        # Carrega alertas existentes
        existing_alerts = []
        if alerts_file.exists():
            try:
                with open(alerts_file, "r", encoding="utf-8") as f:
                    existing_alerts = json.load(f)
            except Exception:
                existing_alerts = []

        # Adiciona novos alertas
        existing_alerts.extend(alerts)

        # Mantém apenas alertas das últimas 48 horas
        cutoff_time = datetime.now() - timedelta(hours=48)
        filtered_alerts = []
        for alert in existing_alerts:
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time > cutoff_time:
                    filtered_alerts.append(alert)
            except Exception:
                # Se não conseguir parsear timestamp, mantém o alerta
                filtered_alerts.append(alert)

        # Salva alertas atualizados
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump(filtered_alerts, f, indent=2, ensure_ascii=False)

        # Log dos alertas
        for alert in alerts:
            print(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")

    def generate_performance_report(
        self, hours_back: int = 24, include_charts: bool = False
    ) -> Dict[str, Any]:
        """
        Gera relatório de performance das últimas N horas.

        Args:
            hours_back: Número de horas para incluir no relatório
            include_charts: Se deve incluir dados para gráficos
        """
        cutoff_time = time.time() - (hours_back * 3600)
        recent_metrics = [
            m for m in self._metrics_history if m["timestamp"] > cutoff_time
        ]

        if not recent_metrics:
            return {"error": "Não há dados suficientes para gerar relatório"}

        # Calcula estatísticas agregadas
        hit_rates = [m["performance"]["hit_rate_percent"] for m in recent_metrics]
        cache_usages = [m["storage"]["usage_percent"] for m in recent_metrics]
        file_counts = [m["storage"]["total_files"] for m in recent_metrics]

        report = {
            "period": {
                "hours": hours_back,
                "start": datetime.fromtimestamp(
                    recent_metrics[0]["timestamp"]
                ).isoformat(),
                "end": datetime.fromtimestamp(
                    recent_metrics[-1]["timestamp"]
                ).isoformat(),
                "data_points": len(recent_metrics),
            },
            "performance_summary": {
                "hit_rate": {
                    "average": sum(hit_rates) / len(hit_rates),
                    "min": min(hit_rates),
                    "max": max(hit_rates),
                    "current": hit_rates[-1] if hit_rates else 0,
                },
                "cache_usage": {
                    "average": sum(cache_usages) / len(cache_usages),
                    "min": min(cache_usages),
                    "max": max(cache_usages),
                    "current": cache_usages[-1] if cache_usages else 0,
                },
                "file_count": {
                    "average": sum(file_counts) / len(file_counts),
                    "min": min(file_counts),
                    "max": max(file_counts),
                    "current": file_counts[-1] if file_counts else 0,
                },
            },
            "trends": self._analyze_trends(recent_metrics),
            "recommendations": self._generate_recommendations(recent_metrics),
            "current_status": recent_metrics[-1] if recent_metrics else None,
        }

        if include_charts:
            report["chart_data"] = {
                "timestamps": [m["datetime"] for m in recent_metrics],
                "hit_rates": hit_rates,
                "cache_usages": cache_usages,
                "file_counts": file_counts,
            }

        return report

    def _analyze_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analisa tendências nos dados de performance."""
        if len(metrics) < 2:
            return {"error": "Dados insuficientes para análise de tendências"}

        # Analisa hit rate
        hit_rates = [m["performance"]["hit_rate_percent"] for m in metrics]
        hit_rate_trend = "stable"
        if len(hit_rates) >= 3:
            recent_avg = sum(hit_rates[-3:]) / 3
            older_avg = sum(hit_rates[:3]) / 3
            if recent_avg > older_avg + 5:
                hit_rate_trend = "improving"
            elif recent_avg < older_avg - 5:
                hit_rate_trend = "declining"

        # Analisa uso do cache
        usages = [m["storage"]["usage_percent"] for m in metrics]
        usage_trend = "stable"
        if len(usages) >= 3:
            recent_avg = sum(usages[-3:]) / 3
            older_avg = sum(usages[:3]) / 3
            if recent_avg > older_avg + 10:
                usage_trend = "increasing"
            elif recent_avg < older_avg - 10:
                usage_trend = "decreasing"

        return {
            "hit_rate": hit_rate_trend,
            "cache_usage": usage_trend,
            "overall_health": (
                "good"
                if hit_rate_trend != "declining" and usage_trend != "increasing"
                else "needs_attention"
            ),
        }

    def _generate_recommendations(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Gera recomendações baseadas nas métricas."""
        if not metrics:
            return []

        recommendations = []
        latest = metrics[-1]

        # Recomendações de hit rate
        hit_rate = latest["performance"]["hit_rate_percent"]
        if hit_rate < 50:
            recommendations.append(
                "Hit rate muito baixo. Considere ajustar estratégia de caching ou revisar chaves de cache."
            )
        elif hit_rate < 70:
            recommendations.append(
                "Hit rate abaixo do ideal. Analise padrões de acesso e otimize política de cache."
            )

        # Recomendações de uso
        usage = latest["storage"]["usage_percent"]
        if usage > 85:
            recommendations.append(
                "Cache quase cheio. Execute limpeza ou aumente limite de tamanho."
            )
        elif usage > 70:
            recommendations.append(
                "Uso moderadamente alto. Monitore crescimento e planeje limpeza preventiva."
            )

        # Recomendações de arquivos
        files = latest["storage"]["total_files"]
        if files > 5000:
            recommendations.append(
                f"Muitos arquivos no cache ({files:,}). Considere compressão ou limpeza de arquivos antigos."
            )

        # Recomendações de compressão
        compressed_ratio = latest["storage"]["compressed_files"] / max(
            latest["storage"]["total_files"], 1
        )
        if compressed_ratio < 0.1 and files > 100:
            recommendations.append(
                "Baixa taxa de compressão. Execute otimização para comprimir arquivos grandes."
            )

        if not recommendations:
            recommendations.append("Sistema funcionando bem. Continue monitorando.")

        return recommendations

    def export_metrics_csv(
        self, filename: Optional[str] = None, hours_back: int = 24
    ) -> Path:
        """Exporta métricas para arquivo CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cache_metrics_{timestamp}.csv"

        output_file = self.reports_dir / filename

        # Filtra métricas pelo período
        cutoff_time = time.time() - (hours_back * 3600)
        filtered_metrics = [
            m for m in self._metrics_history if m["timestamp"] > cutoff_time
        ]

        if not filtered_metrics:
            raise ValueError("Não há dados para exportar")

        # Escreve CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "datetime",
                "hit_rate_percent",
                "total_requests",
                "hits",
                "misses",
                "cache_usage_percent",
                "total_files",
                "compressed_files",
                "memory_items",
                "memory_size_mb",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for metric in filtered_metrics:
                writer.writerow(
                    {
                        "datetime": metric["datetime"],
                        "hit_rate_percent": metric["performance"]["hit_rate_percent"],
                        "total_requests": metric["performance"]["hits"]
                        + metric["performance"]["misses"],
                        "hits": metric["performance"]["hits"],
                        "misses": metric["performance"]["misses"],
                        "cache_usage_percent": metric["storage"]["usage_percent"],
                        "total_files": metric["storage"]["total_files"],
                        "compressed_files": metric["storage"]["compressed_files"],
                        "memory_items": metric["memory_cache"]["items"],
                        "memory_size_mb": metric["memory_cache"]["size_mb"],
                    }
                )

        return output_file

    def get_cache_health_score(self) -> Dict[str, Any]:
        """Calcula score de saúde do cache (0-100)."""
        if not self._metrics_history:
            return {"score": 0, "status": "no_data", "factors": []}

        latest = self._metrics_history[-1]
        factors = []
        total_score = 0
        max_score = 0

        # Fator: Hit Rate (peso 30)
        hit_rate = latest["performance"]["hit_rate_percent"]
        hit_score = min(hit_rate / 80 * 30, 30)  # 80% = score máximo
        total_score += hit_score
        max_score += 30
        factors.append(
            {
                "name": "Hit Rate",
                "value": hit_rate,
                "score": hit_score,
                "max_score": 30,
                "status": (
                    "good"
                    if hit_rate >= 70
                    else "warning" if hit_rate >= 50 else "critical"
                ),
            }
        )

        # Fator: Uso do Cache (peso 25)
        usage = latest["storage"]["usage_percent"]
        usage_score = max(
            0, min((100 - usage) / 20 * 25, 25)
        )  # Melhor quando uso é menor
        total_score += usage_score
        max_score += 25
        factors.append(
            {
                "name": "Cache Usage",
                "value": usage,
                "score": usage_score,
                "max_score": 25,
                "status": (
                    "good" if usage <= 70 else "warning" if usage <= 85 else "critical"
                ),
            }
        )

        # Fator: Taxa de Erro (peso 20)
        total_ops = latest["operations"]["saves"] + latest["operations"]["errors"]
        error_rate = (latest["operations"]["errors"] / max(total_ops, 1)) * 100
        error_score = max(
            0, min((5 - error_rate) / 5 * 20, 20)
        )  # Melhor quando erro é menor
        total_score += error_score
        max_score += 20
        factors.append(
            {
                "name": "Error Rate",
                "value": error_rate,
                "score": error_score,
                "max_score": 20,
                "status": (
                    "good"
                    if error_rate <= 1
                    else "warning" if error_rate <= 3 else "critical"
                ),
            }
        )

        # Fator: Organização (peso 15)
        compressed_ratio = latest["storage"]["compressed_files"] / max(
            latest["storage"]["total_files"], 1
        )
        organization_score = min(compressed_ratio * 15, 15)
        total_score += organization_score
        max_score += 15
        factors.append(
            {
                "name": "Organization",
                "value": compressed_ratio * 100,
                "score": organization_score,
                "max_score": 15,
                "status": (
                    "good"
                    if compressed_ratio >= 0.3
                    else "warning" if compressed_ratio >= 0.1 else "poor"
                ),
            }
        )

        # Fator: Estabilidade (peso 10)
        stability_score = 10  # Base score, reduzido por problemas
        if len(self._metrics_history) >= 5:
            recent_hit_rates = [
                m["performance"]["hit_rate_percent"] for m in self._metrics_history[-5:]
            ]
            variance = sum(
                (x - sum(recent_hit_rates) / len(recent_hit_rates)) ** 2
                for x in recent_hit_rates
            ) / len(recent_hit_rates)
            if variance > 100:  # Alta variância
                stability_score = 5
        total_score += stability_score
        max_score += 10
        factors.append(
            {
                "name": "Stability",
                "value": stability_score,
                "score": stability_score,
                "max_score": 10,
                "status": "good" if stability_score >= 8 else "warning",
            }
        )

        # Score final
        final_score = (total_score / max_score) * 100

        # Status geral
        if final_score >= 80:
            status = "excellent"
        elif final_score >= 60:
            status = "good"
        elif final_score >= 40:
            status = "fair"
        else:
            status = "poor"

        return {
            "score": round(final_score, 1),
            "status": status,
            "factors": factors,
            "timestamp": latest["datetime"],
        }

    def _get_directory_size(self, directory: Path) -> int:
        """Calcula tamanho total do diretório em bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def _get_available_space(self, directory: Path) -> int:
        """Retorna espaço disponível no sistema de arquivos."""
        try:
            return shutil.disk_usage(directory).free
        except Exception:
            return 0

    def __del__(self):
        """Cleanup quando objeto é destruído."""
        self.stop_monitoring()
