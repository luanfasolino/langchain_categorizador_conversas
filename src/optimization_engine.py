"""
Engine de recomendações de otimização baseado em IA.

Analisa padrões de processamento e gera recomendações acionáveis para
redução de custos e melhoria de eficiência operacional.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import math


class OptimizationCategory(Enum):
    """Categorias de otimização disponíveis."""

    COST_REDUCTION = "cost_reduction"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    CONFIGURATION_TUNING = "configuration_tuning"
    DATA_PREPROCESSING = "data_preprocessing"


class ImpactLevel(Enum):
    """Níveis de impacto das recomendações."""

    LOW = "low"  # <10% improvement
    MEDIUM = "medium"  # 10-30% improvement
    HIGH = "high"  # 30-50% improvement
    CRITICAL = "critical"  # >50% improvement


class ImplementationComplexity(Enum):
    """Complexidade de implementação."""

    EASY = "easy"  # Mudança de configuração
    MODERATE = "moderate"  # Algumas modificações de código
    COMPLEX = "complex"  # Reestruturação significativa


@dataclass
class OptimizationRecommendation:
    """Representa uma recomendação de otimização."""

    id: str
    category: OptimizationCategory
    title: str
    description: str
    impact_level: ImpactLevel
    complexity: ImplementationComplexity
    estimated_savings_usd: float
    estimated_time_savings_hours: float
    confidence_score: float  # 0.0 to 1.0
    implementation_steps: List[str]
    risks: List[str]
    prerequisites: List[str]
    metrics_to_track: List[str]
    supporting_data: Dict[str, Any]
    priority_score: float = 0.0

    def __post_init__(self):
        """Calcula score de prioridade após inicialização."""
        self.priority_score = self._calculate_priority_score()

    def _calculate_priority_score(self) -> float:
        """Calcula score de prioridade baseado em múltiplos fatores."""
        # Componentes do score
        impact_weights = {
            ImpactLevel.LOW: 1.0,
            ImpactLevel.MEDIUM: 2.5,
            ImpactLevel.HIGH: 4.0,
            ImpactLevel.CRITICAL: 6.0,
        }

        complexity_weights = {
            ImplementationComplexity.EASY: 3.0,
            ImplementationComplexity.MODERATE: 2.0,
            ImplementationComplexity.COMPLEX: 1.0,
        }

        # Score base por impacto e complexidade
        impact_score = impact_weights.get(self.impact_level, 1.0)
        complexity_score = complexity_weights.get(self.complexity, 1.0)

        # Bonus por economia financeira
        savings_bonus = min(self.estimated_savings_usd / 100, 2.0)  # Max 2.0 bonus

        # Penalty por riscos
        risk_penalty = len(self.risks) * 0.2

        # Score final
        priority = (
            (impact_score * complexity_score * self.confidence_score)
            + savings_bonus
            - risk_penalty
        )

        return max(0.0, priority)


class OptimizationEngine:
    """
    Engine principal de análise e recomendações de otimização.

    Utiliza dados históricos e padrões de uso para gerar recomendações
    acionáveis de otimização de custos e performance.
    """

    def __init__(self, storage_dir: Path = None):
        """
        Inicializa a engine de otimização.

        Args:
            storage_dir: Diretório para armazenar análises e recomendações
        """
        self.storage_dir = storage_dir or Path("database/optimization")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Cache de análises
        self._analysis_cache = {}
        self._recommendations_cache = {}

        # Thresholds para análise
        self.efficiency_thresholds = {
            "cost_per_item_warning": 0.10,  # $0.10 per item
            "cost_per_item_critical": 0.20,  # $0.20 per item
            "tokens_per_second_min": 50,  # 50 tokens/s minimum
            "hit_rate_min": 0.75,  # 75% cache hit rate
            "parallel_efficiency_min": 0.8,  # 80% parallel efficiency
        }

    def analyze_performance_patterns(
        self,
        sessions_data: List[Dict],
        operation_breakdown: Dict = None,
        cache_stats: Dict = None,
    ) -> Dict[str, Any]:
        """
        Analisa padrões de performance históricos.

        Args:
            sessions_data: Dados de sessões históricas
            operation_breakdown: Breakdown de operações por tipo
            cache_stats: Estatísticas de cache

        Returns:
            Análise detalhada de padrões
        """
        if not sessions_data:
            return {"error": "No sessions data available"}

        # Análise temporal
        temporal_analysis = self._analyze_temporal_patterns(sessions_data)

        # Análise de eficiência
        efficiency_analysis = self._analyze_efficiency_patterns(sessions_data)

        # Análise de recursos
        resource_analysis = self._analyze_resource_utilization(sessions_data)

        # Análise de operações
        operation_analysis = self._analyze_operation_patterns(operation_breakdown or {})

        # Análise de cache
        cache_analysis = self._analyze_cache_patterns(cache_stats or {})

        # Identifica anomalias
        anomalies = self._detect_performance_anomalies(sessions_data)

        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "sessions_analyzed": len(sessions_data),
            "temporal_patterns": temporal_analysis,
            "efficiency_patterns": efficiency_analysis,
            "resource_patterns": resource_analysis,
            "operation_patterns": operation_analysis,
            "cache_patterns": cache_analysis,
            "anomalies": anomalies,
            "overall_health_score": self._calculate_overall_health_score(
                efficiency_analysis, resource_analysis, cache_analysis
            ),
        }

        return analysis

    def _analyze_temporal_patterns(self, sessions_data: List[Dict]) -> Dict[str, Any]:
        """Analisa padrões temporais de uso."""
        if not sessions_data:
            return {}

        # Extrai métricas temporais
        durations = []
        throughputs = []
        costs_per_hour = []

        for session in sessions_data:
            duration = session.get("duration_seconds", 0) / 3600  # em horas
            if duration > 0:
                durations.append(duration)

                total_tokens = session.get("total_input_tokens", 0) + session.get(
                    "total_output_tokens", 0
                )
                throughput = total_tokens / duration if duration > 0 else 0
                throughputs.append(throughput)

                cost_per_hour = session.get("total_cost_usd", 0) / duration
                costs_per_hour.append(cost_per_hour)

        if not durations:
            return {"error": "No valid temporal data"}

        return {
            "average_session_duration_hours": statistics.mean(durations),
            "median_session_duration_hours": statistics.median(durations),
            "duration_variance": (
                statistics.variance(durations) if len(durations) > 1 else 0
            ),
            "average_throughput_tokens_per_hour": statistics.mean(throughputs),
            "throughput_consistency": (
                1.0 - (statistics.stdev(throughputs) / statistics.mean(throughputs))
                if len(throughputs) > 1 and statistics.mean(throughputs) > 0
                else 0
            ),
            "average_cost_per_hour": statistics.mean(costs_per_hour),
            "cost_stability": (
                1.0
                - (statistics.stdev(costs_per_hour) / statistics.mean(costs_per_hour))
                if len(costs_per_hour) > 1 and statistics.mean(costs_per_hour) > 0
                else 0
            ),
        }

    def _analyze_efficiency_patterns(self, sessions_data: List[Dict]) -> Dict[str, Any]:
        """Analisa padrões de eficiência."""
        costs_per_item = []
        tokens_per_item = []
        tokens_per_second = []

        for session in sessions_data:
            dataset_size = session.get("dataset_size", 0)
            if dataset_size > 0:
                cost_per_item = session.get("total_cost_usd", 0) / dataset_size
                costs_per_item.append(cost_per_item)

                total_tokens = session.get("total_input_tokens", 0) + session.get(
                    "total_output_tokens", 0
                )
                tokens_per_item.append(total_tokens / dataset_size)

            duration = session.get("duration_seconds", 0)
            if duration > 0:
                total_tokens = session.get("total_input_tokens", 0) + session.get(
                    "total_output_tokens", 0
                )
                tokens_per_second.append(total_tokens / duration)

        if not costs_per_item:
            return {"error": "No valid efficiency data"}

        avg_cost_per_item = statistics.mean(costs_per_item)
        avg_tokens_per_second = (
            statistics.mean(tokens_per_second) if tokens_per_second else 0
        )

        return {
            "average_cost_per_item": avg_cost_per_item,
            "median_cost_per_item": statistics.median(costs_per_item),
            "cost_per_item_trend": self._calculate_trend(costs_per_item),
            "average_tokens_per_item": (
                statistics.mean(tokens_per_item) if tokens_per_item else 0
            ),
            "average_tokens_per_second": avg_tokens_per_second,
            "efficiency_score": self._calculate_efficiency_score(
                avg_cost_per_item, avg_tokens_per_second
            ),
            "cost_consistency": (
                1.0 - (statistics.stdev(costs_per_item) / avg_cost_per_item)
                if len(costs_per_item) > 1 and avg_cost_per_item > 0
                else 0
            ),
        }

    def _analyze_resource_utilization(
        self, sessions_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analisa utilização de recursos."""
        parallel_workers = []
        operations_counts = []

        for session in sessions_data:
            # Extrai informações de workers (se disponível)
            workers = session.get("parallel_workers", 4)  # Assume padrão de 4
            parallel_workers.append(workers)

            operations = session.get("operations_count", 0)
            operations_counts.append(operations)

        avg_workers = statistics.mean(parallel_workers) if parallel_workers else 4
        total_operations = sum(operations_counts)

        return {
            "average_parallel_workers": avg_workers,
            "total_operations_processed": total_operations,
            "resource_utilization_score": self._calculate_resource_utilization_score(
                avg_workers
            ),
            "operations_distribution": self._analyze_operations_distribution(
                operations_counts
            ),
        }

    def _analyze_operation_patterns(self, operation_breakdown: Dict) -> Dict[str, Any]:
        """Analisa padrões de operações."""
        if not operation_breakdown:
            return {"message": "No operation data available"}

        # Calcula eficiência por tipo de operação
        operation_efficiencies = {}
        total_cost = 0
        total_operations = 0

        for op_type, stats in operation_breakdown.items():
            operations = stats.get("operations", 0)
            cost = stats.get("cost_usd", 0)
            total_operations += operations
            total_cost += cost

            if operations > 0:
                cost_per_operation = cost / operations
                tokens_per_operation = stats.get("total_tokens", 0) / operations

                operation_efficiencies[op_type] = {
                    "cost_per_operation": cost_per_operation,
                    "tokens_per_operation": tokens_per_operation,
                    "operations_count": operations,
                    "percentage_of_total": (operations / max(total_operations, 1))
                    * 100,
                }

        # Identifica operações mais caras
        most_expensive = (
            max(
                operation_efficiencies.items(), key=lambda x: x[1]["cost_per_operation"]
            )
            if operation_efficiencies
            else None
        )

        return {
            "operation_efficiencies": operation_efficiencies,
            "most_expensive_operation": most_expensive[0] if most_expensive else None,
            "total_operations": total_operations,
            "average_cost_per_operation": total_cost / max(total_operations, 1),
        }

    def _analyze_cache_patterns(self, cache_stats: Dict) -> Dict[str, Any]:
        """Analisa padrões de cache."""
        if not cache_stats:
            return {"message": "No cache data available"}

        performance = cache_stats.get("performance", {})
        storage = cache_stats.get("storage", {})

        hit_rate = performance.get("hit_rate_percent", 0) / 100
        cache_usage = storage.get("usage_percent", 0) / 100

        cache_efficiency_score = (hit_rate * 0.7) + (
            (1 - cache_usage) * 0.3
        )  # Prefer high hit rate, moderate usage

        return {
            "hit_rate": hit_rate,
            "cache_usage": cache_usage,
            "cache_efficiency_score": cache_efficiency_score,
            "total_files": storage.get("total_files", 0),
            "compressed_files": storage.get("compressed_files", 0),
            "compression_ratio": storage.get("compressed_files", 0)
            / max(storage.get("total_files", 1), 1),
        }

    def _detect_performance_anomalies(
        self, sessions_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Detecta anomalias de performance."""
        anomalies = []

        if len(sessions_data) < 3:
            return anomalies

        # Analisa custos por item
        costs_per_item = []
        for session in sessions_data:
            dataset_size = session.get("dataset_size", 0)
            if dataset_size > 0:
                cost_per_item = session.get("total_cost_usd", 0) / dataset_size
                costs_per_item.append((cost_per_item, session))

        if len(costs_per_item) > 2:
            mean_cost = statistics.mean(cost for cost, _ in costs_per_item)
            stdev_cost = statistics.stdev(cost for cost, _ in costs_per_item)

            # Detecta outliers (>2 desvios padrão)
            for cost, session in costs_per_item:
                if abs(cost - mean_cost) > 2 * stdev_cost:
                    anomalies.append(
                        {
                            "type": "cost_anomaly",
                            "session_id": session.get("session_id", "unknown"),
                            "cost_per_item": cost,
                            "expected_range": (
                                mean_cost - stdev_cost,
                                mean_cost + stdev_cost,
                            ),
                            "severity": (
                                "high"
                                if abs(cost - mean_cost) > 3 * stdev_cost
                                else "medium"
                            ),
                        }
                    )

        return anomalies

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência de uma lista de valores."""
        if len(values) < 3:
            return "insufficient_data"

        # Análise de tendência simples
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        change_percent = (
            ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        )

        if change_percent < -5:
            return "improving"
        elif change_percent > 5:
            return "deteriorating"
        else:
            return "stable"

    def _calculate_efficiency_score(
        self, cost_per_item: float, tokens_per_second: float
    ) -> float:
        """Calcula score de eficiência geral."""
        # Score baseado em cost_per_item (lower is better)
        cost_score = 1.0 / (1.0 + cost_per_item * 100)  # Normalize

        # Score baseado em throughput (higher is better)
        throughput_score = min(
            tokens_per_second / 200, 1.0
        )  # Normalize para 200 tokens/s

        return (cost_score * 0.6) + (throughput_score * 0.4)

    def _calculate_resource_utilization_score(self, avg_workers: float) -> float:
        """Calcula score de utilização de recursos."""
        # Score ótimo em torno de 4-6 workers
        optimal_workers = 4
        if avg_workers <= optimal_workers:
            return avg_workers / optimal_workers
        else:
            return max(0.5, 1.0 - ((avg_workers - optimal_workers) / 10))

    def _analyze_operations_distribution(
        self, operations_counts: List[int]
    ) -> Dict[str, Any]:
        """Analisa distribuição de operações."""
        if not operations_counts:
            return {}

        return {
            "total_operations": sum(operations_counts),
            "average_operations_per_session": statistics.mean(operations_counts),
            "operations_variance": (
                statistics.variance(operations_counts)
                if len(operations_counts) > 1
                else 0
            ),
            "max_operations_session": max(operations_counts),
            "min_operations_session": min(operations_counts),
        }

    def _calculate_overall_health_score(
        self, efficiency_analysis: Dict, resource_analysis: Dict, cache_analysis: Dict
    ) -> float:
        """Calcula score geral de saúde do sistema."""
        efficiency_score = efficiency_analysis.get("efficiency_score", 0.5)
        resource_score = resource_analysis.get("resource_utilization_score", 0.5)
        cache_score = cache_analysis.get("cache_efficiency_score", 0.5)

        # Média ponderada
        weights = [0.4, 0.3, 0.3]  # efficiency, resource, cache
        scores = [efficiency_score, resource_score, cache_score]

        return sum(w * s for w, s in zip(weights, scores))

    def generate_optimization_recommendations(
        self, analysis: Dict[str, Any], current_config: Dict = None
    ) -> List[OptimizationRecommendation]:
        """
        Gera recomendações de otimização baseadas na análise.

        Args:
            analysis: Análise de padrões de performance
            current_config: Configuração atual do sistema

        Returns:
            Lista de recomendações ordenadas por prioridade
        """
        recommendations = []

        # Recomendações baseadas em eficiência
        recommendations.extend(self._generate_efficiency_recommendations(analysis))

        # Recomendações baseadas em recursos
        recommendations.extend(self._generate_resource_recommendations(analysis))

        # Recomendações baseadas em cache
        recommendations.extend(self._generate_cache_recommendations(analysis))

        # Recomendações baseadas em operações
        recommendations.extend(self._generate_operation_recommendations(analysis))

        # Recomendações baseadas em configuração
        recommendations.extend(
            self._generate_configuration_recommendations(analysis, current_config)
        )

        # Ordena por prioridade
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _generate_efficiency_recommendations(
        self, analysis: Dict
    ) -> List[OptimizationRecommendation]:
        """Gera recomendações de eficiência."""
        recommendations = []
        efficiency = analysis.get("efficiency_patterns", {})

        if not efficiency or "error" in efficiency:
            return recommendations

        avg_cost_per_item = efficiency.get("average_cost_per_item", 0)
        avg_tokens_per_second = efficiency.get("average_tokens_per_second", 0)

        # Alto custo por item
        if avg_cost_per_item > self.efficiency_thresholds["cost_per_item_critical"]:
            recommendations.append(
                OptimizationRecommendation(
                    id="high_cost_per_item",
                    category=OptimizationCategory.COST_REDUCTION,
                    title="Reduzir custo por item processado",
                    description=f"Custo atual de ${avg_cost_per_item:.6f} por item está acima do threshold crítico de ${self.efficiency_thresholds['cost_per_item_critical']:.2f}",
                    impact_level=ImpactLevel.HIGH,
                    complexity=ImplementationComplexity.MODERATE,
                    estimated_savings_usd=avg_cost_per_item
                    * 1000
                    * 0.3,  # 30% economia em 1000 items
                    estimated_time_savings_hours=0,
                    confidence_score=0.9,
                    implementation_steps=[
                        "Analisar breakdown de custos por operação",
                        "Otimizar tamanho de chunks para reduzir overhead",
                        "Implementar pre-processamento para reduzir tokens",
                        "Ajustar configurações do modelo para eficiência",
                    ],
                    risks=[
                        "Redução de qualidade dos resultados",
                        "Tempo inicial de implementação",
                        "Necessidade de revalidação",
                    ],
                    prerequisites=[
                        "Análise detalhada do breakdown de custos",
                        "Dataset de teste para validação",
                    ],
                    metrics_to_track=[
                        "cost_per_item",
                        "output_quality_score",
                        "processing_time",
                    ],
                    supporting_data={
                        "current_cost_per_item": avg_cost_per_item,
                        "threshold": self.efficiency_thresholds[
                            "cost_per_item_critical"
                        ],
                    },
                )
            )

        # Baixo throughput
        if avg_tokens_per_second < self.efficiency_thresholds["tokens_per_second_min"]:
            recommendations.append(
                OptimizationRecommendation(
                    id="low_throughput",
                    category=OptimizationCategory.PERFORMANCE_IMPROVEMENT,
                    title="Melhorar throughput de processamento",
                    description=f"Throughput atual de {avg_tokens_per_second:.1f} tokens/s está abaixo do mínimo de {self.efficiency_thresholds['tokens_per_second_min']} tokens/s",
                    impact_level=ImpactLevel.MEDIUM,
                    complexity=ImplementationComplexity.MODERATE,
                    estimated_savings_usd=0,  # Foco em tempo, não custo direto
                    estimated_time_savings_hours=20,  # Economia de tempo
                    confidence_score=0.8,
                    implementation_steps=[
                        "Aumentar paralelismo de workers",
                        "Otimizar I/O operations",
                        "Implementar batching mais eficiente",
                        "Ajustar timeouts e retry logic",
                    ],
                    risks=[
                        "Aumento de complexidade",
                        "Possível aumento de custos de infraestrutura",
                    ],
                    prerequisites=[
                        "Monitoring de recursos do sistema",
                        "Análise de bottlenecks",
                    ],
                    metrics_to_track=[
                        "tokens_per_second",
                        "cpu_utilization",
                        "memory_usage",
                        "api_latency",
                    ],
                    supporting_data={
                        "current_throughput": avg_tokens_per_second,
                        "target_throughput": self.efficiency_thresholds[
                            "tokens_per_second_min"
                        ],
                    },
                )
            )

        return recommendations

    def _generate_resource_recommendations(
        self, analysis: Dict
    ) -> List[OptimizationRecommendation]:
        """Gera recomendações de recursos."""
        recommendations = []
        resource = analysis.get("resource_patterns", {})

        if not resource:
            return recommendations

        avg_workers = resource.get("average_parallel_workers", 4)

        # Otimização de workers
        if avg_workers < 4:
            recommendations.append(
                OptimizationRecommendation(
                    id="increase_parallelism",
                    category=OptimizationCategory.PERFORMANCE_IMPROVEMENT,
                    title="Aumentar paralelismo de processamento",
                    description=f"Uso atual de {avg_workers:.1f} workers pode ser otimizado para 4-6 workers",
                    impact_level=ImpactLevel.MEDIUM,
                    complexity=ImplementationComplexity.EASY,
                    estimated_savings_usd=0,
                    estimated_time_savings_hours=10,
                    confidence_score=0.85,
                    implementation_steps=[
                        "Ajustar max_workers para 4-6",
                        "Monitorar utilização de CPU",
                        "Validar que não há contenção de recursos",
                    ],
                    risks=[
                        "Possível contenção de recursos",
                        "Aumento marginal de overhead",
                    ],
                    prerequisites=["Recursos computacionais adequados"],
                    metrics_to_track=[
                        "processing_time",
                        "cpu_utilization",
                        "memory_usage",
                    ],
                    supporting_data={
                        "current_workers": avg_workers,
                        "recommended_workers": "4-6",
                    },
                )
            )

        elif avg_workers > 8:
            recommendations.append(
                OptimizationRecommendation(
                    id="reduce_parallelism",
                    category=OptimizationCategory.RESOURCE_OPTIMIZATION,
                    title="Reduzir paralelismo excessivo",
                    description=f"Uso de {avg_workers:.1f} workers pode estar causando overhead",
                    impact_level=ImpactLevel.LOW,
                    complexity=ImplementationComplexity.EASY,
                    estimated_savings_usd=50,  # Economia de overhead
                    estimated_time_savings_hours=0,
                    confidence_score=0.7,
                    implementation_steps=[
                        "Reduzir max_workers para 4-6",
                        "Monitorar impact na performance",
                        "Ajustar baseado em resultados",
                    ],
                    risks=["Possível aumento de tempo de processamento"],
                    prerequisites=[],
                    metrics_to_track=["processing_time", "cost_per_item", "throughput"],
                    supporting_data={
                        "current_workers": avg_workers,
                        "recommended_workers": "4-6",
                    },
                )
            )

        return recommendations

    def _generate_cache_recommendations(
        self, analysis: Dict
    ) -> List[OptimizationRecommendation]:
        """Gera recomendações de cache."""
        recommendations = []
        cache = analysis.get("cache_patterns", {})

        if not cache or "message" in cache:
            return recommendations

        hit_rate = cache.get("hit_rate", 0)
        cache_usage = cache.get("cache_usage", 0)
        compression_ratio = cache.get("compression_ratio", 0)

        # Baixo hit rate
        if hit_rate < self.efficiency_thresholds["hit_rate_min"]:
            recommendations.append(
                OptimizationRecommendation(
                    id="improve_cache_hit_rate",
                    category=OptimizationCategory.PERFORMANCE_IMPROVEMENT,
                    title="Melhorar hit rate do cache",
                    description=f"Hit rate atual de {hit_rate*100:.1f}% está abaixo do mínimo de {self.efficiency_thresholds['hit_rate_min']*100:.0f}%",
                    impact_level=ImpactLevel.MEDIUM,
                    complexity=ImplementationComplexity.MODERATE,
                    estimated_savings_usd=100,  # Economia por reprocessamento evitado
                    estimated_time_savings_hours=5,
                    confidence_score=0.8,
                    implementation_steps=[
                        "Analisar padrões de invalidação de cache",
                        "Otimizar chaves de cache para maior reutilização",
                        "Implementar cache warming strategies",
                        "Ajustar políticas de LRU",
                    ],
                    risks=["Aumento no uso de memória", "Complexidade de invalidação"],
                    prerequisites=[
                        "Análise de padrões de acesso",
                        "Monitoring de cache",
                    ],
                    metrics_to_track=[
                        "cache_hit_rate",
                        "cache_memory_usage",
                        "processing_time",
                    ],
                    supporting_data={
                        "current_hit_rate": hit_rate,
                        "target_hit_rate": self.efficiency_thresholds["hit_rate_min"],
                    },
                )
            )

        # Baixa compressão
        if compression_ratio < 0.3:  # Menos de 30% dos arquivos comprimidos
            recommendations.append(
                OptimizationRecommendation(
                    id="improve_cache_compression",
                    category=OptimizationCategory.RESOURCE_OPTIMIZATION,
                    title="Melhorar compressão de cache",
                    description=f"Apenas {compression_ratio*100:.1f}% dos arquivos estão comprimidos",
                    impact_level=ImpactLevel.LOW,
                    complexity=ImplementationComplexity.EASY,
                    estimated_savings_usd=20,  # Economia de storage
                    estimated_time_savings_hours=0,
                    confidence_score=0.9,
                    implementation_steps=[
                        "Reduzir threshold de compressão",
                        "Implementar compressão automática",
                        "Otimizar algoritmo de compressão",
                    ],
                    risks=["Overhead de CPU para compressão"],
                    prerequisites=[],
                    metrics_to_track=[
                        "compression_ratio",
                        "storage_usage",
                        "compression_overhead",
                    ],
                    supporting_data={
                        "current_compression_ratio": compression_ratio,
                        "total_files": cache.get("total_files", 0),
                    },
                )
            )

        return recommendations

    def _generate_operation_recommendations(
        self, analysis: Dict
    ) -> List[OptimizationRecommendation]:
        """Gera recomendações baseadas em operações."""
        recommendations = []
        operations = analysis.get("operation_patterns", {})

        if not operations or "message" in operations:
            return recommendations

        most_expensive = operations.get("most_expensive_operation")
        operation_efficiencies = operations.get("operation_efficiencies", {})

        if most_expensive and most_expensive in operation_efficiencies:
            expensive_op = operation_efficiencies[most_expensive]
            cost_per_op = expensive_op.get("cost_per_operation", 0)

            if cost_per_op > 0.01:  # $0.01 per operation
                recommendations.append(
                    OptimizationRecommendation(
                        id="optimize_expensive_operation",
                        category=OptimizationCategory.COST_REDUCTION,
                        title=f"Otimizar operação '{most_expensive}'",
                        description=f"Operação '{most_expensive}' custa ${cost_per_op:.6f} por execução, acima do threshold",
                        impact_level=(
                            ImpactLevel.HIGH
                            if cost_per_op > 0.05
                            else ImpactLevel.MEDIUM
                        ),
                        complexity=ImplementationComplexity.MODERATE,
                        estimated_savings_usd=cost_per_op
                        * expensive_op.get("operations_count", 0)
                        * 0.3,
                        estimated_time_savings_hours=0,
                        confidence_score=0.8,
                        implementation_steps=[
                            f"Analisar implementação da operação '{most_expensive}'",
                            "Identificar gargalos específicos",
                            "Implementar otimizações direcionadas",
                            "Medir impact das mudanças",
                        ],
                        risks=[
                            "Mudanças podem afetar qualidade",
                            "Necessidade de revalidação",
                        ],
                        prerequisites=[
                            "Análise detalhada da operação",
                            "Métricas de baseline",
                        ],
                        metrics_to_track=[
                            "cost_per_operation",
                            "operation_latency",
                            "output_quality",
                        ],
                        supporting_data={
                            "operation_name": most_expensive,
                            "current_cost": cost_per_op,
                            "operation_count": expensive_op.get("operations_count", 0),
                        },
                    )
                )

        return recommendations

    def _generate_configuration_recommendations(
        self, analysis: Dict, current_config: Dict = None
    ) -> List[OptimizationRecommendation]:
        """Gera recomendações de configuração."""
        recommendations = []

        if not current_config:
            current_config = {}

        # Recomendações baseadas em padrões de análise
        efficiency = analysis.get("efficiency_patterns", {})

        if not efficiency or "error" in efficiency:
            return recommendations

        cost_trend = efficiency.get("cost_per_item_trend", "stable")

        if cost_trend == "deteriorating":
            recommendations.append(
                OptimizationRecommendation(
                    id="review_model_configuration",
                    category=OptimizationCategory.CONFIGURATION_TUNING,
                    title="Revisar configuração do modelo",
                    description="Tendência de deterioração nos custos indica necessidade de ajuste",
                    impact_level=ImpactLevel.MEDIUM,
                    complexity=ImplementationComplexity.MODERATE,
                    estimated_savings_usd=200,
                    estimated_time_savings_hours=0,
                    confidence_score=0.7,
                    implementation_steps=[
                        "Revisar parâmetros do modelo (temperature, top_p, top_k)",
                        "Testar configurações alternativas",
                        "Implementar A/B testing",
                        "Monitorar impact na qualidade",
                    ],
                    risks=[
                        "Mudanças podem afetar qualidade de output",
                        "Necessidade de re-tuning",
                    ],
                    prerequisites=["Dataset de validação", "Métricas de qualidade"],
                    metrics_to_track=[
                        "cost_per_item",
                        "output_quality_score",
                        "processing_time",
                    ],
                    supporting_data={
                        "cost_trend": cost_trend,
                        "current_config": current_config,
                    },
                )
            )

        return recommendations

    def prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        constraints: Dict[str, Any] = None,
    ) -> List[OptimizationRecommendation]:
        """
        Prioriza recomendações baseado em constraints.

        Args:
            recommendations: Lista de recomendações
            constraints: Constraints como budget, complexity_limit, etc.

        Returns:
            Lista priorizada de recomendações
        """
        if not constraints:
            # Retorna ordenado por priority_score
            return sorted(recommendations, key=lambda r: r.priority_score, reverse=True)

        filtered_recommendations = []

        for rec in recommendations:
            # Filtra por complexity se especificado
            if "max_complexity" in constraints:
                max_complexity = constraints["max_complexity"]
                complexity_order = [
                    ImplementationComplexity.EASY,
                    ImplementationComplexity.MODERATE,
                    ImplementationComplexity.COMPLEX,
                ]
                if complexity_order.index(rec.complexity) > complexity_order.index(
                    max_complexity
                ):
                    continue

            # Filtra por budget se especificado
            if "min_savings" in constraints:
                if rec.estimated_savings_usd < constraints["min_savings"]:
                    continue

            # Filtra por impact level se especificado
            if "min_impact" in constraints:
                impact_order = [
                    ImpactLevel.LOW,
                    ImpactLevel.MEDIUM,
                    ImpactLevel.HIGH,
                    ImpactLevel.CRITICAL,
                ]
                min_impact = constraints["min_impact"]
                if impact_order.index(rec.impact_level) < impact_order.index(
                    min_impact
                ):
                    continue

            filtered_recommendations.append(rec)

        # Ordena por prioridade
        return sorted(
            filtered_recommendations, key=lambda r: r.priority_score, reverse=True
        )

    def export_recommendations_report(
        self, recommendations: List[OptimizationRecommendation], filename: str = None
    ) -> Path:
        """Exporta relatório de recomendações."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_recommendations_{timestamp}.json"

        report_file = self.storage_dir / filename

        # Prepara dados para export
        report_data = {
            "report_generated": datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "recommendations": [asdict(rec) for rec in recommendations],
            "summary": {
                "total_estimated_savings": sum(
                    r.estimated_savings_usd for r in recommendations
                ),
                "total_estimated_time_savings": sum(
                    r.estimated_time_savings_hours for r in recommendations
                ),
                "by_category": self._summarize_by_category(recommendations),
                "by_impact": self._summarize_by_impact(recommendations),
                "by_complexity": self._summarize_by_complexity(recommendations),
            },
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"📊 Relatório de recomendações exportado: {report_file}")
        return report_file

    def _summarize_by_category(
        self, recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, int]:
        """Sumariza recomendações por categoria."""
        summary = {}
        for rec in recommendations:
            category = rec.category.value
            summary[category] = summary.get(category, 0) + 1
        return summary

    def _summarize_by_impact(
        self, recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, int]:
        """Sumariza recomendações por nível de impacto."""
        summary = {}
        for rec in recommendations:
            impact = rec.impact_level.value
            summary[impact] = summary.get(impact, 0) + 1
        return summary

    def _summarize_by_complexity(
        self, recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, int]:
        """Sumariza recomendações por complexidade."""
        summary = {}
        for rec in recommendations:
            complexity = rec.complexity.value
            summary[complexity] = summary.get(complexity, 0) + 1
        return summary

    def generate_implementation_plan(
        self,
        recommendations: List[OptimizationRecommendation],
        timeline_weeks: int = 12,
    ) -> Dict[str, Any]:
        """
        Gera plano de implementação para as recomendações.

        Args:
            recommendations: Lista priorizada de recomendações
            timeline_weeks: Timeline em semanas para implementação

        Returns:
            Plano de implementação estruturado
        """
        # Agrupa por complexidade e impact
        easy_high_impact = [
            r
            for r in recommendations
            if r.complexity == ImplementationComplexity.EASY
            and r.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]
        ]

        quick_wins = [
            r for r in recommendations if r.complexity == ImplementationComplexity.EASY
        ]

        major_projects = [
            r
            for r in recommendations
            if r.complexity == ImplementationComplexity.COMPLEX
        ]

        # Distribui ao longo do timeline
        weeks_per_phase = timeline_weeks // 3

        implementation_plan = {
            "timeline_weeks": timeline_weeks,
            "phases": {
                "phase_1_immediate": {
                    "weeks": f"1-{weeks_per_phase}",
                    "focus": "Quick wins e high-impact easy implementations",
                    "recommendations": easy_high_impact[:3] + quick_wins[:2],
                    "expected_savings": sum(
                        r.estimated_savings_usd
                        for r in easy_high_impact[:3] + quick_wins[:2]
                    ),
                },
                "phase_2_medium_term": {
                    "weeks": f"{weeks_per_phase+1}-{weeks_per_phase*2}",
                    "focus": "Moderate complexity optimizations",
                    "recommendations": [
                        r
                        for r in recommendations
                        if r.complexity == ImplementationComplexity.MODERATE
                    ][:4],
                    "expected_savings": sum(
                        r.estimated_savings_usd
                        for r in recommendations
                        if r.complexity == ImplementationComplexity.MODERATE
                    ),
                },
                "phase_3_long_term": {
                    "weeks": f"{weeks_per_phase*2+1}-{timeline_weeks}",
                    "focus": "Complex optimizations e major refactoring",
                    "recommendations": major_projects[:2],
                    "expected_savings": sum(
                        r.estimated_savings_usd for r in major_projects[:2]
                    ),
                },
            },
            "total_estimated_savings": sum(
                r.estimated_savings_usd for r in recommendations
            ),
            "success_metrics": [
                "cost_per_item_reduction",
                "throughput_improvement",
                "cache_hit_rate_improvement",
                "overall_efficiency_score",
            ],
        }

        return implementation_plan
