"""
Sistema de projeção de custos para diferentes tamanhos de dataset.

Implementa modelos matemáticos para projetar custos de processamento
baseados em métricas históricas e padrões de uso de tokens.
"""

import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ProcessingOption(Enum):
    """Opções de processamento disponíveis."""

    OPTION_A = "discovery_only"  # Apenas descoberta
    OPTION_B = "application_only"  # Apenas aplicação
    OPTION_C = "parallel_discovery"  # Descoberta paralela
    OPTION_D = "optimized_application"  # Aplicação otimizada
    OPTION_E = "discovery_application"  # Descoberta + Aplicação (baseline)


@dataclass
class ProjectionScenario:
    """Cenário de projeção com diferentes parâmetros."""

    dataset_size: int
    processing_option: ProcessingOption
    avg_tokens_per_ticket: float
    efficiency_factor: float = 1.0  # Fator de eficiência (1.0 = baseline)
    parallel_workers: int = 4
    chunk_size_multiplier: float = 1.0  # Multiplicador do tamanho do chunk


@dataclass
class CostProjection:
    """Resultado de projeção de custos."""

    scenario: ProjectionScenario
    estimated_total_cost: float
    cost_per_item: float
    estimated_tokens: int
    processing_time_hours: float
    confidence_interval: Tuple[float, float]  # (min, max) custo
    risk_factors: List[str]
    optimization_opportunities: List[str]


class CostProjector:
    """
    Sistema de projeção de custos baseado em modelos matemáticos.

    Utiliza dados históricos para criar projeções precisas de custo
    para diferentes tamanhos de dataset e opções de processamento.
    """

    # Baseline baseado no Option E (discovery + application)
    BASELINE_COST_PER_TICKET = 0.048  # $0.048 baseado na análise inicial
    BASELINE_TOKENS_PER_TICKET = 3200  # ~800 chars * 4 = 3200 tokens médio

    # Pricing Gemini 2.5 Flash
    INPUT_TOKEN_COST = 0.125 / 1000  # $0.000125 per token
    OUTPUT_TOKEN_COST = 0.375 / 1000  # $0.000375 per token

    # Fatores de eficiência por opção de processamento
    PROCESSING_EFFICIENCY_FACTORS = {
        ProcessingOption.OPTION_A: 0.6,  # Apenas descoberta - mais eficiente
        ProcessingOption.OPTION_B: 0.8,  # Apenas aplicação - eficiente
        ProcessingOption.OPTION_C: 0.7,  # Paralela - overhead de coordenação
        ProcessingOption.OPTION_D: 0.85,  # Otimizada - boa eficiência
        ProcessingOption.OPTION_E: 1.0,  # Baseline completo
    }

    def __init__(self, historical_sessions: List[Dict] = None):
        """
        Inicializa o projetor de custos.

        Args:
            historical_sessions: Lista de sessões históricas para calibração
        """
        self.historical_sessions = historical_sessions or []
        self._calibrated_baseline = None

        # Calibra baseline se há dados históricos
        if self.historical_sessions:
            self._calibrate_baseline()

    def _calibrate_baseline(self):
        """Calibra baseline baseado em dados históricos."""
        if not self.historical_sessions:
            return

        # Extrai métricas históricas
        costs_per_item = []
        tokens_per_item = []

        for session in self.historical_sessions:
            if session.get("dataset_size", 0) > 0:
                cost_per_item = (
                    session.get("total_cost_usd", 0) / session["dataset_size"]
                )
                total_tokens = session.get("total_input_tokens", 0) + session.get(
                    "total_output_tokens", 0
                )
                tokens_per_ticket = total_tokens / session["dataset_size"]

                costs_per_item.append(cost_per_item)
                tokens_per_item.append(tokens_per_ticket)

        if costs_per_item and tokens_per_item:
            self._calibrated_baseline = {
                "cost_per_ticket": statistics.median(costs_per_item),
                "tokens_per_ticket": statistics.median(tokens_per_item),
                "cost_variance": (
                    statistics.stdev(costs_per_item) if len(costs_per_item) > 1 else 0
                ),
                "samples": len(costs_per_item),
            }

            print(f"📊 Baseline calibrado com {len(costs_per_item)} amostras:")
            print(
                f"   Custo médio por ticket: ${self._calibrated_baseline['cost_per_ticket']:.6f}"
            )
            print(
                f"   Tokens médios por ticket: {self._calibrated_baseline['tokens_per_ticket']:.0f}"
            )

    def project_cost(self, scenario: ProjectionScenario) -> CostProjection:
        """
        Projeta custo para um cenário específico.

        Args:
            scenario: Cenário de projeção

        Returns:
            Projeção detalhada de custos
        """
        # Usa baseline calibrado se disponível
        if self._calibrated_baseline:
            base_cost_per_ticket = self._calibrated_baseline["cost_per_ticket"]
            base_tokens_per_ticket = self._calibrated_baseline["tokens_per_ticket"]
        else:
            base_cost_per_ticket = self.BASELINE_COST_PER_TICKET
            base_tokens_per_ticket = self.BASELINE_TOKENS_PER_TICKET

        # Aplica fatores de eficiência
        efficiency_factor = (
            self.PROCESSING_EFFICIENCY_FACTORS.get(scenario.processing_option, 1.0)
            * scenario.efficiency_factor
        )

        # Calcula tokens estimados
        estimated_tokens_per_ticket = (
            base_tokens_per_ticket
            * scenario.avg_tokens_per_ticket
            / self.BASELINE_TOKENS_PER_TICKET
        )
        total_estimated_tokens = int(
            estimated_tokens_per_ticket * scenario.dataset_size
        )

        # Aplica fatores de escala
        scale_factor = self._calculate_scale_factor(scenario.dataset_size)
        chunk_factor = self._calculate_chunk_factor(scenario.chunk_size_multiplier)
        parallel_factor = self._calculate_parallel_factor(scenario.parallel_workers)

        # Custo base ajustado
        adjusted_cost_per_ticket = (
            base_cost_per_ticket
            * efficiency_factor
            * scale_factor
            * chunk_factor
            * parallel_factor
        )

        total_cost = adjusted_cost_per_ticket * scenario.dataset_size

        # Calcula tempo de processamento estimado
        processing_time = self._estimate_processing_time(
            scenario.dataset_size, scenario.parallel_workers, efficiency_factor
        )

        # Calcula intervalo de confiança
        confidence_interval = self._calculate_confidence_interval(
            total_cost, scenario.dataset_size
        )

        # Identifica fatores de risco
        risk_factors = self._identify_risk_factors(scenario, scale_factor)

        # Identifica oportunidades de otimização
        optimization_opportunities = self._identify_optimizations(scenario)

        return CostProjection(
            scenario=scenario,
            estimated_total_cost=total_cost,
            cost_per_item=adjusted_cost_per_ticket,
            estimated_tokens=total_estimated_tokens,
            processing_time_hours=processing_time,
            confidence_interval=confidence_interval,
            risk_factors=risk_factors,
            optimization_opportunities=optimization_opportunities,
        )

    def _calculate_scale_factor(self, dataset_size: int) -> float:
        """
        Calcula fator de escala baseado no tamanho do dataset.
        Datasets maiores podem ter eficiências diferentes.
        """
        if dataset_size <= 1000:
            return 1.1  # Pequenos datasets têm overhead
        elif dataset_size <= 10000:
            return 1.0  # Tamanho padrão
        elif dataset_size <= 50000:
            return 0.95  # Ligeira eficiência de escala
        elif dataset_size <= 100000:
            return 0.92  # Boa eficiência de escala
        else:
            return 0.90  # Máxima eficiência de escala

    def _calculate_chunk_factor(self, chunk_multiplier: float) -> float:
        """
        Calcula fator baseado no tamanho do chunk.
        Chunks maiores podem ser mais eficientes mas têm limites.
        """
        if chunk_multiplier < 0.5:
            return 1.2  # Chunks muito pequenos = ineficiência
        elif chunk_multiplier <= 1.0:
            return 1.0  # Tamanho padrão
        elif chunk_multiplier <= 2.0:
            return 0.95  # Chunks maiores = eficiência
        else:
            return 1.05  # Chunks muito grandes = overhead

    def _calculate_parallel_factor(self, workers: int) -> float:
        """
        Calcula fator baseado no número de workers paralelos.
        """
        if workers <= 1:
            return 1.0  # Sem paralelismo
        elif workers <= 4:
            return 0.95  # Paralelismo eficiente
        elif workers <= 8:
            return 0.98  # Paralelismo com overhead mínimo
        else:
            return 1.05  # Muito paralelismo = overhead

    def _estimate_processing_time(
        self, dataset_size: int, workers: int, efficiency_factor: float
    ) -> float:
        """
        Estima tempo de processamento em horas.
        """
        # Baseline: ~1000 tickets por hora com 4 workers
        base_throughput = 1000 * efficiency_factor

        # Ajusta pela paralelização
        if workers > 1:
            parallel_efficiency = min(workers * 0.8, workers)  # Não é linear
            adjusted_throughput = base_throughput * parallel_efficiency / 4
        else:
            adjusted_throughput = base_throughput / 4

        return dataset_size / adjusted_throughput

    def _calculate_confidence_interval(
        self, estimated_cost: float, dataset_size: int
    ) -> Tuple[float, float]:
        """
        Calcula intervalo de confiança baseado na variância histórica.
        """
        if self._calibrated_baseline and self._calibrated_baseline["samples"] > 1:
            # Usa variância histórica
            variance_factor = (
                self._calibrated_baseline["cost_variance"]
                / self._calibrated_baseline["cost_per_ticket"]
            )
        else:
            # Variância estimada baseada no tamanho do dataset
            if dataset_size < 1000:
                variance_factor = 0.3  # Alta variância para datasets pequenos
            elif dataset_size < 10000:
                variance_factor = 0.2  # Variância média
            else:
                variance_factor = 0.15  # Baixa variância para datasets grandes

        margin = estimated_cost * variance_factor
        return (estimated_cost - margin, estimated_cost + margin)

    def _identify_risk_factors(
        self, scenario: ProjectionScenario, scale_factor: float
    ) -> List[str]:
        """Identifica fatores de risco para a projeção."""
        risks = []

        if scenario.dataset_size > 100000:
            risks.append("Dataset muito grande pode ter variabilidade alta")

        if scenario.dataset_size < 100:
            risks.append("Dataset pequeno pode ter alto overhead relativo")

        if scenario.parallel_workers > 8:
            risks.append("Alto paralelismo pode causar contenção de recursos")

        if scenario.chunk_size_multiplier > 2.0:
            risks.append("Chunks muito grandes podem exceder limites de API")

        if scenario.chunk_size_multiplier < 0.5:
            risks.append("Chunks muito pequenos podem ser ineficientes")

        if scenario.avg_tokens_per_ticket > 10000:
            risks.append("Tickets muito longos podem ter custo desproporcional")

        if scale_factor > 1.05:
            risks.append("Fator de escala sugere ineficiência para este tamanho")

        if not self._calibrated_baseline:
            risks.append("Projeção baseada em baseline teórico (sem dados históricos)")

        return risks

    def _identify_optimizations(self, scenario: ProjectionScenario) -> List[str]:
        """Identifica oportunidades de otimização."""
        optimizations = []

        if scenario.parallel_workers < 4:
            optimizations.append(
                "Aumentar paralelismo pode reduzir tempo de processamento"
            )

        if scenario.chunk_size_multiplier < 1.0:
            optimizations.append("Aumentar tamanho do chunk pode melhorar eficiência")

        if scenario.processing_option not in [
            ProcessingOption.OPTION_D,
            ProcessingOption.OPTION_E,
        ]:
            optimizations.append("Considerar opção de processamento otimizada")

        if scenario.dataset_size > 50000:
            optimizations.append(
                "Dataset grande pode se beneficiar de pré-processamento"
            )

        if scenario.avg_tokens_per_ticket > 5000:
            optimizations.append(
                "Tickets longos podem se beneficiar de summarização prévia"
            )

        if scenario.efficiency_factor == 1.0:
            optimizations.append("Aplicar otimizações específicas do domínio")

        return optimizations

    def compare_scenarios(self, scenarios: List[ProjectionScenario]) -> Dict[str, Any]:
        """
        Compara múltiplos cenários de projeção.

        Args:
            scenarios: Lista de cenários para comparar

        Returns:
            Análise comparativa detalhada
        """
        projections = [self.project_cost(scenario) for scenario in scenarios]

        # Ordena por custo total
        sorted_projections = sorted(projections, key=lambda p: p.estimated_total_cost)

        best_projection = sorted_projections[0]
        worst_projection = sorted_projections[-1]

        # Calcula médias
        avg_cost = statistics.mean(p.estimated_total_cost for p in projections)
        avg_cost_per_item = statistics.mean(p.cost_per_item for p in projections)
        avg_time = statistics.mean(p.processing_time_hours for p in projections)

        comparison = {
            "scenarios_analyzed": len(scenarios),
            "best_option": {
                "scenario": best_projection.scenario.processing_option.value,
                "total_cost": best_projection.estimated_total_cost,
                "cost_per_item": best_projection.cost_per_item,
                "processing_time_hours": best_projection.processing_time_hours,
            },
            "worst_option": {
                "scenario": worst_projection.scenario.processing_option.value,
                "total_cost": worst_projection.estimated_total_cost,
                "cost_per_item": worst_projection.cost_per_item,
                "processing_time_hours": worst_projection.processing_time_hours,
            },
            "cost_range": {
                "min": best_projection.estimated_total_cost,
                "max": worst_projection.estimated_total_cost,
                "avg": avg_cost,
                "savings_potential": worst_projection.estimated_total_cost
                - best_projection.estimated_total_cost,
            },
            "performance_analysis": {
                "avg_cost_per_item": avg_cost_per_item,
                "avg_processing_time_hours": avg_time,
                "time_range": {
                    "min": min(p.processing_time_hours for p in projections),
                    "max": max(p.processing_time_hours for p in projections),
                },
            },
            "detailed_projections": [
                {
                    "processing_option": p.scenario.processing_option.value,
                    "dataset_size": p.scenario.dataset_size,
                    "total_cost": p.estimated_total_cost,
                    "cost_per_item": p.cost_per_item,
                    "processing_time_hours": p.processing_time_hours,
                    "confidence_interval": p.confidence_interval,
                    "risk_count": len(p.risk_factors),
                    "optimization_count": len(p.optimization_opportunities),
                }
                for p in projections
            ],
        }

        return comparison

    def generate_cost_projections_for_targets(
        self, target_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Gera projeções para tamanhos de dataset comuns.

        Args:
            target_sizes: Lista de tamanhos alvo (default: [50K, 100K, 500K])

        Returns:
            Projeções para diferentes tamanhos
        """
        if target_sizes is None:
            target_sizes = [50000, 100000, 500000]

        results = {}

        for size in target_sizes:
            scenarios = []

            # Cria cenários para cada opção de processamento
            for option in ProcessingOption:
                scenario = ProjectionScenario(
                    dataset_size=size,
                    processing_option=option,
                    avg_tokens_per_ticket=self.BASELINE_TOKENS_PER_TICKET,
                    parallel_workers=4,
                )
                scenarios.append(scenario)

            # Compara cenários para este tamanho
            comparison = self.compare_scenarios(scenarios)
            results[f"{size:,}_tickets"] = comparison

        # Adiciona análise cross-size
        all_best_costs = [
            results[f"{size:,}_tickets"]["best_option"]["total_cost"]
            for size in target_sizes
        ]
        all_worst_costs = [
            results[f"{size:,}_tickets"]["worst_option"]["total_cost"]
            for size in target_sizes
        ]

        results["scaling_analysis"] = {
            "dataset_sizes": target_sizes,
            "best_costs": all_best_costs,
            "worst_costs": all_worst_costs,
            "cost_per_10k_scale": [
                cost / (size / 10000)
                for cost, size in zip(all_best_costs, target_sizes)
            ],
            "efficiency_trend": (
                "improving"
                if all_best_costs[-1] / target_sizes[-1]
                < all_best_costs[0] / target_sizes[0]
                else "stable"
            ),
        }

        return results

    def estimate_roi_metrics(
        self,
        projection: CostProjection,
        manual_cost_per_ticket: float = 50.0,
        manual_time_per_ticket_hours: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calcula métricas de ROI comparando com processamento manual.

        Args:
            projection: Projeção de custo automatizado
            manual_cost_per_ticket: Custo de processamento manual por ticket
            manual_time_per_ticket_hours: Tempo manual por ticket em horas

        Returns:
            Métricas de ROI detalhadas
        """
        dataset_size = projection.scenario.dataset_size

        # Custos
        automated_cost = projection.estimated_total_cost
        manual_cost = dataset_size * manual_cost_per_ticket
        cost_savings = manual_cost - automated_cost

        # Tempo
        automated_time = projection.processing_time_hours
        manual_time = dataset_size * manual_time_per_ticket_hours
        time_savings_hours = manual_time - automated_time

        # ROI
        roi_percentage = (
            (cost_savings / automated_cost) * 100 if automated_cost > 0 else 0
        )

        # Payback
        setup_cost = 1000  # Custo estimado de setup do sistema
        total_investment = automated_cost + setup_cost
        payback_ratio = cost_savings / total_investment if total_investment > 0 else 0

        return {
            "dataset_size": dataset_size,
            "cost_analysis": {
                "automated_cost_usd": automated_cost,
                "manual_cost_usd": manual_cost,
                "cost_savings_usd": cost_savings,
                "cost_reduction_percent": (cost_savings / manual_cost) * 100,
            },
            "time_analysis": {
                "automated_time_hours": automated_time,
                "manual_time_hours": manual_time,
                "time_savings_hours": time_savings_hours,
                "time_reduction_percent": (time_savings_hours / manual_time) * 100,
            },
            "roi_metrics": {
                "roi_percentage": roi_percentage,
                "payback_ratio": payback_ratio,
                "cost_per_ticket_automated": projection.cost_per_item,
                "cost_per_ticket_manual": manual_cost_per_ticket,
                "efficiency_multiplier": manual_cost_per_ticket
                / projection.cost_per_item,
            },
            "business_impact": {
                "break_even_volume": setup_cost
                / (manual_cost_per_ticket - projection.cost_per_item),
                "monthly_savings_potential": cost_savings,  # Assumindo processamento mensal
                "scalability_factor": roi_percentage
                / 100,  # Quanto melhor fica com escala
            },
        }
