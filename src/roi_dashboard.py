"""
Dashboard de analytics ROI para análise de custos e retorno sobre investimento.

Fornece interface visual e relatórios para análise de performance financeira,
comparação de cenários e métricas de eficiência operacional.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import csv
from dataclasses import asdict

try:
    from .cost_tracker import CostTracker, CostSession
    from .cost_projector import CostProjector, ProjectionScenario, ProcessingOption
except ImportError:
    from cost_tracker import CostTracker, CostSession
    from cost_projector import CostProjector, ProjectionScenario, ProcessingOption


class ROIDashboard:
    """
    Dashboard principal para análise de ROI e métricas de custo.

    Integra dados de tracking em tempo real com projeções e análises
    comparativas para fornecer insights acionáveis sobre performance financeira.
    """

    def __init__(self, cost_tracker: CostTracker, storage_dir: Path = None):
        """
        Inicializa o dashboard ROI.

        Args:
            cost_tracker: Instância do sistema de tracking de custos
            storage_dir: Diretório para armazenar relatórios
        """
        self.cost_tracker = cost_tracker
        self.storage_dir = storage_dir or Path("database/roi_analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Inicializa projetor com dados históricos
        historical_sessions = self._get_historical_sessions()
        self.cost_projector = CostProjector(historical_sessions)

        # Cache de relatórios
        self._cached_reports = {}
        self._cache_timeout = 300  # 5 minutos

    def _get_historical_sessions(self) -> List[Dict]:
        """Recupera sessões históricas do cost tracker."""
        sessions = []
        for session in self.cost_tracker._sessions_history:
            session_dict = asdict(session)
            sessions.append(session_dict)
        return sessions

    def generate_real_time_overview(self) -> Dict[str, Any]:
        """
        Gera overview em tempo real do status de custos.

        Returns:
            Dashboard overview com métricas principais
        """
        current_stats = self.cost_tracker.get_current_session_stats()
        operation_breakdown = self.cost_tracker.get_operation_breakdown()
        cost_trends = self.cost_tracker.get_cost_trends(hours_back=24)
        efficiency_metrics = self.cost_tracker.get_efficiency_metrics()

        overview = {
            "dashboard_timestamp": datetime.now().isoformat(),
            "current_session": current_stats,
            "performance_summary": {
                "total_historical_cost": self.cost_tracker.get_total_spent(),
                "total_sessions": len(self.cost_tracker._sessions_history),
                "efficiency_metrics": efficiency_metrics,
                "cost_trends_24h": cost_trends,
            },
            "operation_analysis": operation_breakdown,
            "status_indicators": self._generate_status_indicators(current_stats),
            "quick_insights": self._generate_quick_insights(current_stats, cost_trends),
        }

        return overview

    def _generate_status_indicators(self, current_stats: Dict) -> Dict[str, Any]:
        """Gera indicadores visuais de status."""
        indicators = {
            "session_active": "error" not in current_stats,
            "budget_status": "healthy",
            "efficiency_status": "good",
            "cost_trend": "stable",
        }

        if "budget_status" in current_stats:
            indicators["budget_status"] = current_stats["budget_status"]

        # Analisa eficiência baseada em métricas históricas
        if current_stats.get("cost_per_item", 0) > 0.1:  # $0.10 per item
            indicators["efficiency_status"] = "attention"
        elif current_stats.get("cost_per_item", 0) > 0.05:  # $0.05 per item
            indicators["efficiency_status"] = "warning"

        return indicators

    def _generate_quick_insights(self, current_stats: Dict, trends: Dict) -> List[str]:
        """Gera insights rápidos baseados nos dados atuais."""
        insights = []

        if "error" not in current_stats:
            cost_per_item = current_stats.get("cost_per_item", 0)
            if cost_per_item > 0:
                if cost_per_item < 0.01:
                    insights.append(
                        "💚 Excelente eficiência de custo (<$0.01 por item)"
                    )
                elif cost_per_item < 0.05:
                    insights.append("✅ Boa eficiência de custo (<$0.05 por item)")
                else:
                    insights.append("⚠️ Custo por item acima do esperado")

        if trends.get("projected_daily_cost", 0) > 10:
            insights.append("📈 Projeção diária de custo elevada (>$10/dia)")

        efficiency = self.cost_tracker.get_efficiency_metrics()
        if efficiency.get("efficiency_score", 0) > 500:
            insights.append("🚀 Score de eficiência excelente")

        return insights

    def generate_cost_comparison_report(
        self, dataset_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Gera relatório comparativo de custos para diferentes cenários.

        Args:
            dataset_sizes: Tamanhos de dataset para análise

        Returns:
            Relatório comparativo detalhado
        """
        cache_key = f"cost_comparison_{hash(tuple(dataset_sizes or []))}"
        if self._is_cached_valid(cache_key):
            return self._cached_reports[cache_key]

        if dataset_sizes is None:
            dataset_sizes = [1000, 10000, 50000, 100000, 500000]

        comparison_data = {}

        for size in dataset_sizes:
            # Cria cenários para comparação
            scenarios = []
            for option in ProcessingOption:
                scenario = ProjectionScenario(
                    dataset_size=size,
                    processing_option=option,
                    avg_tokens_per_ticket=3200,  # Baseline
                    parallel_workers=4,
                )
                scenarios.append(scenario)

            # Compara cenários
            comparison = self.cost_projector.compare_scenarios(scenarios)
            comparison_data[f"{size:,}"] = comparison

        # Análise cross-dataset
        cross_analysis = self._analyze_cross_dataset_trends(comparison_data)

        report = {
            "report_type": "cost_comparison",
            "generated_at": datetime.now().isoformat(),
            "dataset_sizes_analyzed": dataset_sizes,
            "scenario_comparisons": comparison_data,
            "cross_dataset_analysis": cross_analysis,
            "recommendations": self._generate_cost_recommendations(comparison_data),
        }

        # Cache o relatório
        self._cached_reports[cache_key] = report

        return report

    def _analyze_cross_dataset_trends(self, comparison_data: Dict) -> Dict[str, Any]:
        """Analisa tendências entre diferentes tamanhos de dataset."""
        sizes = []
        best_costs = []
        worst_costs = []
        avg_costs = []

        for size_key, data in comparison_data.items():
            size = int(size_key.replace(",", ""))
            sizes.append(size)
            best_costs.append(data["best_option"]["total_cost"])
            worst_costs.append(data["worst_option"]["total_cost"])
            avg_costs.append(data["cost_range"]["avg"])

        # Calcula eficiência de escala
        cost_per_item_trend = [cost / size for cost, size in zip(best_costs, sizes)]
        scale_efficiency = (
            cost_per_item_trend[0] / cost_per_item_trend[-1]
            if len(cost_per_item_trend) > 1
            else 1.0
        )

        return {
            "scale_efficiency_factor": scale_efficiency,
            "cost_per_item_trend": cost_per_item_trend,
            "optimal_size_range": self._find_optimal_size_range(
                sizes, cost_per_item_trend
            ),
            "diminishing_returns_point": self._find_diminishing_returns(
                sizes, cost_per_item_trend
            ),
            "linear_scaling": abs(scale_efficiency - 1.0) < 0.1,
        }

    def _find_optimal_size_range(
        self, sizes: List[int], costs_per_item: List[float]
    ) -> Dict[str, int]:
        """Encontra faixa de tamanho ótima para eficiência."""
        if len(costs_per_item) < 2:
            return {"min": sizes[0], "max": sizes[0]}

        # Encontra ponto de maior eficiência
        min_cost_idx = costs_per_item.index(min(costs_per_item))

        return {
            "optimal_size": sizes[min_cost_idx],
            "min_efficient": sizes[max(0, min_cost_idx - 1)],
            "max_efficient": sizes[min(len(sizes) - 1, min_cost_idx + 1)],
        }

    def _find_diminishing_returns(
        self, sizes: List[int], costs_per_item: List[float]
    ) -> Optional[int]:
        """Encontra ponto onde retornos começam a diminuir."""
        if len(costs_per_item) < 3:
            return None

        # Calcula derivada (taxa de mudança)
        derivatives = []
        for i in range(1, len(costs_per_item)):
            derivative = (costs_per_item[i] - costs_per_item[i - 1]) / (
                sizes[i] - sizes[i - 1]
            )
            derivatives.append(derivative)

        # Encontra onde a melhoria para de ser significativa
        for i, derivative in enumerate(derivatives):
            if derivative > -0.000001:  # Melhoria muito pequena
                return sizes[i]

        return None

    def _generate_cost_recommendations(self, comparison_data: Dict) -> List[str]:
        """Gera recomendações baseadas na análise de custos."""
        recommendations = []

        # Analisa melhor opção geral
        all_best_options = [
            data["best_option"]["scenario"] for data in comparison_data.values()
        ]
        most_efficient_option = max(set(all_best_options), key=all_best_options.count)

        recommendations.append(f"🏆 Opção mais eficiente: {most_efficient_option}")

        # Analisa economia de escala
        small_dataset_cost = next(iter(comparison_data.values()))["best_option"][
            "cost_per_item"
        ]
        large_dataset_cost = list(comparison_data.values())[-1]["best_option"][
            "cost_per_item"
        ]

        if small_dataset_cost / large_dataset_cost > 2:
            recommendations.append(
                "📈 Forte economia de escala - considere processar em lotes maiores"
            )

        # Analisa diferença entre melhor e pior opção
        for size, data in comparison_data.items():
            savings = data["cost_range"]["savings_potential"]
            if savings > 100:  # $100 de economia potencial
                recommendations.append(
                    f"💰 Dataset {size}: economia de ${savings:.2f} otimizando processamento"
                )

        return recommendations

    def generate_roi_analysis(
        self, projection_scenario: ProjectionScenario = None
    ) -> Dict[str, Any]:
        """
        Gera análise detalhada de ROI.

        Args:
            projection_scenario: Cenário específico para análise (opcional)

        Returns:
            Análise completa de ROI
        """
        if projection_scenario is None:
            # Usa cenário padrão baseado em dados históricos
            avg_dataset_size = 19251  # Baseado no dataset atual
            projection_scenario = ProjectionScenario(
                dataset_size=avg_dataset_size,
                processing_option=ProcessingOption.OPTION_E,
                avg_tokens_per_ticket=3200,
                parallel_workers=4,
            )

        projection = self.cost_projector.project_cost(projection_scenario)
        roi_metrics = self.cost_projector.estimate_roi_metrics(projection)

        # Adiciona contexto histórico
        historical_context = self._get_historical_roi_context()

        # Análise de sensibilidade
        sensitivity_analysis = self._perform_sensitivity_analysis(projection_scenario)

        analysis = {
            "analysis_type": "roi_analysis",
            "generated_at": datetime.now().isoformat(),
            "scenario_analyzed": asdict(projection_scenario),
            "cost_projection": asdict(projection),
            "roi_metrics": roi_metrics,
            "historical_context": historical_context,
            "sensitivity_analysis": sensitivity_analysis,
            "business_recommendations": self._generate_business_recommendations(
                roi_metrics
            ),
        }

        return analysis

    def _get_historical_roi_context(self) -> Dict[str, Any]:
        """Fornece contexto ROI baseado em dados históricos."""
        sessions = self.cost_tracker._sessions_history
        if not sessions:
            return {"message": "Sem dados históricos disponíveis"}

        # Calcula métricas históricas
        total_processed = sum(s.dataset_size for s in sessions if s.dataset_size > 0)
        total_cost = sum(s.total_cost_usd for s in sessions)
        avg_cost_per_item = total_cost / max(total_processed, 1)

        # Tendência de eficiência
        if len(sessions) >= 2:
            recent_sessions = sessions[-3:]  # Últimas 3 sessões
            recent_avg_cost = sum(s.total_cost_usd for s in recent_sessions) / len(
                recent_sessions
            )
            older_sessions = sessions[:-3] if len(sessions) > 3 else sessions[:1]
            older_avg_cost = sum(s.total_cost_usd for s in older_sessions) / max(
                len(older_sessions), 1
            )

            efficiency_trend = (
                "improving" if recent_avg_cost < older_avg_cost else "stable"
            )
        else:
            efficiency_trend = "insufficient_data"

        return {
            "total_items_processed": total_processed,
            "total_historical_cost": total_cost,
            "historical_avg_cost_per_item": avg_cost_per_item,
            "sessions_analyzed": len(sessions),
            "efficiency_trend": efficiency_trend,
            "cost_stability": (
                "stable"
                if len(set(s.total_cost_usd for s in sessions[-5:])) < 3
                else "variable"
            ),
        }

    def _perform_sensitivity_analysis(
        self, base_scenario: ProjectionScenario
    ) -> Dict[str, Any]:
        """Realiza análise de sensibilidade variando parâmetros."""
        base_projection = self.cost_projector.project_cost(base_scenario)
        base_cost = base_projection.estimated_total_cost

        sensitivity_results = {}

        # Varia tamanho do dataset (±50%)
        for factor in [0.5, 1.5]:
            scenario = ProjectionScenario(
                dataset_size=int(base_scenario.dataset_size * factor),
                processing_option=base_scenario.processing_option,
                avg_tokens_per_ticket=base_scenario.avg_tokens_per_ticket,
                parallel_workers=base_scenario.parallel_workers,
            )
            projection = self.cost_projector.project_cost(scenario)
            sensitivity_results[f"dataset_size_{factor}x"] = {
                "cost_change_percent": (
                    (projection.estimated_total_cost - base_cost) / base_cost
                )
                * 100,
                "new_cost": projection.estimated_total_cost,
            }

        # Varia número de workers
        for workers in [2, 8]:
            scenario = ProjectionScenario(
                dataset_size=base_scenario.dataset_size,
                processing_option=base_scenario.processing_option,
                avg_tokens_per_ticket=base_scenario.avg_tokens_per_ticket,
                parallel_workers=workers,
            )
            projection = self.cost_projector.project_cost(scenario)
            sensitivity_results[f"workers_{workers}"] = {
                "cost_change_percent": (
                    (projection.estimated_total_cost - base_cost) / base_cost
                )
                * 100,
                "new_cost": projection.estimated_total_cost,
            }

        # Varia tokens por ticket (±30%)
        for factor in [0.7, 1.3]:
            scenario = ProjectionScenario(
                dataset_size=base_scenario.dataset_size,
                processing_option=base_scenario.processing_option,
                avg_tokens_per_ticket=base_scenario.avg_tokens_per_ticket * factor,
                parallel_workers=base_scenario.parallel_workers,
            )
            projection = self.cost_projector.project_cost(scenario)
            sensitivity_results[f"tokens_per_ticket_{factor}x"] = {
                "cost_change_percent": (
                    (projection.estimated_total_cost - base_cost) / base_cost
                )
                * 100,
                "new_cost": projection.estimated_total_cost,
            }

        return {
            "base_cost": base_cost,
            "parameter_variations": sensitivity_results,
            "most_sensitive_factor": max(
                sensitivity_results.items(),
                key=lambda x: abs(x[1]["cost_change_percent"]),
            )[0],
            "cost_range": {
                "min": min(r["new_cost"] for r in sensitivity_results.values()),
                "max": max(r["new_cost"] for r in sensitivity_results.values()),
            },
        }

    def _generate_business_recommendations(self, roi_metrics: Dict) -> List[str]:
        """Gera recomendações de negócio baseadas nas métricas ROI."""
        recommendations = []

        roi_percentage = roi_metrics["roi_metrics"]["roi_percentage"]
        if roi_percentage > 1000:  # 1000% ROI
            recommendations.append(
                "🚀 ROI excepcional - expandir implementação imediatamente"
            )
        elif roi_percentage > 500:  # 500% ROI
            recommendations.append("💰 ROI excelente - considerar escalonamento")
        elif roi_percentage > 100:  # 100% ROI
            recommendations.append("✅ ROI positivo - implementação viável")
        else:
            recommendations.append("⚠️ ROI baixo - revisar estratégia de implementação")

        payback_ratio = roi_metrics["roi_metrics"]["payback_ratio"]
        if payback_ratio > 10:
            recommendations.append("⚡ Payback muito rápido - prioridade alta")
        elif payback_ratio > 3:
            recommendations.append("📈 Payback rápido - implementar em curto prazo")

        efficiency_multiplier = roi_metrics["roi_metrics"]["efficiency_multiplier"]
        if efficiency_multiplier > 1000:
            recommendations.append(
                "🎯 Automação é 1000x+ mais eficiente que processo manual"
            )

        break_even = roi_metrics["business_impact"]["break_even_volume"]
        if break_even < 1000:
            recommendations.append(
                "✨ Break-even em menos de 1000 tickets - viabilidade alta"
            )

        return recommendations

    def export_dashboard_report(
        self, report_type: str = "comprehensive", filename: str = None
    ) -> Path:
        """
        Exporta relatório do dashboard em formato JSON/CSV.

        Args:
            report_type: Tipo de relatório (comprehensive, summary, roi_only)
            filename: Nome do arquivo (opcional)

        Returns:
            Path do arquivo gerado
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"roi_dashboard_{report_type}_{timestamp}"

        if report_type == "comprehensive":
            data = {
                "overview": self.generate_real_time_overview(),
                "cost_comparison": self.generate_cost_comparison_report(),
                "roi_analysis": self.generate_roi_analysis(),
            }
        elif report_type == "summary":
            data = self.generate_real_time_overview()
        elif report_type == "roi_only":
            data = self.generate_roi_analysis()
        else:
            raise ValueError(f"Tipo de relatório inválido: {report_type}")

        # Exporta como JSON
        json_file = self.storage_dir / f"{filename}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Também exporta sumário como CSV para análise externa
        csv_file = self.storage_dir / f"{filename}_summary.csv"
        self._export_summary_csv(data, csv_file)

        print(f"📊 Dashboard report exportado:")
        print(f"   JSON: {json_file}")
        print(f"   CSV:  {csv_file}")

        return json_file

    def _export_summary_csv(self, data: Dict, csv_file: Path):
        """Exporta sumário dos dados em formato CSV."""
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["ROI Dashboard Summary"])
            writer.writerow(["Generated:", datetime.now().isoformat()])
            writer.writerow([])

            # Overview atual
            if "overview" in data:
                current = data["overview"].get("current_session", {})
                if "error" not in current:
                    writer.writerow(["Current Session Metrics"])
                    writer.writerow(["Session ID:", current.get("session_id", "N/A")])
                    writer.writerow(
                        ["Duration (s):", current.get("duration_seconds", 0)]
                    )
                    writer.writerow(
                        ["Total Cost (USD):", current.get("total_cost_usd", 0)]
                    )
                    writer.writerow(
                        ["Cost per Item (USD):", current.get("cost_per_item", 0)]
                    )
                    writer.writerow(["Dataset Size:", current.get("dataset_size", 0)])
                    writer.writerow([])

            # ROI Analysis
            if "roi_analysis" in data:
                roi = data["roi_analysis"].get("roi_metrics", {})
                writer.writerow(["ROI Analysis"])
                writer.writerow(
                    [
                        "ROI Percentage:",
                        roi.get("roi_metrics", {}).get("roi_percentage", 0),
                    ]
                )
                writer.writerow(
                    [
                        "Cost Savings (USD):",
                        roi.get("cost_analysis", {}).get("cost_savings_usd", 0),
                    ]
                )
                writer.writerow(
                    [
                        "Time Savings (hours):",
                        roi.get("time_analysis", {}).get("time_savings_hours", 0),
                    ]
                )
                writer.writerow(
                    [
                        "Efficiency Multiplier:",
                        roi.get("roi_metrics", {}).get("efficiency_multiplier", 0),
                    ]
                )
                writer.writerow([])

            # Performance Summary
            if "overview" in data:
                perf = data["overview"].get("performance_summary", {})
                writer.writerow(["Performance Summary"])
                writer.writerow(
                    ["Total Historical Cost:", perf.get("total_historical_cost", 0)]
                )
                writer.writerow(["Total Sessions:", perf.get("total_sessions", 0)])
                efficiency = perf.get("efficiency_metrics", {})
                writer.writerow(
                    ["Avg Cost per Item:", efficiency.get("avg_cost_per_item", 0)]
                )
                writer.writerow(
                    ["Efficiency Score:", efficiency.get("efficiency_score", 0)]
                )

    def _is_cached_valid(self, cache_key: str) -> bool:
        """Verifica se cache é válido."""
        if cache_key not in self._cached_reports:
            return False

        report = self._cached_reports[cache_key]
        if "cached_at" not in report:
            report["cached_at"] = time.time()

        return time.time() - report["cached_at"] < self._cache_timeout

    def print_dashboard_summary(self):
        """Imprime sumário visual do dashboard."""
        overview = self.generate_real_time_overview()

        print("\n" + "=" * 80)
        print("💰 ROI DASHBOARD - VISÃO GERAL")
        print("=" * 80)

        # Status da sessão atual
        current = overview["current_session"]
        if "error" not in current:
            print(f"📊 SESSÃO ATUAL: {current['session_id']}")
            print(f"   Duração: {current['duration_seconds']:.1f}s")
            print(f"   Dataset: {current['dataset_size']:,} items")
            print(f"   Custo total: ${current['total_cost_usd']:.4f}")
            print(f"   Custo por item: ${current['cost_per_item']:.6f}")
            print(f"   Throughput: {current.get('tokens_per_second', 0):.1f} tokens/s")

            if "budget_limit_usd" in current:
                print(f"   Budget usado: {current['budget_used_percent']:.1f}%")
                print(f"   Status: {current['budget_status']}")
        else:
            print("❌ Nenhuma sessão ativa")

        print()

        # Performance histórica
        perf = overview["performance_summary"]
        print("📈 PERFORMANCE HISTÓRICA:")
        print(f"   Total gasto: ${perf['total_historical_cost']:.4f}")
        print(f"   Sessões: {perf['total_sessions']}")

        efficiency = perf.get("efficiency_metrics", {})
        if efficiency:
            print(f"   Custo médio/item: ${efficiency.get('avg_cost_per_item', 0):.6f}")
            print(f"   Score eficiência: {efficiency.get('efficiency_score', 0):.1f}")

        print()

        # Insights rápidos
        insights = overview.get("quick_insights", [])
        if insights:
            print("💡 INSIGHTS:")
            for insight in insights:
                print(f"   {insight}")

        print("=" * 80)
