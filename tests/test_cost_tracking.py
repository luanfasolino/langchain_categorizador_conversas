"""
Testes para o sistema de cost tracking e ROI analytics.
"""

import pytest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.cost_tracker import CostTracker, CostSession, TokenUsage
from src.cost_projector import CostProjector, ProjectionScenario, ProcessingOption
from src.roi_dashboard import ROIDashboard
from src.budget_monitor import BudgetMonitor, AlertConfig, AlertType, AlertLevel
from src.optimization_engine import OptimizationEngine


class TestCostTracker:
    """Testes para o sistema de tracking de custos."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def cost_tracker(self, temp_storage_dir):
        """Cria instância do CostTracker para testes."""
        return CostTracker(
            storage_dir=temp_storage_dir,
            budget_limit_usd=100.0
        )
    
    def test_cost_tracker_initialization(self, cost_tracker, temp_storage_dir):
        """Testa inicialização do CostTracker."""
        assert cost_tracker.storage_dir == temp_storage_dir
        assert cost_tracker.budget_limit_usd == 100.0
        assert cost_tracker._current_session is None
        assert len(cost_tracker._usage_history) == 0
        assert len(cost_tracker._sessions_history) == 0
    
    def test_session_management(self, cost_tracker):
        """Testa gerenciamento de sessões."""
        # Inicia sessão
        session_id = cost_tracker.start_session(
            session_id="test_session",
            dataset_size=1000,
            processing_mode="test"
        )
        
        assert session_id == "test_session"
        assert cost_tracker._current_session is not None
        assert cost_tracker._current_session.session_id == "test_session"
        assert cost_tracker._current_session.dataset_size == 1000
        
        # Finaliza sessão
        completed_session = cost_tracker.end_session()
        assert completed_session is not None
        assert completed_session.session_id == "test_session"
        assert cost_tracker._current_session is None
        assert len(cost_tracker._sessions_history) == 1
    
    def test_token_usage_tracking(self, cost_tracker):
        """Testa tracking de uso de tokens."""
        cost_tracker.start_session("test", 100)
        
        # Registra operação
        usage = cost_tracker.track_operation(
            input_tokens=1000,
            output_tokens=500,
            operation_type="categorize",
            phase="map"
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.cost_usd > 0
        
        # Verifica se foi adicionado ao histórico
        assert len(cost_tracker._usage_history) == 1
        
        # Verifica atualização da sessão
        session = cost_tracker._current_session
        assert session.total_input_tokens == 1000
        assert session.total_output_tokens == 500
        assert session.total_cost_usd == usage.cost_usd
        assert session.operations_count == 1
    
    def test_text_operation_tracking(self, cost_tracker):
        """Testa tracking por texto."""
        cost_tracker.start_session("test", 10)
        
        input_text = "Este é um texto de input para teste"
        output_text = "Este é o resultado do processamento"
        
        usage = cost_tracker.track_text_operation(
            input_text=input_text,
            output_text=output_text,
            operation_type="summarize"
        )
        
        assert usage.input_tokens == len(input_text) // 4
        assert usage.output_tokens == len(output_text) // 4
        assert usage.cost_usd > 0
    
    def test_budget_alerts(self, cost_tracker):
        """Testa alertas de budget."""
        cost_tracker.start_session("test", 10)
        
        # Simula operações que excedem 50% do budget
        for _ in range(100):  # Muitas operações para gerar custo
            cost_tracker.track_operation(1000, 1000, "expensive_op")
        
        # Verifica que alertas foram disparados
        assert len(cost_tracker._triggered_alerts) > 0
    
    def test_session_statistics(self, cost_tracker):
        """Testa estatísticas de sessão."""
        cost_tracker.start_session("test", 100)
        
        # Adiciona algumas operações
        for i in range(5):
            cost_tracker.track_operation(100 * (i + 1), 50 * (i + 1), f"op_{i}")
        
        stats = cost_tracker.get_current_session_stats()
        
        assert "error" not in stats
        assert stats["session_id"] == "test"
        assert stats["operations_count"] == 5
        assert stats["input_tokens"] == sum(100 * (i + 1) for i in range(5))
        assert stats["output_tokens"] == sum(50 * (i + 1) for i in range(5))
        assert stats["total_cost_usd"] > 0
        assert stats["dataset_size"] == 100
    
    def test_cost_trends_analysis(self, cost_tracker):
        """Testa análise de tendências."""
        # Cria histórico simulado
        for i in range(5):
            usage = TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                timestamp=time.time() - (i * 3600),  # Cada hora
                operation_type="test"
            )
            cost_tracker._usage_history.append(usage)
        
        trends = cost_tracker.get_cost_trends(hours_back=6)
        
        assert "error" not in trends
        assert trends["period_hours"] == 6
        assert trends["total_operations"] == 5
        assert trends["total_cost_usd"] > 0
        assert trends["avg_cost_per_hour"] > 0
    
    def test_efficiency_metrics(self, cost_tracker):
        """Testa métricas de eficiência."""
        # Cria sessões simuladas
        for i in range(3):
            session = CostSession(
                session_id=f"session_{i}",
                start_time=time.time() - 3600,
                end_time=time.time(),
                total_input_tokens=1000,
                total_output_tokens=500,
                total_cost_usd=0.1 * (i + 1),
                dataset_size=100,
                operations_count=10
            )
            cost_tracker._sessions_history.append(session)
        
        efficiency = cost_tracker.get_efficiency_metrics()
        
        assert "avg_cost_per_item" in efficiency
        assert "min_cost_per_item" in efficiency
        assert "max_cost_per_item" in efficiency
        assert "efficiency_score" in efficiency
        assert efficiency["avg_cost_per_item"] > 0


class TestCostProjector:
    """Testes para o sistema de projeção de custos."""
    
    @pytest.fixture
    def cost_projector(self):
        """Cria instância do CostProjector."""
        return CostProjector()
    
    def test_projector_initialization(self, cost_projector):
        """Testa inicialização do CostProjector."""
        assert cost_projector.BASELINE_COST_PER_TICKET == 0.048
        assert cost_projector.BASELINE_TOKENS_PER_TICKET == 3200
        assert len(cost_projector.PROCESSING_EFFICIENCY_FACTORS) == 5
    
    def test_cost_projection(self, cost_projector):
        """Testa projeção de custo básica."""
        scenario = ProjectionScenario(
            dataset_size=1000,
            processing_option=ProcessingOption.OPTION_E,
            avg_tokens_per_ticket=3200,
            parallel_workers=4
        )
        
        projection = cost_projector.project_cost(scenario)
        
        assert projection.scenario == scenario
        assert projection.estimated_total_cost > 0
        assert projection.cost_per_item > 0
        assert projection.estimated_tokens > 0
        assert projection.processing_time_hours > 0
        assert len(projection.confidence_interval) == 2
        assert isinstance(projection.risk_factors, list)
        assert isinstance(projection.optimization_opportunities, list)
    
    def test_scenario_comparison(self, cost_projector):
        """Testa comparação de cenários."""
        scenarios = []
        for option in [ProcessingOption.OPTION_A, ProcessingOption.OPTION_E]:
            scenario = ProjectionScenario(
                dataset_size=5000,
                processing_option=option,
                avg_tokens_per_ticket=3200
            )
            scenarios.append(scenario)
        
        comparison = cost_projector.compare_scenarios(scenarios)
        
        assert "scenarios_analyzed" in comparison
        assert comparison["scenarios_analyzed"] == 2
        assert "best_option" in comparison
        assert "worst_option" in comparison
        assert "cost_range" in comparison
        assert "detailed_projections" in comparison
    
    def test_roi_metrics(self, cost_projector):
        """Testa cálculo de métricas ROI."""
        scenario = ProjectionScenario(
            dataset_size=1000,
            processing_option=ProcessingOption.OPTION_E,
            avg_tokens_per_ticket=3200
        )
        
        projection = cost_projector.project_cost(scenario)
        roi_metrics = cost_projector.estimate_roi_metrics(projection)
        
        assert "cost_analysis" in roi_metrics
        assert "time_analysis" in roi_metrics
        assert "roi_metrics" in roi_metrics
        assert "business_impact" in roi_metrics
        
        cost_analysis = roi_metrics["cost_analysis"]
        assert cost_analysis["automated_cost_usd"] > 0
        assert cost_analysis["manual_cost_usd"] > 0
        assert cost_analysis["cost_savings_usd"] > 0
    
    def test_calibration_with_historical_data(self):
        """Testa calibração com dados históricos."""
        historical_sessions = [
            {
                "dataset_size": 1000,
                "total_cost_usd": 50.0,
                "total_input_tokens": 1500000,
                "total_output_tokens": 500000
            },
            {
                "dataset_size": 2000,
                "total_cost_usd": 95.0,
                "total_input_tokens": 3000000,
                "total_output_tokens": 1000000
            }
        ]
        
        projector = CostProjector(historical_sessions)
        
        # Verifica se baseline foi calibrado
        assert projector._calibrated_baseline is not None
        assert projector._calibrated_baseline["samples"] == 2
        assert projector._calibrated_baseline["cost_per_ticket"] > 0


class TestROIDashboard:
    """Testes para o dashboard ROI."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def cost_tracker_with_data(self, temp_storage_dir):
        """Cria CostTracker com dados simulados."""
        tracker = CostTracker(storage_dir=temp_storage_dir)
        
        # Adiciona sessão simulada
        session = CostSession(
            session_id="test_session",
            start_time=time.time() - 3600,
            end_time=time.time(),
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_cost_usd=2.0,
            dataset_size=1000,
            operations_count=50
        )
        tracker._sessions_history.append(session)
        
        return tracker
    
    @pytest.fixture
    def roi_dashboard(self, cost_tracker_with_data, temp_storage_dir):
        """Cria instância do ROIDashboard."""
        return ROIDashboard(
            cost_tracker=cost_tracker_with_data,
            storage_dir=temp_storage_dir
        )
    
    def test_dashboard_initialization(self, roi_dashboard):
        """Testa inicialização do dashboard."""
        assert roi_dashboard.cost_tracker is not None
        assert roi_dashboard.cost_projector is not None
        assert roi_dashboard.storage_dir.exists()
    
    def test_real_time_overview(self, roi_dashboard):
        """Testa geração de overview em tempo real."""
        overview = roi_dashboard.generate_real_time_overview()
        
        assert "dashboard_timestamp" in overview
        assert "current_session" in overview
        assert "performance_summary" in overview
        assert "operation_analysis" in overview
        assert "status_indicators" in overview
        assert "quick_insights" in overview
    
    def test_cost_comparison_report(self, roi_dashboard):
        """Testa relatório de comparação de custos."""
        dataset_sizes = [1000, 10000]
        comparison = roi_dashboard.generate_cost_comparison_report(dataset_sizes)
        
        assert "report_type" in comparison
        assert comparison["report_type"] == "cost_comparison"
        assert "dataset_sizes_analyzed" in comparison
        assert comparison["dataset_sizes_analyzed"] == dataset_sizes
        assert "scenario_comparisons" in comparison
        assert len(comparison["scenario_comparisons"]) == 2
    
    def test_roi_analysis(self, roi_dashboard):
        """Testa análise ROI."""
        analysis = roi_dashboard.generate_roi_analysis()
        
        assert "analysis_type" in analysis
        assert analysis["analysis_type"] == "roi_analysis"
        assert "scenario_analyzed" in analysis
        assert "cost_projection" in analysis
        assert "roi_metrics" in analysis
        assert "business_recommendations" in analysis
    
    def test_dashboard_export(self, roi_dashboard, temp_storage_dir):
        """Testa export do dashboard."""
        report_file = roi_dashboard.export_dashboard_report("summary")
        
        assert report_file.exists()
        assert report_file.suffix == ".json"
        
        # Verifica se CSV também foi criado
        csv_file = report_file.with_name(report_file.stem + "_summary.csv")
        assert csv_file.exists()


class TestBudgetMonitor:
    """Testes para o monitor de budget."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def budget_monitor(self, temp_storage_dir):
        """Cria instância do BudgetMonitor."""
        return BudgetMonitor(storage_dir=temp_storage_dir)
    
    @pytest.fixture
    def cost_tracker_mock(self):
        """Cria mock do CostTracker."""
        mock = Mock()
        mock.get_current_session_stats.return_value = {
            "session_id": "test",
            "total_cost_usd": 50.0,
            "duration_seconds": 3600,
            "cost_per_item": 0.05
        }
        return mock
    
    def test_monitor_initialization(self, budget_monitor):
        """Testa inicialização do monitor."""
        assert budget_monitor.storage_dir.exists()
        assert not budget_monitor._monitoring_active
        assert len(budget_monitor._alert_configs) > 0  # Alertas padrão
    
    def test_alert_configuration(self, budget_monitor):
        """Testa configuração de alertas."""
        config = AlertConfig(
            alert_type=AlertType.BUDGET_THRESHOLD,
            alert_level=AlertLevel.WARNING,
            threshold_value=0.8
        )
        
        budget_monitor.add_alert_config(config)
        
        config_key = f"{config.alert_type.value}_{config.threshold_value}"
        assert config_key in budget_monitor._alert_configs
    
    def test_monitoring_lifecycle(self, budget_monitor, cost_tracker_mock):
        """Testa ciclo de vida do monitoramento."""
        # Inicia monitoramento
        budget_monitor.start_monitoring(cost_tracker_mock, budget_limit=100.0)
        assert budget_monitor._monitoring_active
        assert budget_monitor.cost_tracker == cost_tracker_mock
        assert budget_monitor.budget_limit == 100.0
        
        # Para monitoramento
        budget_monitor.stop_monitoring()
        assert not budget_monitor._monitoring_active
    
    def test_alert_statistics(self, budget_monitor):
        """Testa estatísticas de alertas."""
        stats = budget_monitor.get_alert_statistics()
        
        assert "total_alerts" in stats
        assert "active_alerts" in stats
        assert "alerts_by_type" in stats
        assert "alerts_by_level" in stats
        assert "monitoring_active" in stats
        assert "configured_alerts" in stats


class TestOptimizationEngine:
    """Testes para a engine de otimização."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Cria diretório temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def optimization_engine(self, temp_storage_dir):
        """Cria instância da OptimizationEngine."""
        return OptimizationEngine(storage_dir=temp_storage_dir)
    
    def test_engine_initialization(self, optimization_engine):
        """Testa inicialização da engine."""
        assert optimization_engine.storage_dir.exists()
        assert len(optimization_engine.efficiency_thresholds) > 0
    
    def test_performance_analysis(self, optimization_engine):
        """Testa análise de padrões de performance."""
        sessions_data = [
            {
                "session_id": "session_1",
                "duration_seconds": 3600,
                "dataset_size": 1000,
                "total_cost_usd": 50.0,
                "total_input_tokens": 100000,
                "total_output_tokens": 50000,
                "operations_count": 100
            },
            {
                "session_id": "session_2",
                "duration_seconds": 1800,
                "dataset_size": 500,
                "total_cost_usd": 30.0,
                "total_input_tokens": 60000,
                "total_output_tokens": 30000,
                "operations_count": 50
            }
        ]
        
        analysis = optimization_engine.analyze_performance_patterns(sessions_data)
        
        assert "sessions_analyzed" in analysis
        assert analysis["sessions_analyzed"] == 2
        assert "temporal_patterns" in analysis
        assert "efficiency_patterns" in analysis
        assert "resource_patterns" in analysis
        assert "overall_health_score" in analysis
    
    def test_optimization_recommendations(self, optimization_engine):
        """Testa geração de recomendações."""
        # Análise simulada com problemas de eficiência
        analysis = {
            "efficiency_patterns": {
                "average_cost_per_item": 0.25,  # Alto custo
                "average_tokens_per_second": 30,  # Baixo throughput
                "cost_per_item_trend": "deteriorating"
            },
            "resource_patterns": {
                "average_parallel_workers": 2  # Baixo paralelismo
            },
            "cache_patterns": {
                "hit_rate": 0.6,  # Baixo hit rate
                "compression_ratio": 0.2  # Baixa compressão
            },
            "operation_patterns": {
                "most_expensive_operation": "expensive_op",
                "operation_efficiencies": {
                    "expensive_op": {
                        "cost_per_operation": 0.02,
                        "operations_count": 1000
                    }
                }
            }
        }
        
        recommendations = optimization_engine.generate_optimization_recommendations(analysis)
        
        assert len(recommendations) > 0
        
        # Verifica estrutura das recomendações
        for rec in recommendations:
            assert hasattr(rec, 'title')
            assert hasattr(rec, 'category')
            assert hasattr(rec, 'impact_level')
            assert hasattr(rec, 'complexity')
            assert hasattr(rec, 'estimated_savings_usd')
            assert hasattr(rec, 'priority_score')
            assert isinstance(rec.implementation_steps, list)
            assert isinstance(rec.risks, list)
    
    def test_recommendation_prioritization(self, optimization_engine):
        """Testa priorização de recomendações."""
        from src.optimization_engine import OptimizationRecommendation, OptimizationCategory, ImpactLevel, ImplementationComplexity
        
        # Cria recomendações simuladas
        recommendations = [
            OptimizationRecommendation(
                id="low_impact",
                category=OptimizationCategory.COST_REDUCTION,
                title="Low impact recommendation",
                description="Test",
                impact_level=ImpactLevel.LOW,
                complexity=ImplementationComplexity.COMPLEX,
                estimated_savings_usd=10.0,
                estimated_time_savings_hours=0,
                confidence_score=0.5,
                implementation_steps=[],
                risks=[],
                prerequisites=[],
                metrics_to_track=[],
                supporting_data={}
            ),
            OptimizationRecommendation(
                id="high_impact",
                category=OptimizationCategory.COST_REDUCTION,
                title="High impact recommendation",
                description="Test",
                impact_level=ImpactLevel.HIGH,
                complexity=ImplementationComplexity.EASY,
                estimated_savings_usd=100.0,
                estimated_time_savings_hours=10,
                confidence_score=0.9,
                implementation_steps=[],
                risks=[],
                prerequisites=[],
                metrics_to_track=[],
                supporting_data={}
            )
        ]
        
        # Testa priorização sem constraints
        prioritized = optimization_engine.prioritize_recommendations(recommendations)
        assert prioritized[0].id == "high_impact"  # Maior prioridade
        
        # Testa priorização com constraints
        constraints = {"max_complexity": ImplementationComplexity.EASY}
        constrained = optimization_engine.prioritize_recommendations(recommendations, constraints)
        assert len(constrained) == 1  # Apenas a fácil
        assert constrained[0].id == "high_impact"
    
    def test_implementation_plan(self, optimization_engine):
        """Testa geração de plano de implementação."""
        from src.optimization_engine import OptimizationRecommendation, OptimizationCategory, ImpactLevel, ImplementationComplexity
        
        recommendations = [
            OptimizationRecommendation(
                id="easy_high",
                category=OptimizationCategory.COST_REDUCTION,
                title="Easy high impact",
                description="Test",
                impact_level=ImpactLevel.HIGH,
                complexity=ImplementationComplexity.EASY,
                estimated_savings_usd=100.0,
                estimated_time_savings_hours=5,
                confidence_score=0.9,
                implementation_steps=[],
                risks=[],
                prerequisites=[],
                metrics_to_track=[],
                supporting_data={}
            ),
            OptimizationRecommendation(
                id="complex_critical",
                category=OptimizationCategory.PERFORMANCE_IMPROVEMENT,
                title="Complex critical",
                description="Test",
                impact_level=ImpactLevel.CRITICAL,
                complexity=ImplementationComplexity.COMPLEX,
                estimated_savings_usd=500.0,
                estimated_time_savings_hours=20,
                confidence_score=0.8,
                implementation_steps=[],
                risks=[],
                prerequisites=[],
                metrics_to_track=[],
                supporting_data={}
            )
        ]
        
        plan = optimization_engine.generate_implementation_plan(recommendations, 12)
        
        assert "timeline_weeks" in plan
        assert plan["timeline_weeks"] == 12
        assert "phases" in plan
        assert len(plan["phases"]) == 3
        assert "total_estimated_savings" in plan
        assert plan["total_estimated_savings"] == 600.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])