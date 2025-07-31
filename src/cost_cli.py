"""
Interface de linha de comando para sistema de cost tracking e ROI analytics.

Fornece acesso fácil a todas as funcionalidades do sistema de tracking
de custos, projeções e otimizações via CLI.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from .cost_tracker import CostTracker
    from .cost_projector import CostProjector, ProjectionScenario, ProcessingOption
    from .roi_dashboard import ROIDashboard
    from .budget_monitor import BudgetMonitor, AlertConfig, AlertType, AlertLevel
    from .optimization_engine import OptimizationEngine
except ImportError:
    from cost_tracker import CostTracker
    from cost_projector import CostProjector, ProjectionScenario, ProcessingOption
    from roi_dashboard import ROIDashboard
    from budget_monitor import BudgetMonitor, AlertConfig, AlertType, AlertLevel
    from optimization_engine import OptimizationEngine


class CostCLI:
    """Interface CLI principal para o sistema de cost tracking."""

    def __init__(self, storage_dir: Path = None):
        """
        Inicializa o CLI.

        Args:
            storage_dir: Diretório base para armazenamento
        """
        self.storage_dir = storage_dir or Path("database")

        # Inicializa componentes
        self.cost_tracker = CostTracker(storage_dir=self.storage_dir / "cost_tracking")
        self.dashboard = ROIDashboard(
            cost_tracker=self.cost_tracker,
            storage_dir=self.storage_dir / "roi_analytics",
        )
        self.optimization_engine = OptimizationEngine(
            storage_dir=self.storage_dir / "optimization"
        )
        self.budget_monitor: Optional[BudgetMonitor] = None

    def setup_parser(self) -> argparse.ArgumentParser:
        """Configura parser de argumentos."""
        parser = argparse.ArgumentParser(
            description="Cost Tracking e ROI Analytics CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemplos de uso:

  # Iniciar sessão de tracking
  python cost_cli.py start-session --session-id "test_run_001" --dataset-size 1000

  # Rastrear operação
  python cost_cli.py track-operation --input-tokens 1000 --output-tokens 500 --operation-type "categorize"

  # Finalizar sessão
  python cost_cli.py end-session

  # Visualizar dashboard
  python cost_cli.py dashboard

  # Gerar projeções de custo
  python cost_cli.py project-cost --dataset-size 50000 --processing-option OPTION_E

  # Iniciar monitoramento de budget
  python cost_cli.py start-monitor --budget-limit 100.0

  # Gerar recomendações de otimização
  python cost_cli.py optimize

  # Exportar relatórios
  python cost_cli.py export --type comprehensive
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

        # Comandos de tracking
        self._add_tracking_commands(subparsers)

        # Comandos de dashboard
        self._add_dashboard_commands(subparsers)

        # Comandos de projeção
        self._add_projection_commands(subparsers)

        # Comandos de monitoring
        self._add_monitoring_commands(subparsers)

        # Comandos de otimização
        self._add_optimization_commands(subparsers)

        # Comandos de export
        self._add_export_commands(subparsers)

        return parser

    def _add_tracking_commands(self, subparsers):
        """Adiciona comandos de tracking."""
        # Start session
        start_parser = subparsers.add_parser(
            "start-session", help="Inicia sessão de tracking"
        )
        start_parser.add_argument(
            "--session-id", required=True, help="ID único da sessão"
        )
        start_parser.add_argument(
            "--dataset-size", type=int, default=0, help="Tamanho do dataset"
        )
        start_parser.add_argument(
            "--processing-mode", default="unknown", help="Modo de processamento"
        )
        start_parser.add_argument(
            "--budget-limit", type=float, help="Limite de budget em USD"
        )

        # Track operation
        track_parser = subparsers.add_parser(
            "track-operation", help="Rastreia operação"
        )
        track_parser.add_argument(
            "--input-tokens", type=int, required=True, help="Tokens de input"
        )
        track_parser.add_argument(
            "--output-tokens", type=int, required=True, help="Tokens de output"
        )
        track_parser.add_argument(
            "--operation-type", default="unknown", help="Tipo da operação"
        )
        track_parser.add_argument(
            "--phase", default="unknown", help="Fase do processamento"
        )

        # Track text operation
        track_text_parser = subparsers.add_parser(
            "track-text", help="Rastreia operação por texto"
        )
        track_text_parser.add_argument(
            "--input-text", required=True, help="Texto de input"
        )
        track_text_parser.add_argument(
            "--output-text", required=True, help="Texto de output"
        )
        track_text_parser.add_argument(
            "--operation-type", default="unknown", help="Tipo da operação"
        )
        track_text_parser.add_argument(
            "--phase", default="unknown", help="Fase do processamento"
        )

        # End session
        subparsers.add_parser("end-session", help="Finaliza sessão atual")

        # Current stats
        subparsers.add_parser("stats", help="Mostra estatísticas da sessão atual")

    def _add_dashboard_commands(self, subparsers):
        """Adiciona comandos de dashboard."""
        # Dashboard overview
        subparsers.add_parser("dashboard", help="Mostra dashboard overview")

        # Cost trends
        trends_parser = subparsers.add_parser(
            "trends", help="Analisa tendências de custo"
        )
        trends_parser.add_argument(
            "--hours", type=int, default=24, help="Horas para análise"
        )

        # Cost comparison
        compare_parser = subparsers.add_parser(
            "compare", help="Compara cenários de custo"
        )
        compare_parser.add_argument(
            "--sizes",
            nargs="+",
            type=int,
            default=[1000, 10000, 50000],
            help="Tamanhos de dataset para comparar",
        )

    def _add_projection_commands(self, subparsers):
        """Adiciona comandos de projeção."""
        # Project cost
        project_parser = subparsers.add_parser("project-cost", help="Projeta custos")
        project_parser.add_argument(
            "--dataset-size", type=int, required=True, help="Tamanho do dataset"
        )
        project_parser.add_argument(
            "--processing-option",
            choices=[opt.value for opt in ProcessingOption],
            default="discovery_application",
            help="Opção de processamento",
        )
        project_parser.add_argument(
            "--tokens-per-ticket",
            type=float,
            default=3200,
            help="Tokens médios por ticket",
        )
        project_parser.add_argument(
            "--workers", type=int, default=4, help="Número de workers"
        )

        # ROI analysis
        roi_parser = subparsers.add_parser("roi-analysis", help="Analisa ROI")
        roi_parser.add_argument(
            "--dataset-size", type=int, default=19251, help="Tamanho do dataset"
        )
        roi_parser.add_argument(
            "--manual-cost", type=float, default=50.0, help="Custo manual por ticket"
        )
        roi_parser.add_argument(
            "--manual-time",
            type=float,
            default=0.5,
            help="Tempo manual por ticket (horas)",
        )

    def _add_monitoring_commands(self, subparsers):
        """Adiciona comandos de monitoring."""
        # Start monitoring
        monitor_parser = subparsers.add_parser(
            "start-monitor", help="Inicia monitoramento de budget"
        )
        monitor_parser.add_argument(
            "--budget-limit", type=float, required=True, help="Limite de budget em USD"
        )
        monitor_parser.add_argument(
            "--email-config", help="Arquivo de configuração de email"
        )

        # Stop monitoring
        subparsers.add_parser("stop-monitor", help="Para monitoramento de budget")

        # Alert status
        subparsers.add_parser("alerts", help="Mostra alertas ativos")

        # Acknowledge alerts
        ack_parser = subparsers.add_parser("ack-alert", help="Reconhece alerta")
        ack_parser.add_argument("--alert-id", required=True, help="ID do alerta")

    def _add_optimization_commands(self, subparsers):
        """Adiciona comandos de otimização."""
        # Generate recommendations
        opt_parser = subparsers.add_parser(
            "optimize", help="Gera recomendações de otimização"
        )
        opt_parser.add_argument(
            "--max-complexity",
            choices=["easy", "moderate", "complex"],
            help="Complexidade máxima",
        )
        opt_parser.add_argument(
            "--min-savings", type=float, help="Economia mínima em USD"
        )

        # Implementation plan
        plan_parser = subparsers.add_parser("plan", help="Gera plano de implementação")
        plan_parser.add_argument(
            "--timeline-weeks", type=int, default=12, help="Timeline em semanas"
        )

    def _add_export_commands(self, subparsers):
        """Adiciona comandos de export."""
        # Export reports
        export_parser = subparsers.add_parser("export", help="Exporta relatórios")
        export_parser.add_argument(
            "--type",
            choices=["comprehensive", "summary", "roi_only", "cost_only"],
            default="comprehensive",
            help="Tipo de relatório",
        )
        export_parser.add_argument("--filename", help="Nome do arquivo")
        export_parser.add_argument(
            "--format",
            choices=["json", "csv"],
            default="json",
            help="Formato do arquivo",
        )

    def run_command(self, args):
        """Executa comando baseado nos argumentos."""
        if args.command == "start-session":
            return self._cmd_start_session(args)
        elif args.command == "track-operation":
            return self._cmd_track_operation(args)
        elif args.command == "track-text":
            return self._cmd_track_text(args)
        elif args.command == "end-session":
            return self._cmd_end_session(args)
        elif args.command == "stats":
            return self._cmd_stats(args)
        elif args.command == "dashboard":
            return self._cmd_dashboard(args)
        elif args.command == "trends":
            return self._cmd_trends(args)
        elif args.command == "compare":
            return self._cmd_compare(args)
        elif args.command == "project-cost":
            return self._cmd_project_cost(args)
        elif args.command == "roi-analysis":
            return self._cmd_roi_analysis(args)
        elif args.command == "start-monitor":
            return self._cmd_start_monitor(args)
        elif args.command == "stop-monitor":
            return self._cmd_stop_monitor(args)
        elif args.command == "alerts":
            return self._cmd_alerts(args)
        elif args.command == "ack-alert":
            return self._cmd_ack_alert(args)
        elif args.command == "optimize":
            return self._cmd_optimize(args)
        elif args.command == "plan":
            return self._cmd_plan(args)
        elif args.command == "export":
            return self._cmd_export(args)
        else:
            print("❌ Comando não reconhecido")
            return False

    def _cmd_start_session(self, args):
        """Inicia sessão de tracking."""
        try:
            if args.budget_limit:
                self.cost_tracker.budget_limit_usd = args.budget_limit

            session_id = self.cost_tracker.start_session(
                session_id=args.session_id,
                dataset_size=args.dataset_size,
                processing_mode=args.processing_mode,
            )
            print(f"✅ Sessão '{session_id}' iniciada com sucesso")
            return True
        except Exception as e:
            print(f"❌ Erro ao iniciar sessão: {e}")
            return False

    def _cmd_track_operation(self, args):
        """Rastreia operação."""
        try:
            usage = self.cost_tracker.track_operation(
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                operation_type=args.operation_type,
                phase=args.phase,
            )
            print(f"✅ Operação rastreada: ${usage.cost_usd:.6f}")
            return True
        except Exception as e:
            print(f"❌ Erro ao rastrear operação: {e}")
            return False

    def _cmd_track_text(self, args):
        """Rastreia operação por texto."""
        try:
            usage = self.cost_tracker.track_text_operation(
                input_text=args.input_text,
                output_text=args.output_text,
                operation_type=args.operation_type,
                phase=args.phase,
            )
            print(f"✅ Operação rastreada: ${usage.cost_usd:.6f}")
            return True
        except Exception as e:
            print(f"❌ Erro ao rastrear operação: {e}")
            return False

    def _cmd_end_session(self, args):
        """Finaliza sessão."""
        try:
            session = self.cost_tracker.end_session()
            if session:
                print("✅ Sessão finalizada com sucesso")
            else:
                print("⚠️ Nenhuma sessão ativa para finalizar")
            return True
        except Exception as e:
            print(f"❌ Erro ao finalizar sessão: {e}")
            return False

    def _cmd_stats(self, args):
        """Mostra estatísticas."""
        try:
            stats = self.cost_tracker.get_current_session_stats()
            if "error" in stats:
                print("⚠️ Nenhuma sessão ativa")
                return False

            print("\n📊 ESTATÍSTICAS DA SESSÃO:")
            print(f"   Session ID: {stats['session_id']}")
            print(f"   Duração: {stats['duration_seconds']:.1f}s")
            print(
                f"   Total tokens: {stats['input_tokens'] + stats['output_tokens']:,}"
            )
            print(f"   Custo total: ${stats['total_cost_usd']:.4f}")
            print(f"   Custo por item: ${stats['cost_per_item']:.6f}")
            print(f"   Operações: {stats['operations_count']}")

            if "budget_limit_usd" in stats:
                print(f"   Budget usado: {stats['budget_used_percent']:.1f}%")

            return True
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return False

    def _cmd_dashboard(self, args):
        """Mostra dashboard."""
        try:
            self.dashboard.print_dashboard_summary()
            return True
        except Exception as e:
            print(f"❌ Erro ao gerar dashboard: {e}")
            return False

    def _cmd_trends(self, args):
        """Analisa tendências."""
        try:
            trends = self.cost_tracker.get_cost_trends(hours_back=args.hours)
            if "error" in trends:
                print(f"⚠️ {trends['error']}")
                return False

            print(f"\n📈 TENDÊNCIAS DE CUSTO ({args.hours}h):")
            print(f"   Total operações: {trends['total_operations']}")
            print(f"   Custo total: ${trends['total_cost_usd']:.4f}")
            print(f"   Custo médio/hora: ${trends['avg_cost_per_hour']:.4f}")
            print(f"   Tokens médio/hora: {trends['avg_tokens_per_hour']:,}")

            if "projected_daily_cost" in trends:
                print(f"   Projeção diária: ${trends['projected_daily_cost']:.2f}")

            return True
        except Exception as e:
            print(f"❌ Erro ao analisar tendências: {e}")
            return False

    def _cmd_compare(self, args):
        """Compara cenários."""
        try:
            comparison = self.dashboard.generate_cost_comparison_report(args.sizes)

            print(f"\n🔍 COMPARAÇÃO DE CENÁRIOS:")
            print(f"   Tamanhos analisados: {args.sizes}")

            for size_key, data in comparison["scenario_comparisons"].items():
                print(f"\n   📊 Dataset {size_key}:")
                best = data["best_option"]
                print(f"      Melhor opção: {best['scenario']}")
                print(f"      Custo: ${best['total_cost']:.4f}")
                print(f"      Custo/item: ${best['cost_per_item']:.6f}")
                print(
                    f"      Economia potencial: ${data['cost_range']['savings_potential']:.2f}"
                )

            return True
        except Exception as e:
            print(f"❌ Erro ao comparar cenários: {e}")
            return False

    def _cmd_project_cost(self, args):
        """Projeta custos."""
        try:
            scenario = ProjectionScenario(
                dataset_size=args.dataset_size,
                processing_option=ProcessingOption(args.processing_option),
                avg_tokens_per_ticket=args.tokens_per_ticket,
                parallel_workers=args.workers,
            )

            projection = self.dashboard.cost_projector.project_cost(scenario)

            print(f"\n💰 PROJEÇÃO DE CUSTO:")
            print(f"   Dataset: {args.dataset_size:,} items")
            print(f"   Opção: {args.processing_option}")
            print(f"   Custo total estimado: ${projection.estimated_total_cost:.4f}")
            print(f"   Custo por item: ${projection.cost_per_item:.6f}")
            print(f"   Tokens estimados: {projection.estimated_tokens:,}")
            print(f"   Tempo estimado: {projection.processing_time_hours:.1f}h")
            print(
                f"   Intervalo confiança: ${projection.confidence_interval[0]:.4f} - ${projection.confidence_interval[1]:.4f}"
            )

            if projection.risk_factors:
                print(f"\n   ⚠️ Fatores de risco:")
                for risk in projection.risk_factors:
                    print(f"      • {risk}")

            return True
        except Exception as e:
            print(f"❌ Erro ao projetar custo: {e}")
            return False

    def _cmd_roi_analysis(self, args):
        """Analisa ROI."""
        try:
            scenario = ProjectionScenario(
                dataset_size=args.dataset_size,
                processing_option=ProcessingOption.OPTION_E,
                avg_tokens_per_ticket=3200,
            )

            projection = self.dashboard.cost_projector.project_cost(scenario)
            roi_metrics = self.dashboard.cost_projector.estimate_roi_metrics(
                projection, args.manual_cost, args.manual_time
            )

            print(f"\n📈 ANÁLISE DE ROI:")
            cost_analysis = roi_metrics["cost_analysis"]
            roi_metrics_data = roi_metrics["roi_metrics"]

            print(f"   Dataset: {args.dataset_size:,} items")
            print(f"   Custo automatizado: ${cost_analysis['automated_cost_usd']:.2f}")
            print(f"   Custo manual: ${cost_analysis['manual_cost_usd']:.2f}")
            print(f"   Economia: ${cost_analysis['cost_savings_usd']:.2f}")
            print(f"   ROI: {roi_metrics_data['roi_percentage']:.1f}%")
            print(
                f"   Multiplicador eficiência: {roi_metrics_data['efficiency_multiplier']:.1f}x"
            )

            return True
        except Exception as e:
            print(f"❌ Erro ao analisar ROI: {e}")
            return False

    def _cmd_start_monitor(self, args):
        """Inicia monitoramento."""
        try:
            email_config = None
            if args.email_config:
                with open(args.email_config, "r") as f:
                    email_config = json.load(f)

            self.budget_monitor = BudgetMonitor(
                storage_dir=self.storage_dir / "budget_monitoring",
                email_config=email_config,
            )

            self.budget_monitor.start_monitoring(
                cost_tracker=self.cost_tracker, budget_limit=args.budget_limit
            )

            print(f"✅ Monitoramento iniciado com budget de ${args.budget_limit:.2f}")
            return True
        except Exception as e:
            print(f"❌ Erro ao iniciar monitoramento: {e}")
            return False

    def _cmd_stop_monitor(self, args):
        """Para monitoramento."""
        try:
            if self.budget_monitor:
                self.budget_monitor.stop_monitoring()
                print("✅ Monitoramento parado")
            else:
                print("⚠️ Nenhum monitoramento ativo")
            return True
        except Exception as e:
            print(f"❌ Erro ao parar monitoramento: {e}")
            return False

    def _cmd_alerts(self, args):
        """Mostra alertas."""
        try:
            if not self.budget_monitor:
                print("⚠️ Monitoramento não está ativo")
                return False

            active_alerts = self.budget_monitor.get_active_alerts()
            if not active_alerts:
                print("✅ Nenhum alerta ativo")
                return True

            print(f"\n🚨 ALERTAS ATIVOS ({len(active_alerts)}):")
            for alert in active_alerts:
                print(f"   • {alert.alert_id}: {alert.message}")
                print(f"     Level: {alert.alert_level.value}")
                print(f"     Type: {alert.alert_type.value}")

            return True
        except Exception as e:
            print(f"❌ Erro ao obter alertas: {e}")
            return False

    def _cmd_ack_alert(self, args):
        """Reconhece alerta."""
        try:
            if not self.budget_monitor:
                print("⚠️ Monitoramento não está ativo")
                return False

            success = self.budget_monitor.acknowledge_alert(args.alert_id)
            if success:
                print(f"✅ Alerta {args.alert_id} reconhecido")
            else:
                print(f"❌ Alerta {args.alert_id} não encontrado")

            return success
        except Exception as e:
            print(f"❌ Erro ao reconhecer alerta: {e}")
            return False

    def _cmd_optimize(self, args):
        """Gera recomendações."""
        try:
            # Coleta dados para análise
            sessions_data = [
                {
                    "session_id": s.session_id,
                    "duration_seconds": s.duration_seconds,
                    "dataset_size": s.dataset_size,
                    "total_cost_usd": s.total_cost_usd,
                    "total_input_tokens": s.total_input_tokens,
                    "total_output_tokens": s.total_output_tokens,
                    "operations_count": s.operations_count,
                }
                for s in self.cost_tracker._sessions_history
            ]

            operation_breakdown = self.cost_tracker.get_operation_breakdown()

            # Gera análise
            analysis = self.optimization_engine.analyze_performance_patterns(
                sessions_data, operation_breakdown
            )

            # Gera recomendações
            recommendations = (
                self.optimization_engine.generate_optimization_recommendations(analysis)
            )

            # Filtra por constraints se especificado
            constraints = {}
            if args.max_complexity:
                from optimization_engine import ImplementationComplexity

                complexity_map = {
                    "easy": ImplementationComplexity.EASY,
                    "moderate": ImplementationComplexity.MODERATE,
                    "complex": ImplementationComplexity.COMPLEX,
                }
                constraints["max_complexity"] = complexity_map[args.max_complexity]

            if args.min_savings:
                constraints["min_savings"] = args.min_savings

            if constraints:
                recommendations = self.optimization_engine.prioritize_recommendations(
                    recommendations, constraints
                )

            print(f"\n🎯 RECOMENDAÇÕES DE OTIMIZAÇÃO ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5
                print(f"\n   {i}. {rec.title}")
                print(f"      Categoria: {rec.category.value}")
                print(f"      Impacto: {rec.impact_level.value}")
                print(f"      Complexidade: {rec.complexity.value}")
                print(f"      Economia estimada: ${rec.estimated_savings_usd:.2f}")
                print(f"      Score prioridade: {rec.priority_score:.1f}")
                if rec.implementation_steps:
                    print(f"      Próximo passo: {rec.implementation_steps[0]}")

            return True
        except Exception as e:
            print(f"❌ Erro ao gerar recomendações: {e}")
            return False

    def _cmd_plan(self, args):
        """Gera plano de implementação."""
        try:
            # Similar ao optimize, mas foca no plano
            sessions_data = [
                {
                    "session_id": s.session_id,
                    "duration_seconds": s.duration_seconds,
                    "dataset_size": s.dataset_size,
                    "total_cost_usd": s.total_cost_usd,
                    "total_input_tokens": s.total_input_tokens,
                    "total_output_tokens": s.total_output_tokens,
                    "operations_count": s.operations_count,
                }
                for s in self.cost_tracker._sessions_history
            ]

            analysis = self.optimization_engine.analyze_performance_patterns(
                sessions_data
            )
            recommendations = (
                self.optimization_engine.generate_optimization_recommendations(analysis)
            )

            plan = self.optimization_engine.generate_implementation_plan(
                recommendations, args.timeline_weeks
            )

            print(f"\n📋 PLANO DE IMPLEMENTAÇÃO ({args.timeline_weeks} semanas):")

            for phase_name, phase_data in plan["phases"].items():
                print(f"\n   📅 {phase_name.upper()}:")
                print(f"      Semanas: {phase_data['weeks']}")
                print(f"      Foco: {phase_data['focus']}")
                print(f"      Economia esperada: ${phase_data['expected_savings']:.2f}")
                print(f"      Recomendações: {len(phase_data['recommendations'])}")

            print(f"\n   💰 TOTAL ESTIMADO: ${plan['total_estimated_savings']:.2f}")

            return True
        except Exception as e:
            print(f"❌ Erro ao gerar plano: {e}")
            return False

    def _cmd_export(self, args):
        """Exporta relatórios."""
        try:
            if args.type == "cost_only":
                # Export apenas cost tracker
                report_file = self.cost_tracker.export_cost_report_csv(args.filename)
            else:
                # Export dashboard completo
                report_file = self.dashboard.export_dashboard_report(
                    report_type=args.type, filename=args.filename
                )

            print(f"✅ Relatório exportado: {report_file}")
            return True
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")
            return False


def main():
    """Função principal."""
    cli = CostCLI()
    parser = cli.setup_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    try:
        success = cli.run_command(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Operação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
