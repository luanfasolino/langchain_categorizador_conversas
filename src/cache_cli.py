#!/usr/bin/env python3
"""
Cache CLI - Ferramenta de linha de comando para gerenciamento de cache.

Oferece funcionalidades de manutenção, monitoramento e relatórios via CLI.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from cache_manager import CacheManager
from cache_reporter import CacheReporter


class CacheCLI:
    """Interface de linha de comando para gerenciamento de cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            print(f"❌ Diretório de cache não existe: {cache_dir}")
            sys.exit(1)

        # Inicializa componentes
        self.cache_manager = CacheManager(
            cache_dir=self.cache_dir,
            max_cache_size_mb=1024,
            use_compression=True,
            enable_statistics=True,
        )

        self.reports_dir = self.cache_dir.parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.cache_reporter = CacheReporter(
            cache_manager=self.cache_manager,
            reports_dir=self.reports_dir,
            enable_continuous_monitoring=False,
        )

    def status(self) -> None:
        """Mostra status atual do cache."""
        print("📊 STATUS DO CACHE")
        print("=" * 50)

        stats = self.cache_manager.get_statistics()

        # Performance
        perf = stats["performance"]
        print("🎯 PERFORMANCE:")
        print(f"   Hit Rate: {perf['hit_rate_percent']:.1f}%")
        print(f"   Total Requests: {perf['hits'] + perf['misses']:,}")
        print(f"   Hits: {perf['hits']:,}")
        print(f"   Misses: {perf['misses']:,}")

        # Storage
        storage = stats["storage"]
        print("\n💾 ARMAZENAMENTO:")
        print(
            f"   Uso: {storage['usage_percent']:.1f}% ({storage['total_size_mb']:.1f}MB de {storage['max_size_mb']:.1f}MB)"
        )
        print(f"   Arquivos: {storage['total_files']:,} total")
        print(f"   Comprimidos: {storage['compressed_files']:,}")

        # Memory Cache
        memory = stats["memory_cache"]
        print("\n🧠 CACHE EM MEMÓRIA:")
        print(f"   Itens: {memory['items']:,}")
        print(f"   Tamanho: {memory['size_mb']:.1f}MB")

        # Operations
        ops = stats["operations"]
        print("\n⚙️  OPERAÇÕES:")
        print(f"   Saves: {ops['saves']:,}")
        print(f"   Errors: {ops['errors']:,}")
        print(f"   Último cleanup: {ops['last_cleanup']}")

        # Health Score
        health = self.cache_reporter.get_cache_health_score()
        if "score" in health:
            status_emoji = {
                "excellent": "🟢",
                "good": "🟡",
                "fair": "🟠",
                "poor": "🔴",
            }.get(health["status"], "⚪")

            print(f"\n{status_emoji} SAÚDE DO CACHE:")
            print(f"   Score: {health['score']:.1f}/100 ({health['status']})")

            # Mostra fatores críticos
            critical_factors = [
                f for f in health["factors"] if f["status"] in ["critical", "warning"]
            ]
            if critical_factors:
                print("   ⚠️  Fatores que precisam atenção:")
                for factor in critical_factors:
                    print(f"      - {factor['name']}: {factor['status']}")

    def cleanup(self, max_age_hours: int = 72, force: bool = False) -> None:
        """Executa limpeza de arquivos antigos."""
        if not force:
            print(
                f"🧹 Executando limpeza de arquivos com mais de {max_age_hours} horas..."
            )

        removed = self.cache_manager.cleanup_old_files(max_age_hours)

        if removed > 0:
            print(f"✅ Limpeza concluída: {removed} arquivos removidos")
        else:
            print("ℹ️  Nenhum arquivo antigo encontrado para remoção")

    def optimize(self) -> None:
        """Executa otimização completa do cache."""
        print("⚡ Executando otimização do cache...")

        results = self.cache_manager.optimize_cache()

        print("✅ Otimização concluída:")
        print(f"   Arquivos comprimidos: {results['files_compressed']}")
        print(f"   Arquivos removidos: {results['files_removed']}")
        print(f"   Espaço economizado: {results['space_saved_bytes']:,} bytes")

    def clear(self, confirm: bool = False) -> None:
        """Limpa todo o cache."""
        if not confirm:
            print("⚠️  ATENÇÃO: Esta operação irá remover TODOS os arquivos de cache!")
            response = input("Digite 'confirmo' para continuar: ")
            if response.lower() != "confirmo":
                print("❌ Operação cancelada")
                return

        print("🗑️  Limpando todo o cache...")
        success = self.cache_manager.clear_all()

        if success:
            print("✅ Cache limpo com sucesso")
        else:
            print("❌ Erro ao limpar cache")

    def report(
        self, hours: int = 24, format: str = "text", output: Optional[str] = None
    ) -> None:
        """Gera relatório de performance."""
        print(f"📈 Gerando relatório das últimas {hours} horas...")

        report = self.cache_reporter.generate_performance_report(
            hours_back=hours, include_charts=(format == "json")
        )

        if "error" in report:
            print(f"❌ {report['error']}")
            return

        if format == "json":
            output_data = json.dumps(report, indent=2, ensure_ascii=False, default=str)

            if output:
                output_file = Path(output)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_data)
                print(f"📄 Relatório salvo em: {output_file}")
            else:
                print(output_data)

        else:  # format == "text"
            self._print_text_report(report)

            if output:
                output_file = Path(output)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(self._format_text_report(report))
                print(f"📄 Relatório salvo em: {output_file}")

    def _print_text_report(self, report: dict) -> None:
        """Imprime relatório em formato texto."""
        print("\n📊 RELATÓRIO DE PERFORMANCE")
        print("=" * 60)

        period = report["period"]
        print(f"Período: {period['hours']} horas ({period['start']} a {period['end']})")
        print(f"Pontos de dados: {period['data_points']}")

        print("\n🎯 RESUMO DE PERFORMANCE:")
        summary = report["performance_summary"]

        hit_rate = summary["hit_rate"]
        print(
            f"   Hit Rate: {hit_rate['current']:.1f}% (média: {hit_rate['average']:.1f}%)"
        )

        usage = summary["cache_usage"]
        print(
            f"   Uso do Cache: {usage['current']:.1f}% (média: {usage['average']:.1f}%)"
        )

        files = summary["file_count"]
        print(f"   Arquivos: {files['current']:,} (média: {files['average']:.0f})")

        print("\n📈 TENDÊNCIAS:")
        trends = report["trends"]
        print(f"   Hit Rate: {trends['hit_rate']}")
        print(f"   Uso do Cache: {trends['cache_usage']}")
        print(f"   Saúde Geral: {trends['overall_health']}")

        print("\n💡 RECOMENDAÇÕES:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")

    def _format_text_report(self, report: dict) -> str:
        """Formata relatório como string de texto."""
        lines = []
        lines.append("RELATÓRIO DE PERFORMANCE DO CACHE")
        lines.append("=" * 60)

        period = report["period"]
        lines.append(f"Período: {period['hours']} horas")
        lines.append(f"Início: {period['start']}")
        lines.append(f"Fim: {period['end']}")
        lines.append(f"Pontos de dados: {period['data_points']}")

        lines.append("\nRESUMO DE PERFORMANCE:")
        summary = report["performance_summary"]

        hit_rate = summary["hit_rate"]
        lines.append(f"Hit Rate atual: {hit_rate['current']:.1f}%")
        lines.append(f"Hit Rate médio: {hit_rate['average']:.1f}%")

        usage = summary["cache_usage"]
        lines.append(f"Uso atual: {usage['current']:.1f}%")
        lines.append(f"Uso médio: {usage['average']:.1f}%")

        lines.append("\nTENDÊNCIAS:")
        trends = report["trends"]
        lines.append(f"Hit Rate: {trends['hit_rate']}")
        lines.append(f"Uso do Cache: {trends['cache_usage']}")
        lines.append(f"Saúde Geral: {trends['overall_health']}")

        lines.append("\nRECOMENDAÇÕES:")
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def export_csv(self, hours: int = 24, output: Optional[str] = None) -> None:
        """Exporta métricas para CSV."""
        try:
            if output:
                output_file = self.cache_reporter.export_metrics_csv(
                    filename=output, hours_back=hours
                )
            else:
                output_file = self.cache_reporter.export_metrics_csv(hours_back=hours)

            print(f"📊 Métricas exportadas para: {output_file}")

        except ValueError as e:
            print(f"❌ Erro: {e}")

    def monitor(self, interval: int = 300) -> None:
        """Inicia monitoramento contínuo."""
        print(f"👁️  Iniciando monitoramento contínuo (intervalo: {interval}s)")
        print("Pressione Ctrl+C para parar...")

        try:
            # Recria reporter com monitoramento habilitado
            monitor_reporter = CacheReporter(
                cache_manager=self.cache_manager,
                reports_dir=self.reports_dir,
                monitoring_interval=interval,
                enable_continuous_monitoring=True,
            )

            # Aguarda interrupção
            import time

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️  Monitoramento interrompido")
            monitor_reporter.stop_monitoring()


def main():
    """Função principal da CLI."""
    parser = argparse.ArgumentParser(
        description="Ferramenta de gerenciamento de cache do Categorizador de Conversas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s status                           # Mostra status do cache
  %(prog)s cleanup --max-age 48             # Remove arquivos com mais de 48h
  %(prog)s optimize                         # Otimiza cache (compressão, LRU)
  %(prog)s clear --confirm                  # Limpa todo o cache
  %(prog)s report --hours 12 --format json # Relatório das últimas 12h em JSON
  %(prog)s export-csv --hours 24           # Exporta métricas para CSV
  %(prog)s monitor --interval 60           # Monitora cache a cada 60s
        """,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("database/cache"),
        help="Diretório do cache (default: database/cache)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Status command
    subparsers.add_parser("status", help="Mostra status atual do cache")

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Remove arquivos antigos do cache"
    )
    cleanup_parser.add_argument(
        "--max-age", type=int, default=72, help="Idade máxima em horas (default: 72)"
    )
    cleanup_parser.add_argument(
        "--force", action="store_true", help="Executa sem confirmação"
    )

    # Optimize command
    subparsers.add_parser("optimize", help="Otimiza cache (compressão, LRU)")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Limpa todo o cache")
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Confirma operação sem prompt"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Gera relatório de performance"
    )
    report_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Horas para incluir no relatório (default: 24)",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Formato do relatório (default: text)",
    )
    report_parser.add_argument(
        "--output", type=str, help="Arquivo de saída (default: stdout)"
    )

    # Export CSV command
    csv_parser = subparsers.add_parser("export-csv", help="Exporta métricas para CSV")
    csv_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Horas para incluir na exportação (default: 24)",
    )
    csv_parser.add_argument(
        "--output", type=str, help="Nome do arquivo CSV (default: auto-gerado)"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Inicia monitoramento contínuo"
    )
    monitor_parser.add_argument(
        "--interval", type=int, default=300, help="Intervalo em segundos (default: 300)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    try:
        cli = CacheCLI(cache_dir=args.cache_dir)
    except Exception as e:
        print(f"❌ Erro ao inicializar CLI: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.command == "status":
            cli.status()

        elif args.command == "cleanup":
            cli.cleanup(max_age_hours=args.max_age, force=args.force)

        elif args.command == "optimize":
            cli.optimize()

        elif args.command == "clear":
            cli.clear(confirm=args.confirm)

        elif args.command == "report":
            cli.report(hours=args.hours, format=args.format, output=args.output)

        elif args.command == "export-csv":
            cli.export_csv(hours=args.hours, output=args.output)

        elif args.command == "monitor":
            cli.monitor(interval=args.interval)

    except Exception as e:
        print(f"❌ Erro ao executar comando: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
