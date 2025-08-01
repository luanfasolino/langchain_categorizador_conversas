import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
import shutil
import time
from datetime import datetime, timedelta
from categorizer import TicketCategorizer
from summarizer import TicketSummarizer
from merger import TicketDataMerger
from ticket_report_generator import TicketReportGenerator


def get_available_files(database_dir):
    """Lista arquivos válidos na pasta database"""
    valid_extensions = [".csv", ".xlsx", ".xls"]
    files = []

    for file in database_dir.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            files.append(file)

    return sorted(files)


def select_input_file(database_dir, specified_file=None):
    """Seleciona o arquivo de entrada de forma interativa ou via parâmetro"""

    # Se foi especificado via linha de comando
    if specified_file:
        input_file = Path(specified_file)
        if not input_file.is_absolute():
            input_file = database_dir / input_file

        if not input_file.exists():
            print(f"❌ Arquivo especificado não encontrado: {input_file}")
            return None

        print(f"✅ Usando arquivo especificado: {input_file}")
        return input_file

    # Lista arquivos disponíveis
    available_files = get_available_files(database_dir)

    if not available_files:
        print(f"❌ Nenhum arquivo CSV ou Excel encontrado em: {database_dir}")
        print(
            "💡 Coloque seu arquivo na pasta 'database' com extensão .csv, .xlsx ou .xls"
        )
        return None

    # Se há apenas um arquivo, pergunta se quer usar
    if len(available_files) == 1:
        file = available_files[0]
        response = (
            input(f"📁 Encontrado arquivo: {file.name}\n🤔 Usar este arquivo? (s/n): ")
            .strip()
            .lower()
        )
        if response in ["s", "sim", "y", "yes", ""]:
            return file
        else:
            print("❌ Operação cancelada.")
            return None

    # Se há múltiplos arquivos, mostra menu
    print(f"\n📁 Arquivos encontrados em {database_dir}:")
    print("=" * 50)

    for i, file in enumerate(available_files, 1):
        file_size = file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"{i:2d}. {file.name:<30} ({size_mb:.1f} MB)")

    print("=" * 50)

    while True:
        try:
            choice = input(
                f"\n🔢 Escolha o arquivo (1-{len(available_files)}) ou 'q' para sair: "
            ).strip()

            if choice.lower() in ["q", "quit", "sair"]:
                print("❌ Operação cancelada.")
                return None

            file_index = int(choice) - 1
            if 0 <= file_index < len(available_files):
                selected_file = available_files[file_index]
                print(f"✅ Arquivo selecionado: {selected_file.name}")
                return selected_file
            else:
                print(f"❌ Opção inválida. Escolha entre 1 e {len(available_files)}")

        except ValueError:
            print("❌ Por favor, digite um número válido ou 'q' para sair")


def get_cache_statistics(cache_dir):
    """Analisa estatísticas do cache existente"""
    if not cache_dir.exists():
        return None
    
    cache_files = list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.pkl.gz"))
    
    if not cache_files:
        return None
    
    total_size = sum(f.stat().st_size for f in cache_files if f.exists())
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    # Encontra o arquivo mais antigo para calcular idade do cache
    oldest_file = min(cache_files, key=lambda f: f.stat().st_mtime)
    cache_age_hours = (time.time() - oldest_file.stat().st_mtime) / 3600
    
    return {
        "file_count": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size_mb,
        "total_size_gb": total_size_gb,
        "cache_age_hours": cache_age_hours,
        "oldest_file": oldest_file,
        "cache_files": cache_files
    }


def estimate_processing_time(cache_stats, estimated_total_tickets=1000):
    """Estima tempo de processamento com e sem cache"""
    if not cache_stats:
        return {
            "with_cache": "Não aplicável (sem cache)",
            "without_cache": "~2-4 horas (dependendo do dataset)",
            "time_saved": "N/A"
        }
    
    # Estimativas baseadas em experiência com o sistema
    # Estes valores podem ser ajustados baseados em medições reais
    base_time_minutes = max(120, estimated_total_tickets * 0.1)  # ~0.1 min por ticket
    
    # Se há cache, assume que pode economizar 60-80% do tempo na fase MAP
    cache_savings_percentage = 0.7  # 70% de economia
    with_cache_minutes = base_time_minutes * (1 - cache_savings_percentage)
    
    time_saved_minutes = base_time_minutes - with_cache_minutes
    
    def format_time(minutes):
        if minutes < 60:
            return f"~{int(minutes)} minutos"
        else:
            hours = minutes / 60
            return f"~{hours:.1f} horas"
    
    return {
        "with_cache": format_time(with_cache_minutes),
        "without_cache": format_time(base_time_minutes),
        "time_saved": format_time(time_saved_minutes)
    }


def format_cache_age(hours):
    """Formata idade do cache de forma amigável"""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minutos atrás"
    elif hours < 24:
        return f"{int(hours)} horas atrás"
    else:
        days = int(hours / 24)
        return f"{days} dias atrás"


def clear_all_cache(cache_dir):
    """Remove completamente todo o cache"""
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            return True
        return True
    except Exception as e:
        print(f"❌ Erro ao limpar cache: {str(e)}")
        return False


def interactive_cache_control(cache_dir, cache_control_arg=None):
    """
    Controla o cache de forma interativa ou via argumento.
    
    Returns:
        - True: Usar cache (continue)
        - False: Não usar cache (fresh start)
    """
    cache_stats = get_cache_statistics(cache_dir)
    
    # Se não há cache, processa fresh e permite gerar novo cache
    if not cache_stats:
        print("📦 Nenhum cache encontrado. Processamento fresh será executado.")
        return True  # CORREÇÃO: Permite gerar novo cache quando não há cache existente
    
    # Se foi especificado via argumento
    if cache_control_arg:
        if cache_control_arg == "continue":
            print("🚀 Usando cache existente (especificado via CLI)")
            return True
        elif cache_control_arg == "fresh":
            print("🔄 Iniciando fresh start (especificado via CLI)")
            if clear_all_cache(cache_dir):
                print("✅ Cache limpo com sucesso")
                return True  # CORREÇÃO: Permite gerar novo cache após limpeza
            else:
                print("❌ Erro ao limpar cache, continuando com cache existente")
                return True
    
    # Modo interativo
    print("\n" + "=" * 50)
    print("📦 CONTROLE DE CACHE DETECTADO")
    print("=" * 50)
    
    # Mostra estatísticas do cache
    if cache_stats["total_size_gb"] >= 1:
        size_str = f"{cache_stats['total_size_gb']:.1f}GB"
    else:
        size_str = f"{cache_stats['total_size_mb']:.1f}MB"
    
    print(f"✅ Cache encontrado: {cache_stats['file_count']} arquivos ({size_str})")
    print(f"📅 Criado: {format_cache_age(cache_stats['cache_age_hours'])}")
    
    # Estimativas de tempo
    time_estimates = estimate_processing_time(cache_stats)
    print(f"⚡ Estimativa com cache: {time_estimates['with_cache']}")
    print(f"🔄 Estimativa sem cache: {time_estimates['without_cache']}")
    if time_estimates['time_saved'] != "N/A":
        print(f"💾 Tempo economizado: {time_estimates['time_saved']}")
    
    print("\n⚠️  IMPORTANTE: Usar cache pode afetar a assertividade da categorização se:")
    print("   • Os prompts foram modificados desde a última execução")
    print("   • O dataset mudou significativamente")
    print("   • A configuração do modelo foi alterada")
    
    print("\nOpções disponíveis:")
    print("1. 🚀 CONTINUE - Usar cache existente (Mais rápido)")
    print("2. 🔄 FRESH - Deletar cache e processar tudo (Máxima assertividade)")
    
    while True:
        try:
            choice = input("\nEscolha (1-2): ").strip()
            
            if choice == "1":
                print("\n✅ Usando cache existente...")
                return True
            elif choice == "2":
                print("\n🔄 Iniciando fresh start...")
                print("⚠️  Deletando cache existente...")
                if clear_all_cache(cache_dir):
                    print("✅ Cache limpo com sucesso")
                    return True  # CORREÇÃO: Permite gerar novo cache após limpeza
                else:
                    print("❌ Erro ao limpar cache, continuando com cache existente")
                    return True
            else:
                print("❌ Opção inválida. Escolha 1 ou 2.")
                
        except KeyboardInterrupt:
            print("\n\n❌ Operação cancelada pelo usuário.")
            exit(1)
        except Exception as e:
            print(f"❌ Erro inesperado: {str(e)}")
            print("Continuando com cache existente por segurança...")
            return True


def main():
    parser = argparse.ArgumentParser(description="Processa tickets de suporte.")
    parser.add_argument(
        "--mode",
        choices=["categorize", "summarize", "merge", "analyze", "all"],
        required=True,
        help="Modo de execução",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Número de linhas para processar (opcional)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Desativa o uso de cache (processa tudo novamente)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número máximo de workers para processamento paralelo",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Caminho para o arquivo de entrada (CSV ou Excel)",
    )
    parser.add_argument(
        "--no-export-excel",
        action="store_true",
        help="Desativar exportação de relatório Excel",
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Desativar exportação de relatórios CSV",
    )
    parser.add_argument(
        "--no-export-text",
        action="store_true",
        help="Desativar exportação de relatório de texto",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Desativar todas as exportações (apenas análise)",
    )
    parser.add_argument(
        "--cache-control",
        choices=["continue", "fresh"],
        default=None,
        help="Controle de cache: 'continue' (usar cache), 'fresh' (limpar cache e processar tudo)",
    )
    args = parser.parse_args()

    load_dotenv()
    DATABASE_DIR = Path(__file__).parent.parent / "database"
    DATABASE_DIR.mkdir(exist_ok=True)

    # Para os modos analyze e merge, o input file é opcional (usa arquivos do pipeline)
    if args.mode in ["analyze", "merge"]:
        input_file = None  # Será detectado automaticamente
    else:
        # Seleciona o arquivo de entrada para outros modos
        input_file = select_input_file(DATABASE_DIR, args.input_file)

        if not input_file:
            print("\n❌ Nenhum arquivo foi selecionado. Operação cancelada.")
            return

    # Sistema de controle de cache
    cache_dir = DATABASE_DIR / "cache"
    
    # Se --no-cache foi especificado, sempre force fresh
    if args.no_cache:
        use_cache = False
        if cache_dir.exists():
            print("🔄 --no-cache especificado: processamento fresh será executado")
    else:
        # Usa o sistema interativo de controle de cache
        use_cache = interactive_cache_control(cache_dir, args.cache_control)

    try:
        if args.mode in ["categorize", "all"]:
            categorizer = TicketCategorizer(
                os.getenv("GOOGLE_API_KEY"),
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache,
            )
            categories_file = categorizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Classificação concluída (Map→Reduce→Classify): {categories_file}")
            
            # Log de uso de cache
            if use_cache:
                print("📦 Cache foi utilizado durante o processamento")
            else:
                print("🔄 Processamento fresh executado (sem cache)")

        if args.mode in ["summarize", "all"]:
            summarizer = TicketSummarizer(
                os.getenv("GOOGLE_API_KEY"),
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache,
            )
            summaries_file = summarizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Resumos concluídos: {summaries_file}")
            
            # Log de uso de cache
            if use_cache:
                print("📦 Cache foi utilizado durante o processamento")
            else:
                print("🔄 Processamento fresh executado (sem cache)")

        if args.mode in ["merge", "all"]:
            analysis_dir = DATABASE_DIR / "analysis_reports"
            categories_file = analysis_dir / "categorized_tickets.csv"
            summaries_file = analysis_dir / "summarized_tickets.csv"

            if not categories_file.exists() or not summaries_file.exists():
                print("⚠️  Arquivos necessários para merge não encontrados em:")
                print(f"   • Categorias: {categories_file}")
                print(f"   • Resumos: {summaries_file}")
                print("💡 Execute primeiro: python main.py --mode categorize e --mode summarize")
                return

            merger = TicketDataMerger(DATABASE_DIR)
            final_file = merger.merge_results(categories_file, summaries_file)
            print(f"Arquivo final gerado: {final_file}")

        if args.mode in ["analyze", "all"]:
            print("\n📊 Iniciando análise dos resultados do pipeline...")

            # Configurar opções de exportação
            export_reports = not args.no_export
            export_excel = not args.no_export_excel and export_reports
            export_csv = not args.no_export_csv and export_reports
            export_text = not args.no_export_text and export_reports

            # Inicializar o gerador de relatórios
            report_generator = TicketReportGenerator(
                api_key=os.getenv("GOOGLE_API_KEY"),
                database_dir=DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache,
            )

            try:
                # Processar resultados do pipeline
                results = report_generator.process_pipeline_results(
                    export_reports=export_reports,
                    export_excel=export_excel,
                    export_csv=export_csv,
                    export_text=export_text,
                )

                # Exibir resultados
                print("\n" + "=" * 60)
                print("📈 ANÁLISE DOS RESULTADOS DO PIPELINE")
                print("=" * 60)
                print(f"✅ Análise concluída com sucesso!")
                print(f"📊 Total de tickets analisados: {results['total_tickets']:,}")
                print(
                    f"🏷️  Total de categorias encontradas: {results['total_categories']}"
                )

                if export_reports and results["export_info"]["total_files"] > 0:
                    print(f"\n📁 Relatórios gerados:")
                    print(
                        f"   Total de arquivos: {results['export_info']['total_files']}"
                    )
                    print(
                        f"   Diretório: {results['export_info']['storage_directory']}"
                    )

                    exported_files = results["export_info"]["files_exported"]
                    if exported_files.get("excel"):
                        print(f"   📗 Excel: {len(exported_files['excel'])} arquivo(s)")
                    if exported_files.get("csv"):
                        print(f"   📄 CSV: {len(exported_files['csv'])} arquivo(s)")
                    if exported_files.get("text"):
                        print(f"   📝 Texto: {len(exported_files['text'])} arquivo(s)")

                # Mostrar insights principais
                analysis_data = results["analysis_data"]
                if "category_analysis" in analysis_data:
                    top_categories = analysis_data["category_analysis"][
                        "top_categories"
                    ][:5]
                    print(f"\n🔝 Top 5 Categorias:")
                    for i, cat in enumerate(top_categories, 1):
                        print(
                            f"   {i}. {cat['category']}: {cat['count']:,} tickets ({cat['percentage']}%)"
                        )

                print("=" * 60)

            except FileNotFoundError as e:
                print(f"\n❌ Erro: {str(e)}")
                print(
                    "💡 Execute primeiro a categorização: python main.py --mode categorize"
                )
            except Exception as e:
                print(f"\n❌ Erro durante a análise: {str(e)}")
                raise

        # Resumo final do uso de cache
        print("\n" + "=" * 60)
        print("📋 RESUMO DA SESSÃO DE PROCESSAMENTO")
        print("=" * 60)
        if use_cache:
            print("📦 Status do Cache: UTILIZADO")
            print("💡 Para máxima assertividade, considere usar --cache-control fresh")
        else:
            print("🔄 Status do Cache: PROCESSAMENTO FRESH")
            print("✅ Máxima assertividade garantida")
        
        # Mostrar estatísticas atuais do cache
        final_cache_stats = get_cache_statistics(cache_dir)
        if final_cache_stats:
            if final_cache_stats["total_size_gb"] >= 1:
                size_str = f"{final_cache_stats['total_size_gb']:.1f}GB"
            else:
                size_str = f"{final_cache_stats['total_size_mb']:.1f}MB"
            print(f"💾 Cache atual: {final_cache_stats['file_count']} arquivos ({size_str})")
        print("=" * 60)

    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        print(f"Tipo do erro: {type(e)}")
        import traceback

        print("Traceback completo:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
