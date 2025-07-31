import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
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
        "--export-excel",
        action="store_true",
        default=True,
        help="Exportar relatório Excel (padrão: ativado)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Exportar relatórios CSV (padrão: ativado)",
    )
    parser.add_argument(
        "--export-text",
        action="store_true",
        default=True,
        help="Exportar relatório de texto (padrão: ativado)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Desativar todas as exportações (apenas análise)",
    )
    args = parser.parse_args()

    load_dotenv()
    DATABASE_DIR = Path(__file__).parent.parent / "database"
    DATABASE_DIR.mkdir(exist_ok=True)

    # Para o modo analyze, o input file é opcional (usa arquivos do pipeline)
    if args.mode == "analyze":
        input_file = None  # Será detectado automaticamente pelo TicketReportGenerator
    else:
        # Seleciona o arquivo de entrada para outros modos
        input_file = select_input_file(DATABASE_DIR, args.input_file)

        if not input_file:
            print("\n❌ Nenhum arquivo foi selecionado. Operação cancelada.")
            return

    # Define se deve usar cache ou não
    use_cache = not args.no_cache

    try:
        if args.mode in ["categorize", "all"]:
            categorizer = TicketCategorizer(
                os.getenv("GOOGLE_API_KEY"),
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache,
            )
            categories_file = categorizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Categorização concluída: {categories_file}")

        if args.mode in ["summarize", "all"]:
            summarizer = TicketSummarizer(
                os.getenv("GOOGLE_API_KEY"),
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache,
            )
            summaries_file = summarizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Resumos concluídos: {summaries_file}")

        if args.mode in ["merge", "all"]:
            categories_file = DATABASE_DIR / "categorized_tickets.csv"
            summaries_file = DATABASE_DIR / "summarized_tickets.csv"

            if not categories_file.exists() or not summaries_file.exists():
                print("Arquivos necessários para merge não encontrados")
                return

            merger = TicketDataMerger(DATABASE_DIR)
            final_file = merger.merge_results(categories_file, summaries_file)
            print(f"Arquivo final gerado: {final_file}")

        if args.mode in ["analyze", "all"]:
            print("\n📊 Iniciando análise dos resultados do pipeline...")

            # Configurar opções de exportação
            export_reports = not args.no_export
            export_excel = args.export_excel and export_reports
            export_csv = args.export_csv and export_reports
            export_text = args.export_text and export_reports

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
                    export_reports=export_reports
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

    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        print(f"Tipo do erro: {type(e)}")
        import traceback

        print("Traceback completo:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
