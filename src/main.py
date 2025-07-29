import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
from categorizer import TicketCategorizer
from summarizer import TicketSummarizer
from merger import TicketDataMerger

def get_available_files(database_dir):
    """Lista arquivos válidos na pasta database"""
    valid_extensions = ['.csv', '.xlsx', '.xls']
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
        print("💡 Coloque seu arquivo na pasta 'database' com extensão .csv, .xlsx ou .xls")
        return None
    
    # Se há apenas um arquivo, pergunta se quer usar
    if len(available_files) == 1:
        file = available_files[0]
        response = input(f"📁 Encontrado arquivo: {file.name}\n🤔 Usar este arquivo? (s/n): ").strip().lower()
        if response in ['s', 'sim', 'y', 'yes', '']:
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
            choice = input(f"\n🔢 Escolha o arquivo (1-{len(available_files)}) ou 'q' para sair: ").strip()
            
            if choice.lower() in ['q', 'quit', 'sair']:
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
    parser = argparse.ArgumentParser(description='Processa tickets de suporte.')
    parser.add_argument('--mode', choices=['categorize', 'summarize', 'merge', 'all'], 
                       required=True, help='Modo de execução')
    parser.add_argument('--nrows', type=int, default=None, 
                       help='Número de linhas para processar (opcional)')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Desativa o uso de cache (processa tudo novamente)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número máximo de workers para processamento paralelo')
    parser.add_argument('--input-file', type=str, default=None,
                       help='Caminho para o arquivo de entrada (CSV ou Excel)')
    args = parser.parse_args()
    
    load_dotenv()
    DATABASE_DIR = Path(__file__).parent.parent / "database"
    DATABASE_DIR.mkdir(exist_ok=True)
    
    # Seleciona o arquivo de entrada
    input_file = select_input_file(DATABASE_DIR, args.input_file)
    
    if not input_file:
        print("\n❌ Nenhum arquivo foi selecionado. Operação cancelada.")
        return
    
    # Define se deve usar cache ou não
    use_cache = not args.no_cache
    
    try:
        if args.mode in ['categorize', 'all']:
            categorizer = TicketCategorizer(
                os.getenv("GOOGLE_API_KEY"), 
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache
            )
            categories_file = categorizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Categorização concluída: {categories_file}")
        
        if args.mode in ['summarize', 'all']:
            summarizer = TicketSummarizer(
                os.getenv("GOOGLE_API_KEY"), 
                DATABASE_DIR,
                max_workers=args.workers,
                use_cache=use_cache
            )
            summaries_file = summarizer.process_tickets(input_file, nrows=args.nrows)
            print(f"Resumos concluídos: {summaries_file}")
        
        if args.mode in ['merge', 'all']:
            categories_file = DATABASE_DIR / "categorized_tickets.csv"
            summaries_file = DATABASE_DIR / "summarized_tickets.csv"
            
            if not categories_file.exists() or not summaries_file.exists():
                print("Arquivos necessários para merge não encontrados")
                return
                
            merger = TicketDataMerger(DATABASE_DIR)
            final_file = merger.merge_results(categories_file, summaries_file)
            print(f"Arquivo final gerado: {final_file}")
            
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        print(f"Tipo do erro: {type(e)}")
        import traceback
        print("Traceback completo:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
