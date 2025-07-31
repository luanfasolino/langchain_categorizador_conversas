"""
Integrated Pipeline Example - Demonstração do sistema completo integrado.

Este exemplo mostra como usar o sistema completo de categorização e análise
de forma integrada, usando os componentes reais do pipeline LangChain.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ticket_report_generator import TicketReportGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_integrated_analysis():
    """
    Demonstração do sistema integrado de análise.
    
    Este exemplo mostra como o TicketReportGenerator trabalha com os 
    resultados reais do pipeline de categorização LangChain.
    """
    print("=" * 80)
    print("🔗 SISTEMA INTEGRADO DE CATEGORIZAÇÃO E ANÁLISE")
    print("=" * 80)
    
    # Configuração
    database_dir = Path("database")
    
    # Verificar se temos dados do pipeline
    categorized_file = database_dir / "categorized_tickets.csv"
    final_analysis_file = database_dir / "final_analysis.csv" 
    summarized_file = database_dir / "summarized_tickets.csv"
    
    print("📁 Verificando arquivos do pipeline...")
    files_status = {
        "Categorized Tickets": categorized_file.exists(),
        "Final Analysis": final_analysis_file.exists(),
        "Summarized Tickets": summarized_file.exists()
    }
    
    for file_name, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file_name}")
    
    if not files_status["Categorized Tickets"]:
        print("\n⚠️  Arquivo de tickets categorizados não encontrado!")
        print("💡 Execute primeiro: python src/main.py --mode categorize")
        print("💡 Ou execute o pipeline completo: python src/main.py --mode all")
        return False
    
    # Inicializar o analisador integrado
    print("\n🤖 Inicializando TicketReportGenerator...")
    
    try:
        report_generator = TicketReportGenerator(
            database_dir=database_dir,
            storage_dir=database_dir / "integrated_reports"
        )
        
        # Executar análise completa
        print("\n📊 Executando análise completa dos resultados do pipeline...")
        
        results = report_generator.process_pipeline_results(
            export_reports=True,
            filename_base="integrated_pipeline_analysis"
        )
        
        # Exibir resultados detalhados
        print("\n" + "=" * 60)
        print("📈 RESULTADOS DA ANÁLISE INTEGRADA")
        print("=" * 60)
        
        print(f"✅ Análise concluída com sucesso!")
        print(f"📊 Total de tickets analisados: {results['total_tickets']:,}")
        print(f"🏷️  Total de categorias encontradas: {results['total_categories']}")
        
        # Informações sobre exportação
        if results['export_info']['total_files'] > 0:
            print(f"\n📁 Relatórios exportados:")
            print(f"   📂 Diretório: {results['export_info']['storage_directory']}")
            print(f"   📋 Total de arquivos: {results['export_info']['total_files']}")
            
            exported_files = results['export_info']['files_exported']
            for format_type, files in exported_files.items():
                if files:
                    emoji = {"excel": "📗", "csv": "📄", "text": "📝"}.get(format_type, "📄")
                    print(f"   {emoji} {format_type.upper()}: {len(files)} arquivo(s)")
                    for file_path in files:
                        print(f"      - {Path(file_path).name}")
        
        # Análise detalhada dos dados
        analysis_data = results['analysis_data']
        
        # Metadados
        print(f"\n📋 METADADOS DA ANÁLISE:")
        metadata = analysis_data['metadata']
        print(f"   🕐 Data da análise: {metadata['analysis_timestamp']}")
        print(f"   🔧 Pipeline: {metadata['pipeline_version']}")
        print(f"   📁 Arquivos processados: {metadata['data_summary']['files_processed']}")
        
        # Estatísticas principais
        print(f"\n📊 ESTATÍSTICAS PRINCIPAIS:")
        stats = analysis_data['statistical_summary']
        print(f"   📈 Total de categorias: {stats['unique_categories']}")
        print(f"   📊 Média por categoria: {stats['category_statistics']['mean_tickets_per_category']}")
        print(f"   📊 Mediana por categoria: {stats['category_statistics']['median_tickets_per_category']}")
        print(f"   📊 Categoria mínima: {stats['category_statistics']['min_tickets_in_category']} tickets")
        print(f"   📊 Categoria máxima: {stats['category_statistics']['max_tickets_in_category']} tickets")
        
        # Top 10 categorias
        print(f"\n🏆 TOP 10 CATEGORIAS (Pipeline Real):")
        top_categories = analysis_data['category_analysis']['top_categories']
        for i, cat in enumerate(top_categories[:10], 1):
            print(f"   {i:2d}. {cat['category']:<40} {cat['count']:>8,} ({cat['percentage']:>5.1f}%)")
        
        # Métricas de qualidade
        print(f"\n✅ MÉTRICAS DE QUALIDADE:")
        quality = analysis_data['quality_metrics']
        completeness = quality['data_completeness']
        print(f"   📊 Taxa de completude: {completeness['completeness_rate']}%")
        print(f"   ✅ Categorizações válidas: {completeness['valid_categorizations']:,}")
        print(f"   ❌ Categorizações vazias: {completeness['empty_categorizations']:,}")
        
        naming_quality = quality['category_naming_quality']
        print(f"   📝 Tamanho médio dos nomes: {naming_quality['average_category_name_length']:.1f} chars")
        
        # Performance do pipeline
        print(f"\n⚡ PERFORMANCE DO PIPELINE:")
        performance = analysis_data['pipeline_performance']
        processing = performance['processing_efficiency']
        print(f"   🔧 Componentes usados: {', '.join(processing['pipeline_components_used'])}")
        
        cache_perf = performance['cache_performance']
        if cache_perf.get('cache_enabled'):
            print(f"   💾 Cache ativado: {cache_perf.get('cache_files_count', 0)} arquivos")
        else:
            print(f"   💾 Cache: desativado")
        
        # Insights avançados (se disponível)
        if 'advanced_insights' in analysis_data:
            print(f"\n🔍 INSIGHTS AVANÇADOS:")
            advanced = analysis_data['advanced_insights']
            richness = advanced['data_richness']
            print(f"   📊 Registros merged: {richness['total_merged_records']:,}")
            print(f"   📋 Colunas disponíveis: {len(richness['available_columns'])}")
            
            if 'summarization_coverage' in advanced:
                sum_coverage = advanced['summarization_coverage']
                if sum_coverage.get('summary_available'):
                    print(f"   📝 Tickets com resumos: {sum_coverage['tickets_with_summaries']:,}")
                    print(f"   📝 Média de bullets: {sum_coverage['average_bullets_per_ticket']}")
        
        print("=" * 60)
        print("🎉 Análise integrada concluída com sucesso!")
        print("📁 Verifique os arquivos gerados no diretório de relatórios")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante a análise: {str(e)}")
        logger.error(f"Erro na análise integrada: {str(e)}")
        return False


def show_usage_examples():
    """Mostra exemplos de uso do sistema integrado."""
    print("\n" + "=" * 60)
    print("📖 EXEMPLOS DE USO DO SISTEMA INTEGRADO")
    print("=" * 60)
    
    examples = [
        {
            "description": "Pipeline completo (categorização + análise)",
            "command": "python src/main.py --mode all",
            "note": "Executa categorização, resumos, merge e análise automaticamente"
        },
        {
            "description": "Apenas análise (usa dados existentes)",
            "command": "python src/main.py --mode analyze",
            "note": "Analisa os resultados já processados do pipeline"
        },
        {
            "description": "Análise sem exportar relatórios",
            "command": "python src/main.py --mode analyze --no-export",
            "note": "Executa análise mas não gera arquivos de relatório"
        },
        {
            "description": "Categorização + análise",
            "command": "python src/main.py --mode categorize && python src/main.py --mode analyze",
            "note": "Executa categorização e depois gera relatórios"
        },
        {
            "description": "Pipeline com limitação de dados",
            "command": "python src/main.py --mode all --nrows 1000",
            "note": "Processa apenas 1000 linhas e gera análise completa"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   💻 {example['command']}")
        print(f"   💡 {example['note']}")
    
    print("\n" + "=" * 60)
    print("🔧 ARGUMENTOS DISPONÍVEIS:")
    print("   --mode: categorize, summarize, merge, analyze, all")
    print("   --nrows: Limitar número de linhas processadas")
    print("   --no-cache: Desativar cache (reprocessar tudo)")
    print("   --workers: Número de workers paralelos")
    print("   --no-export: Não exportar relatórios (apenas análise)")
    print("   --input-file: Arquivo específico para processar")


def compare_with_simple_analyzer():
    """Compara o novo sistema integrado com o SimpleTicketAnalyzer anterior."""
    print("\n" + "=" * 60)
    print("🔄 COMPARAÇÃO: INTEGRADO vs SIMPLES")
    print("=" * 60)
    
    comparison = [
        {
            "aspect": "Fonte de Dados",
            "simple": "Dados brutos + categorização por palavras-chave",
            "integrated": "Pipeline LangChain + categorização por IA"
        },
        {
            "aspect": "Qualidade das Categorias",
            "simple": "Categorização simples (10 categorias fixas)",
            "integrated": "Categorização inteligente (categorias dinâmicas)"
        },
        {
            "aspect": "Integração",
            "simple": "Sistema independente",
            "integrated": "Integrado com BaseProcessor e pipeline"
        },
        {
            "aspect": "Outputs",
            "simple": "Excel, CSV, Texto básicos",
            "integrated": "Excel, CSV, Texto + métricas de qualidade"
        },
        {
            "aspect": "Escalabilidade",
            "simple": "Limitado aos dados de exemplo",
            "integrated": "Funciona com qualquer resultado do pipeline"
        },
        {
            "aspect": "Uso",
            "simple": "Script separado",
            "integrated": "Modo nativo do main.py"
        }
    ]
    
    for comp in comparison:
        print(f"\n📊 {comp['aspect']}:")
        print(f"   📝 Simples: {comp['simple']}")
        print(f"   🚀 Integrado: {comp['integrated']}")
    
    print(f"\n✅ VANTAGENS DO SISTEMA INTEGRADO:")
    advantages = [
        "Usa categorização real por IA (Gemini)",
        "Integrado nativamente com o pipeline",
        "Métricas de qualidade do pipeline",
        "Análise de performance e cache",
        "Suporte a dados de resumos e merge",
        "Consistência com BaseProcessor",
        "Execução via main.py --mode analyze"
    ]
    
    for advantage in advantages:
        print(f"   ✅ {advantage}")


def main():
    """Função principal do exemplo."""
    print("🚀 Iniciando demonstração do sistema integrado")
    
    # Mostrar exemplos de uso
    show_usage_examples()
    
    # Comparação com sistema anterior
    compare_with_simple_analyzer()
    
    # Executar demonstração prática
    print(f"\n🎯 Executando demonstração prática...")
    success = demonstrate_integrated_analysis()
    
    if success:
        print(f"\n🎉 Demonstração concluída com sucesso!")
        print(f"💡 O sistema está pronto para uso em produção")
    else:
        print(f"\n⚠️  Demonstração não pôde ser executada completamente")
        print(f"💡 Execute primeiro o pipeline de categorização")


if __name__ == "__main__":
    main()