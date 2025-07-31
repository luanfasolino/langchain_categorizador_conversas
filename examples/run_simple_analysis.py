"""
Run Simple Ticket Analysis - Demonstration script.

This script demonstrates the SimpleTicketAnalyzer using available data
and creates sample categorized data for testing purposes.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from simple_ticket_analyzer import SimpleTicketAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_categorized_data():
    """Create sample categorized data from available dataset."""
    print("📊 Creating sample categorized data...")
    
    # Load the available data
    data_file = Path("database/databaseOner.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return None
    
    try:
        # Read the data
        df = pd.read_csv(data_file, sep=';', encoding='utf-8')
        
        # Filter for TEXT category and USER/AGENT messages (exclude AI)
        filtered_df = df[
            (df['category'] == 'TEXT') & 
            (df['sender'].isin(['USER', 'AGENT'])) &
            (df['text'].notna()) &
            (df['text'].str.strip() != '')
        ].copy()
        
        print(f"📈 Filtered data: {len(filtered_df):,} messages from {len(df):,} total")
        
        # Create simple categories based on text analysis
        def categorize_text(text):
            """Simple categorization based on keywords."""
            text_lower = str(text).lower()
            
            if any(word in text_lower for word in ['obrigad', 'agradec', 'muito bom', 'excelente']):
                return 'Agradecimento/Satisfação'
            elif any(word in text_lower for word in ['oi', 'ola', 'bom dia', 'boa tarde', 'boa noite']):
                return 'Saudação'
            elif any(word in text_lower for word in ['problema', 'erro', 'não funciona', 'defeito']):
                return 'Problema Técnico'
            elif any(word in text_lower for word in ['dúvida', 'como', 'onde', 'quando', 'ajuda']):
                return 'Dúvida/Solicitação de Ajuda'
            elif any(word in text_lower for word in ['pedido', 'solicito', 'preciso', 'quero']):
                return 'Solicitação de Serviço'
            elif any(word in text_lower for word in ['reclamação', 'insatisfeito', 'ruim', 'péssimo']):
                return 'Reclamação'
            elif any(word in text_lower for word in ['informação', 'dados', 'horário', 'preço']):
                return 'Solicitação de Informação'
            elif any(word in text_lower for word in ['especialista', 'humano', 'atendente']):
                return 'Transferência para Humano'
            elif any(word in text_lower for word in ['finalizada', 'classifique', 'avaliar']):
                return 'Finalização/Avaliação'
            else:
                return 'Comunicação Geral'
        
        # Apply categorization
        filtered_df['categoria'] = filtered_df['text'].apply(categorize_text)
        
        # Select relevant columns for the categorized file
        categorized_df = filtered_df[['ticket_id', 'categoria']].copy()
        
        # Remove duplicates (same ticket_id might have multiple messages)
        # Keep the first occurrence of each ticket_id
        categorized_df = categorized_df.drop_duplicates(subset=['ticket_id'], keep='first')
        
        # Save categorized data
        output_file = Path("database/categorized_tickets.csv")
        categorized_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
        
        print(f"✅ Created categorized data: {len(categorized_df):,} unique tickets")
        print(f"📁 Saved to: {output_file}")
        
        # Show category distribution
        category_counts = categorized_df['categoria'].value_counts()
        print("\n📈 Category distribution:")
        for category, count in category_counts.head(10).items():
            percentage = (count / len(categorized_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")
        return None


def run_analysis_demo():
    """Run the complete analysis demonstration."""
    print("=" * 80)
    print("🎯 SIMPLE TICKET ANALYZER DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Create sample data if needed
    categorized_file = Path("database/categorized_tickets.csv")
    
    if not categorized_file.exists():
        print("📊 Categorized file not found. Creating sample data...")
        categorized_file = create_sample_categorized_data()
        if not categorized_file:
            print("❌ Failed to create sample data. Exiting.")
            return
    else:
        print(f"✅ Using existing categorized file: {categorized_file}")
    
    # Step 2: Initialize analyzer
    analyzer = SimpleTicketAnalyzer(storage_dir=Path("database/analysis_reports"))
    
    # Step 3: Run complete analysis
    print("\n🔍 Running complete ticket analysis...")
    
    try:
        results = analyzer.run_complete_analysis(
            categorized_file=categorized_file,
            original_file=Path("database/databaseOner.csv"),
            export_excel=True,
            export_csv=True,
            export_text=True
        )
        
        # Step 4: Display results
        print("\n" + "=" * 60)
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        print(f"✅ Analysis completed successfully!")
        print(f"📈 Total tickets analyzed: {results['total_tickets_analyzed']:,}")
        print(f"🏷️  Total categories found: {results['total_categories_found']}")
        print(f"📁 Storage directory: {results['storage_directory']}")
        
        print(f"\n📋 Generated {len(results['files_generated'])} files:")
        for file_path in results['files_generated']:
            file_name = Path(file_path).name
            if file_path.endswith('.xlsx'):
                print(f"  📗 Excel Report: {file_name}")
            elif file_path.endswith('.csv'):
                print(f"  📄 CSV Data: {file_name}")
            elif file_path.endswith('.txt'):
                print(f"  📝 Text Summary: {file_name}")
        
        # Step 5: Show key insights
        analysis_data = results['analysis_data']
        stats = analysis_data['statistical_summary']
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"  🏆 Most common category: {stats['most_common_category']['name']}")
        print(f"     └─ {stats['most_common_category']['count']} tickets ({stats['most_common_category']['percentage']}%)")
        print(f"  📉 Least common category: {stats['least_common_category']['name']}")
        print(f"     └─ {stats['least_common_category']['count']} tickets ({stats['least_common_category']['percentage']}%)")
        print(f"  📊 Average tickets per category: {stats['average_tickets_per_category']}")
        
        # Show top 5 categories
        top_5 = analysis_data['category_distribution']['top_5_categories']
        print(f"\n🔝 TOP 5 CATEGORIES:")
        for i, cat in enumerate(top_5, 1):
            print(f"  {i}. {cat['category']}: {cat['count']} tickets ({cat['percentage']}%)")
        
        print("\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("📁 Check the database/analysis_reports/ directory for all generated files")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")
        return None


def show_quick_preview():
    """Show a quick preview of the data without generating files."""
    print("\n" + "=" * 60)
    print("👀 QUICK DATA PREVIEW")
    print("=" * 60)
    
    categorized_file = Path("database/categorized_tickets.csv")
    
    if not categorized_file.exists():
        print("❌ No categorized data available for preview")
        return
    
    try:
        df = pd.read_csv(categorized_file, sep=';', encoding='utf-8-sig')
        
        print(f"📊 Dataset Overview:")
        print(f"  Total tickets: {len(df):,}")
        print(f"  Total categories: {df['categoria'].nunique()}")
        
        print(f"\n📈 Sample data:")
        print(df.head(10).to_string(index=False))
        
        print(f"\n🏷️  Category distribution (top 10):")
        category_counts = df['categoria'].value_counts()
        for i, (category, count) in enumerate(category_counts.head(10).items(), 1):
            percentage = (count / len(df)) * 100
            print(f"  {i:2d}. {category:<30} {count:>6,} ({percentage:>5.1f}%)")
            
    except Exception as e:
        print(f"❌ Error in preview: {str(e)}")


def main():
    """Main function."""
    print("🚀 Starting Simple Ticket Analyzer Demo")
    
    # Show quick preview first
    show_quick_preview()
    
    # Run full demonstration
    results = run_analysis_demo()
    
    if results:
        print(f"\n🎉 Demo completed! Check {results['storage_directory']} for reports.")
    else:
        print("\n❌ Demo failed. Check logs for details.")


if __name__ == "__main__":
    main()