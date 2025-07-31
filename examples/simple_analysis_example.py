"""
Example usage of Simple Ticket Analyzer.

This example demonstrates how to use the SimpleTicketAnalyzer
to generate Excel, CSV, and text reports from categorized ticket data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from simple_ticket_analyzer import SimpleTicketAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def basic_analysis_example():
    """Basic example of ticket analysis."""
    print("="*60)
    print("BASIC TICKET ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SimpleTicketAnalyzer(storage_dir=Path("database/simple_reports"))
    
    # File paths
    categorized_file = Path("database/categorized_tickets.csv")
    
    if not categorized_file.exists():
        print(f"❌ File not found: {categorized_file}")
        print("Please ensure you have run the categorization process first.")
        return
    
    print(f"📁 Analyzing file: {categorized_file}")
    
    # Run analysis
    try:
        analysis = analyzer.analyze_categorized_tickets(categorized_file)
        
        # Print key results
        print(f"✅ Analysis completed!")
        print(f"📊 Total tickets: {analysis['metadata']['total_tickets']:,}")
        print(f"📈 Total categories: {analysis['statistical_summary']['unique_categories']}")
        print(f"🏆 Most common: {analysis['statistical_summary']['most_common_category']['name']} "
              f"({analysis['statistical_summary']['most_common_category']['count']} tickets)")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def export_all_formats_example():
    """Example of exporting to all formats."""
    print("\n" + "="*60)
    print("EXPORT ALL FORMATS EXAMPLE")
    print("="*60)
    
    analyzer = SimpleTicketAnalyzer(storage_dir=Path("database/complete_reports"))
    categorized_file = Path("database/categorized_tickets.csv")
    original_file = Path("database/tickets_conversas_19251_unique_v3.csv")
    
    if not categorized_file.exists():
        print(f"❌ Categorized file not found: {categorized_file}")
        return
    
    try:
        # Run complete analysis with all exports
        results = analyzer.run_complete_analysis(
            categorized_file=categorized_file,
            original_file=original_file if original_file.exists() else None,
            export_excel=True,
            export_csv=True,
            export_text=True
        )
        
        print("🎉 Complete analysis finished!")
        print(f"📊 Analyzed {results['total_tickets_analyzed']:,} tickets")
        print(f"📂 Generated {len(results['files_generated'])} files:")
        
        for file_path in results['files_generated']:
            file_name = Path(file_path).name
            if file_path.endswith('.xlsx'):
                print(f"  📗 Excel: {file_name}")
            elif file_path.endswith('.csv'):
                print(f"  📄 CSV: {file_name}")
            elif file_path.endswith('.txt'):
                print(f"  📝 Text: {file_name}")
        
        print(f"📁 All files saved to: {results['storage_directory']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in complete analysis: {str(e)}")
        return None


def custom_analysis_example():
    """Example of custom analysis for specific categories."""
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    analyzer = SimpleTicketAnalyzer(storage_dir=Path("database/custom_reports"))
    categorized_file = Path("database/categorized_tickets.csv")
    
    if not categorized_file.exists():
        print(f"❌ File not found: {categorized_file}")
        return
    
    try:
        import pandas as pd
        
        # Load data
        df = pd.read_csv(categorized_file, sep=';', encoding='utf-8-sig')
        
        # Run basic analysis
        analysis = analyzer.analyze_categorized_tickets(categorized_file)
        
        # Custom analysis: Focus on top 5 categories
        top_5_categories = analysis["category_distribution"]["top_5_categories"]
        
        print("🔍 Top 5 Categories Analysis:")
        print("-" * 40)
        
        custom_filename = f"top_5_categories_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Filter data for top 5 categories
        top_5_names = [cat["category"] for cat in top_5_categories]
        filtered_df = df[df['categoria'].isin(top_5_names)]
        
        # Export custom Excel with just top 5
        custom_excel = analyzer.storage_dir / f"{custom_filename}.xlsx"
        
        with pd.ExcelWriter(custom_excel, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = []
            for cat_info in top_5_categories:
                overview_data.append({
                    "Category": cat_info["category"],
                    "Count": cat_info["count"],
                    "Percentage": cat_info["percentage"]
                })
            
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Top_5_Overview', index=False)
            
            # Individual sheets for each category
            for cat_info in top_5_categories:
                category_name = cat_info["category"]
                cat_df = filtered_df[filtered_df['categoria'] == category_name]
                
                # Clean sheet name (Excel has naming restrictions)
                sheet_name = category_name.replace("/", "_").replace("\\", "_")[:31]
                cat_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"  📊 {category_name}: {cat_info['count']} tickets ({cat_info['percentage']}%)")
        
        print(f"📗 Custom Excel report: {custom_excel}")
        
    except Exception as e:
        print(f"❌ Error in custom analysis: {str(e)}")


def print_quick_stats():
    """Print quick statistics without generating files."""
    print("\n" + "="*60)
    print("QUICK STATS (NO FILE GENERATION)")
    print("="*60)
    
    categorized_file = Path("database/categorized_tickets.csv")
    
    if not categorized_file.exists():
        print(f"❌ File not found: {categorized_file}")
        return
    
    try:
        import pandas as pd
        
        df = pd.read_csv(categorized_file, sep=';', encoding='utf-8-sig')
        category_counts = df['categoria'].value_counts()
        
        print(f"📊 Total tickets: {len(df):,}")
        print(f"📈 Total categories: {len(category_counts)}")
        print(f"🏆 Most common: {category_counts.index[0]} ({category_counts.iloc[0]} tickets)")
        print(f"📉 Least common: {category_counts.index[-1]} ({category_counts.iloc[-1]} tickets)")
        print(f"📊 Average per category: {len(df) / len(category_counts):.1f}")
        
        print("\n🔝 Top 10 Categories:")
        for i, (category, count) in enumerate(category_counts.head(10).items(), 1):
            percentage = (count / len(df)) * 100
            print(f"  {i:2d}. {category:<30} {count:>6,} ({percentage:>5.1f}%)")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Run all examples."""
    print("🎯 Simple Ticket Analyzer Examples")
    print("=" * 80)
    
    # Run examples
    print("\n🚀 Running examples...")
    
    # 1. Quick stats (no file generation)
    print_quick_stats()
    
    # 2. Basic analysis
    analysis = basic_analysis_example()
    
    if analysis:
        # 3. Export all formats
        results = export_all_formats_example()
        
        if results:
            # 4. Custom analysis
            custom_analysis_example()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("📁 Check the database/ directory for generated reports")
    print("="*80)


if __name__ == "__main__":
    main()