"""
Simple Ticket Analysis and Reporting System.

This module provides simple Excel/CSV report generation for analyzing
the 19,251 categorized tickets instead of a complex web dashboard.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SimpleTicketAnalyzer:
    """
    Simple analyzer that generates Excel and CSV reports for ticket data.
    
    This provides the core functionality needed for Task 9 with a practical,
    file-based approach instead of a complex web dashboard.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("database/ticket_analysis_reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SimpleTicketAnalyzer initialized")

    def analyze_categorized_tickets(
        self, 
        categorized_file: Path, 
        original_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze categorized tickets and generate comprehensive statistics.
        
        Args:
            categorized_file: Path to categorized_tickets.csv
            original_file: Optional path to original dataset for additional analysis
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing categorized tickets from {categorized_file}")
        
        try:
            # Load categorized data
            categorized_df = pd.read_csv(categorized_file, sep=';', encoding='utf-8-sig')
            
            analysis = {
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "categorized_file": str(categorized_file),
                    "original_file": str(original_file) if original_file else None,
                    "total_tickets": len(categorized_df)
                },
                "category_distribution": self._analyze_category_distribution(categorized_df),
                "statistical_summary": self._generate_statistical_summary(categorized_df),
            }
            
            # Add original data analysis if available
            if original_file and original_file.exists():
                original_df = pd.read_csv(original_file, sep=';', encoding='utf-8-sig')
                analysis["original_data_analysis"] = self._analyze_original_data(original_df)
                analysis["cross_analysis"] = self._cross_analyze_data(categorized_df, original_df)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing tickets: {str(e)}")
            raise

    def _analyze_category_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of categories."""
        category_counts = df['categoria'].value_counts()
        total_tickets = len(df)
        
        distribution = {
            "total_categories": len(category_counts),
            "categories": {},
            "top_5_categories": [],
            "category_percentages": {}
        }
        
        # Detailed category analysis
        for category, count in category_counts.items():
            percentage = (count / total_tickets) * 100
            distribution["categories"][category] = {
                "count": int(count),
                "percentage": round(percentage, 2)
            }
            distribution["category_percentages"][category] = round(percentage, 2)
        
        # Top 5 categories
        distribution["top_5_categories"] = [
            {
                "category": category,
                "count": int(count),
                "percentage": round((count / total_tickets) * 100, 2)
            }
            for category, count in category_counts.head(5).items()
        ]
        
        return distribution

    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary of the data."""
        category_counts = df['categoria'].value_counts()
        
        return {
            "total_tickets": len(df),
            "unique_categories": len(category_counts),
            "most_common_category": {
                "name": category_counts.index[0],
                "count": int(category_counts.iloc[0]),
                "percentage": round((category_counts.iloc[0] / len(df)) * 100, 2)
            },
            "least_common_category": {
                "name": category_counts.index[-1],
                "count": int(category_counts.iloc[-1]),
                "percentage": round((category_counts.iloc[-1] / len(df)) * 100, 2)
            },
            "average_tickets_per_category": round(len(df) / len(category_counts), 2),
            "median_tickets_per_category": int(category_counts.median()),
            "category_distribution_stats": {
                "mean": round(category_counts.mean(), 2),
                "std": round(category_counts.std(), 2),
                "min": int(category_counts.min()),
                "max": int(category_counts.max())
            }
        }

    def _analyze_original_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze original dataset characteristics."""
        return {
            "total_records": len(df),
            "columns": list(df.columns),
            "text_stats": {
                "avg_text_length": round(df['text'].str.len().mean(), 2) if 'text' in df.columns else None,
                "min_text_length": int(df['text'].str.len().min()) if 'text' in df.columns else None,
                "max_text_length": int(df['text'].str.len().max()) if 'text' in df.columns else None,
            },
            "sender_distribution": df['sender'].value_counts().to_dict() if 'sender' in df.columns else {},
            "category_filter_stats": {
                "text_category_count": len(df[df['category'].str.lower() == 'text']) if 'category' in df.columns else None,
                "other_categories": df['category'].value_counts().to_dict() if 'category' in df.columns else {}
            }
        }

    def _cross_analyze_data(self, categorized_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Cross-analyze categorized vs original data."""
        return {
            "processing_efficiency": {
                "original_tickets": len(original_df),
                "categorized_tickets": len(categorized_df),
                "processing_rate": round((len(categorized_df) / len(original_df)) * 100, 2),
                "filtered_out": len(original_df) - len(categorized_df)
            },
            "data_quality": {
                "ticket_id_overlap": len(set(categorized_df['ticket_id']) & set(original_df['ticket_id'])) if 'ticket_id' in original_df.columns else None,
                "unique_tickets_categorized": len(categorized_df['ticket_id'].unique()),
                "unique_tickets_original": len(original_df['ticket_id'].unique()) if 'ticket_id' in original_df.columns else None
            }
        }

    def export_to_excel(
        self, 
        analysis: Dict[str, Any], 
        categorized_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export analysis to Excel file with multiple sheets.
        
        Args:
            analysis: Analysis results
            categorized_df: Categorized tickets DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to generated Excel file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ticket_analysis_report_{timestamp}.xlsx"
        
        excel_file = self.storage_dir / filename
        
        logger.info(f"Exporting analysis to Excel: {excel_file}")
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = self._prepare_summary_sheet(analysis)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Category Distribution
            category_data = []
            for category, info in analysis["category_distribution"]["categories"].items():
                category_data.append({
                    "Category": category,
                    "Count": info["count"],
                    "Percentage": info["percentage"]
                })
            
            category_df = pd.DataFrame(category_data)
            category_df = category_df.sort_values('Count', ascending=False)
            category_df.to_excel(writer, sheet_name='Category_Distribution', index=False)
            
            # Sheet 3: All Categorized Tickets
            categorized_df.to_excel(writer, sheet_name='All_Tickets', index=False)
            
            # Sheet 4: Top Categories Detail
            top_categories_detail = self._prepare_top_categories_detail(categorized_df)
            top_categories_detail.to_excel(writer, sheet_name='Top_Categories_Detail', index=False)
            
            # Sheet 5: Statistics
            stats_data = self._prepare_statistics_sheet(analysis)
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        logger.info(f"Excel report generated: {excel_file}")
        return excel_file

    def export_to_csv(
        self, 
        analysis: Dict[str, Any], 
        categorized_df: pd.DataFrame,
        base_filename: Optional[str] = None
    ) -> List[Path]:
        """
        Export analysis to multiple CSV files.
        
        Args:
            analysis: Analysis results
            categorized_df: Categorized tickets DataFrame
            base_filename: Optional base filename
            
        Returns:
            List of generated CSV file paths
        """
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ticket_analysis_{timestamp}"
        
        csv_files = []
        
        # 1. Category Distribution CSV
        category_data = []
        for category, info in analysis["category_distribution"]["categories"].items():
            category_data.append({
                "Category": category,
                "Count": info["count"],
                "Percentage": info["percentage"]
            })
        
        category_df = pd.DataFrame(category_data).sort_values('Count', ascending=False)
        category_csv = self.storage_dir / f"{base_filename}_category_distribution.csv"
        category_df.to_csv(category_csv, index=False, encoding='utf-8-sig')
        csv_files.append(category_csv)
        
        # 2. Summary Statistics CSV
        summary_data = self._prepare_summary_sheet(analysis)
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.storage_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        csv_files.append(summary_csv)
        
        # 3. All tickets CSV (copy of original with better name)
        all_tickets_csv = self.storage_dir / f"{base_filename}_all_categorized_tickets.csv"
        categorized_df.to_csv(all_tickets_csv, index=False, encoding='utf-8-sig')
        csv_files.append(all_tickets_csv)
        
        # 4. Top 10 categories detail CSV
        top_categories = categorized_df['categoria'].value_counts().head(10)
        top_detail_data = []
        for category in top_categories.index:
            category_tickets = categorized_df[categorized_df['categoria'] == category]
            for _, row in category_tickets.head(5).iterrows():  # Sample 5 tickets per category
                top_detail_data.append({
                    "Category": category,
                    "Ticket_ID": row['ticket_id'],
                    "Sample_Type": "Sample Ticket"
                })
        
        if top_detail_data:
            top_detail_df = pd.DataFrame(top_detail_data)
            top_detail_csv = self.storage_dir / f"{base_filename}_top_categories_samples.csv"
            top_detail_df.to_csv(top_detail_csv, index=False, encoding='utf-8-sig')
            csv_files.append(top_detail_csv)
        
        logger.info(f"Generated {len(csv_files)} CSV files")
        return csv_files

    def _prepare_summary_sheet(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare data for summary sheet."""
        metadata = analysis["metadata"]
        stats = analysis["statistical_summary"]
        
        return [
            {"Metric": "Analysis Date", "Value": metadata["analysis_date"]},
            {"Metric": "Total Tickets Analyzed", "Value": metadata["total_tickets"]},
            {"Metric": "Total Categories Found", "Value": stats["unique_categories"]},
            {"Metric": "Most Common Category", "Value": stats["most_common_category"]["name"]},
            {"Metric": "Most Common Category Count", "Value": stats["most_common_category"]["count"]},
            {"Metric": "Most Common Category %", "Value": f"{stats['most_common_category']['percentage']}%"},
            {"Metric": "Least Common Category", "Value": stats["least_common_category"]["name"]},
            {"Metric": "Least Common Category Count", "Value": stats["least_common_category"]["count"]},
            {"Metric": "Average Tickets per Category", "Value": stats["average_tickets_per_category"]},
            {"Metric": "Median Tickets per Category", "Value": stats["median_tickets_per_category"]},
        ]

    def _prepare_top_categories_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare detailed view of top categories."""
        top_10_categories = df['categoria'].value_counts().head(10)
        
        detail_data = []
        for rank, (category, count) in enumerate(top_10_categories.items(), 1):
            percentage = (count / len(df)) * 100
            detail_data.append({
                "Rank": rank,
                "Category": category,
                "Ticket_Count": count,
                "Percentage": round(percentage, 2),
                "Sample_Ticket_IDs": ", ".join(
                    df[df['categoria'] == category]['ticket_id'].head(3).astype(str).tolist()
                )
            })
        
        return pd.DataFrame(detail_data)

    def _prepare_statistics_sheet(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare statistics sheet data."""
        stats = analysis["statistical_summary"]["category_distribution_stats"]
        
        return [
            {"Statistic": "Mean tickets per category", "Value": stats["mean"]},
            {"Statistic": "Standard deviation", "Value": stats["std"]},
            {"Statistic": "Minimum tickets in category", "Value": stats["min"]},
            {"Statistic": "Maximum tickets in category", "Value": stats["max"]},
            {"Statistic": "Categories with >100 tickets", "Value": self._count_categories_above_threshold(analysis, 100)},
            {"Statistic": "Categories with >50 tickets", "Value": self._count_categories_above_threshold(analysis, 50)},
            {"Statistic": "Categories with >10 tickets", "Value": self._count_categories_above_threshold(analysis, 10)},
            {"Statistic": "Categories with only 1 ticket", "Value": self._count_categories_above_threshold(analysis, 1)},
        ]

    def _count_categories_above_threshold(self, analysis: Dict[str, Any], threshold: int) -> int:
        """Count categories above a certain threshold."""
        categories = analysis["category_distribution"]["categories"]
        if threshold == 1:
            return sum(1 for info in categories.values() if info["count"] == 1)
        return sum(1 for info in categories.values() if info["count"] >= threshold)

    def generate_simple_text_report(
        self, 
        analysis: Dict[str, Any], 
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate a simple text report for quick viewing.
        
        Args:
            analysis: Analysis results
            filename: Optional custom filename
            
        Returns:
            Path to generated text report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ticket_analysis_summary_{timestamp}.txt"
        
        report_file = self.storage_dir / filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TICKET ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            metadata = analysis["metadata"]
            f.write(f"Analysis Date: {metadata['analysis_date']}\n")
            f.write(f"Total Tickets: {metadata['total_tickets']:,}\n")
            f.write(f"Source File: {metadata['categorized_file']}\n\n")
            
            # Key Statistics
            stats = analysis["statistical_summary"]
            f.write("KEY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Categories: {stats['unique_categories']}\n")
            f.write(f"Most Common Category: {stats['most_common_category']['name']} ({stats['most_common_category']['count']:,} tickets, {stats['most_common_category']['percentage']}%)\n")
            f.write(f"Least Common Category: {stats['least_common_category']['name']} ({stats['least_common_category']['count']} tickets, {stats['least_common_category']['percentage']}%)\n")
            f.write(f"Average per Category: {stats['average_tickets_per_category']}\n\n")
            
            # Top 10 Categories
            f.write("TOP 10 CATEGORIES:\n")
            f.write("-" * 40 + "\n")
            top_categories = analysis["category_distribution"]["top_5_categories"]
            # Get all categories and sort by count
            all_categories = [(cat, info["count"], info["percentage"]) 
                             for cat, info in analysis["category_distribution"]["categories"].items()]
            all_categories.sort(key=lambda x: x[1], reverse=True)
            
            for i, (category, count, percentage) in enumerate(all_categories[:10], 1):
                f.write(f"{i:2d}. {category:<30} {count:>8,} tickets ({percentage:>5.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated by SimpleTicketAnalyzer\n")
        
        logger.info(f"Text report generated: {report_file}")
        return report_file

    def run_complete_analysis(
        self, 
        categorized_file: Path, 
        original_file: Optional[Path] = None,
        export_excel: bool = True,
        export_csv: bool = True,
        export_text: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete analysis and export all report formats.
        
        Args:
            categorized_file: Path to categorized tickets CSV
            original_file: Optional original dataset file
            export_excel: Whether to export Excel report
            export_csv: Whether to export CSV reports
            export_text: Whether to export text summary
            
        Returns:
            Summary of generated files and analysis
        """
        logger.info("Starting complete ticket analysis")
        
        try:
            # Load data
            categorized_df = pd.read_csv(categorized_file, sep=';', encoding='utf-8-sig')
            
            # Run analysis
            analysis = self.analyze_categorized_tickets(categorized_file, original_file)
            
            # Export reports
            generated_files = []
            
            if export_excel:
                excel_file = self.export_to_excel(analysis, categorized_df)
                generated_files.append(excel_file)
            
            if export_csv:
                csv_files = self.export_to_csv(analysis, categorized_df)
                generated_files.extend(csv_files)
            
            if export_text:
                text_file = self.generate_simple_text_report(analysis)
                generated_files.append(text_file)
            
            # Summary
            result_summary = {
                "analysis_completed": True,
                "total_tickets_analyzed": len(categorized_df),
                "total_categories_found": analysis["statistical_summary"]["unique_categories"],
                "files_generated": [str(f) for f in generated_files],
                "storage_directory": str(self.storage_dir),
                "analysis_data": analysis
            }
            
            logger.info(f"Complete analysis finished. Generated {len(generated_files)} files.")
            return result_summary
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            raise


def main():
    """Main function for running ticket analysis."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize analyzer
    analyzer = SimpleTicketAnalyzer()
    
    # Define file paths
    categorized_file = Path("database/categorized_tickets.csv")
    original_file = Path("database/tickets_conversas_19251_unique_v3.csv")
    
    # Check if files exist
    if not categorized_file.exists():
        print(f"Error: Categorized file not found: {categorized_file}")
        return
    
    print(f"Analyzing tickets from: {categorized_file}")
    if original_file.exists():
        print(f"Original file found: {original_file}")
    else:
        print("Original file not found - analysis will be limited to categorized data only")
        original_file = None
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            categorized_file=categorized_file,
            original_file=original_file,
            export_excel=True,
            export_csv=True,
            export_text=True
        )
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Total tickets analyzed: {results['total_tickets_analyzed']:,}")
        print(f"Total categories found: {results['total_categories_found']}")
        print(f"Files generated: {len(results['files_generated'])}")
        print(f"Storage directory: {results['storage_directory']}")
        print("\nGenerated files:")
        for file_path in results['files_generated']:
            print(f"  - {file_path}")
        print("="*80)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()