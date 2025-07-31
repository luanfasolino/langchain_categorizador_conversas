"""
Ticket Report Generator - Integrated reporting system for LangChain categorization pipeline.

This module generates comprehensive Excel, CSV, and text reports from the real
categorization results produced by the TicketCategorizer and TicketSummarizer pipeline.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter
from base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class TicketReportGenerator(BaseProcessor):
    """
    Report generator that integrates with the existing LangChain categorization pipeline.

    This class generates comprehensive reports from real categorization results,
    working with categorized_tickets.csv and final_analysis.csv outputs.
    """

    def __init__(
        self,
        api_key: str = None,
        database_dir: Path = None,
        storage_dir: Optional[Path] = None,
        **kwargs,
    ):
        # Initialize BaseProcessor (even though we don't use LLM, maintains consistency)
        super().__init__(api_key or "dummy", database_dir, **kwargs)

        # Report-specific storage
        self.storage_dir = storage_dir or (self.database_dir / "analysis_reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TicketReportGenerator initialized")

    def analyze_categorization_results(
        self,
        categorized_file: Optional[Path] = None,
        final_analysis_file: Optional[Path] = None,
        summarized_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyze real categorization results from the pipeline.

        Args:
            categorized_file: Path to categorized_tickets.csv (optional, auto-detected)
            final_analysis_file: Path to final_analysis.csv (optional, auto-detected)
            summarized_file: Path to summarized_tickets.csv (optional, auto-detected)

        Returns:
            Comprehensive analysis results
        """
        logger.info("Analyzing real categorization results from pipeline")

        # Auto-detect files if not provided
        if not categorized_file:
            categorized_file = self.database_dir / "categorized_tickets.csv"
        if not final_analysis_file:
            final_analysis_file = self.database_dir / "final_analysis.csv"
        if not summarized_file:
            summarized_file = self.database_dir / "summarized_tickets.csv"

        # Verify files exist
        available_files = {
            "categorized": categorized_file if categorized_file.exists() else None,
            "final_analysis": (
                final_analysis_file if final_analysis_file.exists() else None
            ),
            "summarized": summarized_file if summarized_file.exists() else None,
        }

        if not available_files["categorized"]:
            raise FileNotFoundError(
                f"Categorized tickets file not found: {categorized_file}\n"
                "Please run categorization first: python main.py --mode categorize"
            )

        # Load data
        analysis_data = self._load_and_validate_data(available_files)

        # Generate comprehensive analysis
        analysis = {
            "metadata": self._generate_metadata(available_files, analysis_data),
            "category_analysis": self._analyze_categories(analysis_data),
            "statistical_summary": self._generate_statistical_summary(analysis_data),
            "quality_metrics": self._calculate_quality_metrics(analysis_data),
            "pipeline_performance": self._analyze_pipeline_performance(analysis_data),
        }

        # Add advanced analysis if final_analysis is available
        if available_files["final_analysis"]:
            analysis["advanced_insights"] = self._generate_advanced_insights(
                analysis_data
            )

        # Add summarization analysis if available
        if available_files["summarized"]:
            analysis["summarization_analysis"] = self._analyze_summarization_results(
                analysis_data
            )

        return analysis

    def _load_and_validate_data(
        self, available_files: Dict[str, Optional[Path]]
    ) -> Dict[str, pd.DataFrame]:
        """Load and validate all available data files."""
        data = {}

        # Load categorized data (required)
        categorized_df = pd.read_csv(
            available_files["categorized"], sep=";", encoding="utf-8-sig"
        )
        logger.info(f"Loaded categorized data: {len(categorized_df):,} tickets")
        data["categorized"] = categorized_df

        # Load final analysis if available
        if available_files["final_analysis"]:
            final_df = pd.read_csv(
                available_files["final_analysis"], sep=";", encoding="utf-8-sig"
            )
            logger.info(f"Loaded final analysis data: {len(final_df):,} records")
            data["final_analysis"] = final_df

        # Load summarized data if available
        if available_files["summarized"]:
            summarized_df = pd.read_csv(
                available_files["summarized"], sep=";", encoding="utf-8-sig"
            )
            logger.info(f"Loaded summarized data: {len(summarized_df):,} records")
            data["summarized"] = summarized_df

        # Validate data integrity
        self._validate_data_integrity(data)

        return data

    def _validate_data_integrity(self, data: Dict[str, pd.DataFrame]):
        """Validate data integrity across files."""
        categorized_df = data["categorized"]

        # Check required columns
        required_columns = ["ticket_id", "categoria"]
        missing_columns = [
            col for col in required_columns if col not in categorized_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in categorized data: {missing_columns}"
            )

        # Check for empty categories
        empty_categories = categorized_df["categoria"].isna().sum()
        if empty_categories > 0:
            logger.warning(f"Found {empty_categories} tickets with empty categories")

        # Cross-validate with other files
        if "final_analysis" in data:
            final_df = data["final_analysis"]

            # Check ticket_id overlap
            categorized_ids = set(categorized_df["ticket_id"])
            final_ids = (
                set(final_df["ticket_id"]) if "ticket_id" in final_df.columns else set()
            )

            if final_ids:
                overlap = len(categorized_ids & final_ids)
                logger.info(f"Ticket ID overlap between files: {overlap:,} tickets")

    def _generate_metadata(
        self, available_files: Dict[str, Optional[Path]], data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate analysis metadata."""
        categorized_df = data["categorized"]

        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "pipeline_version": "LangChain Categorization Pipeline v1.0",
            "files_analyzed": {
                "categorized_tickets": (
                    str(available_files["categorized"])
                    if available_files["categorized"]
                    else None
                ),
                "final_analysis": (
                    str(available_files["final_analysis"])
                    if available_files["final_analysis"]
                    else None
                ),
                "summarized_tickets": (
                    str(available_files["summarized"])
                    if available_files["summarized"]
                    else None
                ),
            },
            "data_summary": {
                "total_tickets": len(categorized_df),
                "unique_categories": categorized_df["categoria"].nunique(),
                "files_processed": len(
                    [f for f in available_files.values() if f is not None]
                ),
            },
        }

    def _analyze_categories(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze category distribution from real AI categorization."""
        categorized_df = data["categorized"]

        # Category distribution
        category_counts = categorized_df["categoria"].value_counts()
        total_tickets = len(categorized_df)

        # Detailed category analysis
        categories = {}
        for category, count in category_counts.items():
            percentage = (count / total_tickets) * 100
            categories[category] = {
                "count": int(count),
                "percentage": round(percentage, 2),
            }

        # Category quality metrics
        category_diversity = self._calculate_category_diversity(category_counts)

        return {
            "total_categories": len(category_counts),
            "category_distribution": categories,
            "top_categories": [
                {
                    "category": category,
                    "count": int(count),
                    "percentage": round((count / total_tickets) * 100, 2),
                }
                for category, count in category_counts.head(10).items()
            ],
            "category_quality_metrics": {
                "diversity_index": category_diversity,
                "concentration_ratio": self._calculate_concentration_ratio(
                    category_counts
                ),
                "dominant_category_share": round(
                    (category_counts.iloc[0] / total_tickets) * 100, 2
                ),
            },
        }

    def _calculate_category_diversity(self, category_counts: pd.Series) -> float:
        """Calculate Shannon diversity index for categories."""
        import numpy as np

        proportions = category_counts / category_counts.sum()
        shannon_entropy = -np.sum(
            proportions * np.log(proportions + 1e-10)
        )  # Add small epsilon to avoid log(0)
        max_entropy = np.log(len(category_counts))

        return round(shannon_entropy / max_entropy if max_entropy > 0 else 0, 3)

    def _calculate_concentration_ratio(self, category_counts: pd.Series) -> float:
        """Calculate concentration ratio (share of top 5 categories)."""
        top_5_sum = category_counts.head(5).sum()
        total_sum = category_counts.sum()
        return round((top_5_sum / total_sum) * 100, 2)

    def _generate_statistical_summary(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate statistical summary of categorization results."""
        categorized_df = data["categorized"]
        category_counts = categorized_df["categoria"].value_counts()

        return {
            "total_tickets": len(categorized_df),
            "unique_categories": len(category_counts),
            "category_statistics": {
                "mean_tickets_per_category": round(category_counts.mean(), 2),
                "median_tickets_per_category": int(category_counts.median()),
                "std_tickets_per_category": round(category_counts.std(), 2),
                "min_tickets_in_category": int(category_counts.min()),
                "max_tickets_in_category": int(category_counts.max()),
            },
            "distribution_analysis": {
                "categories_with_1_ticket": int((category_counts == 1).sum()),
                "categories_with_2_5_tickets": int(
                    ((category_counts >= 2) & (category_counts <= 5)).sum()
                ),
                "categories_with_6_20_tickets": int(
                    ((category_counts >= 6) & (category_counts <= 20)).sum()
                ),
                "categories_with_21_100_tickets": int(
                    ((category_counts >= 21) & (category_counts <= 100)).sum()
                ),
                "categories_with_100plus_tickets": int((category_counts > 100).sum()),
            },
        }

    def _calculate_quality_metrics(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the categorization."""
        categorized_df = data["categorized"]

        # Data completeness
        total_tickets = len(categorized_df)
        valid_categories = categorized_df["categoria"].notna().sum()
        empty_categories = total_tickets - valid_categories

        # Category name quality
        category_names = categorized_df["categoria"].dropna().unique()
        avg_category_name_length = np.mean([len(str(cat)) for cat in category_names])

        return {
            "data_completeness": {
                "total_tickets": total_tickets,
                "valid_categorizations": int(valid_categories),
                "empty_categorizations": int(empty_categories),
                "completeness_rate": round((valid_categories / total_tickets) * 100, 2),
            },
            "category_naming_quality": {
                "average_category_name_length": round(avg_category_name_length, 1),
                "shortest_category_name": (
                    min([len(str(cat)) for cat in category_names])
                    if len(category_names) > 0
                    else 0
                ),
                "longest_category_name": (
                    max([len(str(cat)) for cat in category_names])
                    if len(category_names) > 0
                    else 0
                ),
            },
        }

    def _analyze_pipeline_performance(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze pipeline performance metrics."""
        categorized_df = data["categorized"]

        # Use BaseProcessor's cache management for consistency
        cache_info = self._get_cache_performance_info()

        return {
            "processing_efficiency": {
                "total_tickets_processed": len(categorized_df),
                "processing_timestamp": datetime.now().isoformat(),
                "pipeline_components_used": self._identify_pipeline_components_used(
                    data
                ),
            },
            "cache_performance": cache_info,
            "data_flow_analysis": self._analyze_data_flow(data),
        }

    def _get_cache_performance_info(self) -> Dict[str, Any]:
        """Get cache performance information from BaseProcessor."""
        try:
            cache_dir = self.database_dir / "cache"
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.pkl"))
                return {
                    "cache_enabled": True,
                    "cache_files_count": len(cache_files),
                    "cache_directory": str(cache_dir),
                }
            else:
                return {"cache_enabled": False}
        except Exception as e:
            logger.warning(f"Could not analyze cache performance: {str(e)}")
            return {"cache_enabled": "unknown", "error": str(e)}

    def _identify_pipeline_components_used(
        self, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Identify which pipeline components were used based on available data."""
        components = [
            "TicketCategorizer"
        ]  # Always present since categorized data exists

        if "summarized" in data:
            components.append("TicketSummarizer")

        if "final_analysis" in data:
            components.append("TicketDataMerger")

        return components

    def _analyze_data_flow(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze data flow through the pipeline."""
        flow_analysis = {
            "categorization_stage": {
                "input_tickets": "Auto-detected from processed data",
                "output_tickets": len(data["categorized"]),
                "data_retention_rate": 100.0,  # We only see successful outputs
            }
        }

        if "final_analysis" in data:
            final_df = data["final_analysis"]
            categorized_df = data["categorized"]

            if "ticket_id" in final_df.columns:
                merge_rate = len(final_df) / len(categorized_df) * 100
                flow_analysis["merge_stage"] = {
                    "categorized_input": len(categorized_df),
                    "final_output": len(final_df),
                    "merge_success_rate": round(merge_rate, 2),
                }

        return flow_analysis

    def _generate_advanced_insights(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate advanced insights from final analysis data."""
        final_df = data["final_analysis"]

        insights = {
            "data_richness": {
                "total_merged_records": len(final_df),
                "available_columns": list(final_df.columns),
                "data_coverage": self._calculate_data_coverage(final_df),
            }
        }

        # Analyze bullet points if available
        if "bullets" in final_df.columns:
            insights["summarization_coverage"] = self._analyze_bullet_coverage(final_df)

        return insights

    def _calculate_data_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data coverage for each column."""
        coverage = {}
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            coverage[column] = round((non_null_count / len(df)) * 100, 2)
        return coverage

    def _analyze_bullet_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze bullet point summarization coverage."""
        bullets_series = df["bullets"].dropna()

        if len(bullets_series) == 0:
            return {"summary_available": False}

        # Parse bullet points (assuming JSON format)
        bullet_analyses = []
        for bullets_json in bullets_series:
            try:
                bullets_data = (
                    json.loads(bullets_json)
                    if isinstance(bullets_json, str)
                    else bullets_json
                )
                if isinstance(bullets_data, list):
                    bullet_analyses.append(len(bullets_data))
            except (json.JSONDecodeError, TypeError):
                continue

        if bullet_analyses:
            return {
                "summary_available": True,
                "tickets_with_summaries": len(bullet_analyses),
                "average_bullets_per_ticket": round(np.mean(bullet_analyses), 1),
                "min_bullets": min(bullet_analyses),
                "max_bullets": max(bullet_analyses),
            }

        return {"summary_available": False, "parse_errors": len(bullets_series)}

    def _analyze_summarization_results(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze summarization results if available."""
        summarized_df = data["summarized"]

        return {
            "summarization_coverage": {
                "total_summarized_tickets": len(summarized_df),
                "available_columns": list(summarized_df.columns),
            },
            "summary_quality_indicators": self._assess_summary_quality(summarized_df),
        }

    def _assess_summary_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of summarization results."""
        quality_metrics = {}

        if "bullets" in df.columns:
            bullets_series = df["bullets"].dropna()
            quality_metrics["bullet_completion_rate"] = round(
                (len(bullets_series) / len(df)) * 100, 2
            )

        return quality_metrics

    def export_comprehensive_reports(
        self,
        analysis: Dict[str, Any],
        categorized_df: pd.DataFrame,
        filename_base: Optional[str] = None,
        export_excel: bool = True,
        export_csv: bool = True,
        export_text: bool = True,
    ) -> Dict[str, List[Path]]:
        """
        Export comprehensive reports in multiple formats.

        Args:
            analysis: Analysis results from analyze_categorization_results
            categorized_df: Categorized tickets DataFrame
            filename_base: Base filename for exports
            export_excel: Export Excel report
            export_csv: Export CSV files
            export_text: Export text summary

        Returns:
            Dictionary of exported file paths by format
        """
        if not filename_base:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"pipeline_analysis_{timestamp}"

        exported_files = {"excel": [], "csv": [], "text": []}

        logger.info(f"Exporting comprehensive reports with base name: {filename_base}")

        try:
            if export_excel:
                excel_file = self._export_excel_report(
                    analysis, categorized_df, filename_base
                )
                exported_files["excel"].append(excel_file)

            if export_csv:
                csv_files = self._export_csv_reports(
                    analysis, categorized_df, filename_base
                )
                exported_files["csv"].extend(csv_files)

            if export_text:
                text_file = self._export_text_report(analysis, filename_base)
                exported_files["text"].append(text_file)

            logger.info(
                f"Successfully exported {sum(len(files) for files in exported_files.values())} report files"
            )
            return exported_files

        except Exception as e:
            logger.error(f"Error exporting reports: {str(e)}")
            raise

    def _export_excel_report(
        self, analysis: Dict[str, Any], categorized_df: pd.DataFrame, filename_base: str
    ) -> Path:
        """Export comprehensive Excel report."""
        excel_file = self.storage_dir / f"{filename_base}.xlsx"

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = self._prepare_summary_data(analysis)
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Executive_Summary", index=False
            )

            # Category distribution
            category_data = []
            for category, info in analysis["category_analysis"][
                "category_distribution"
            ].items():
                category_data.append(
                    {
                        "Category": category,
                        "Count": info["count"],
                        "Percentage": info["percentage"],
                    }
                )

            category_df = pd.DataFrame(category_data).sort_values(
                "Count", ascending=False
            )
            category_df.to_excel(
                writer, sheet_name="Category_Distribution", index=False
            )

            # All categorized tickets
            categorized_df.to_excel(
                writer, sheet_name="All_Categorized_Tickets", index=False
            )

            # Quality metrics
            quality_data = self._prepare_quality_metrics_data(analysis)
            pd.DataFrame(quality_data).to_excel(
                writer, sheet_name="Quality_Metrics", index=False
            )

            # Pipeline performance
            performance_data = self._prepare_performance_data(analysis)
            pd.DataFrame(performance_data).to_excel(
                writer, sheet_name="Pipeline_Performance", index=False
            )

        logger.info(f"Excel report exported: {excel_file}")
        return excel_file

    def _export_csv_reports(
        self, analysis: Dict[str, Any], categorized_df: pd.DataFrame, filename_base: str
    ) -> List[Path]:
        """Export multiple CSV reports."""
        csv_files = []

        # Category distribution CSV
        category_data = []
        for category, info in analysis["category_analysis"][
            "category_distribution"
        ].items():
            category_data.append(
                {
                    "Category": category,
                    "Count": info["count"],
                    "Percentage": info["percentage"],
                }
            )

        category_df = pd.DataFrame(category_data).sort_values("Count", ascending=False)
        category_csv = self.storage_dir / f"{filename_base}_categories.csv"
        category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")
        csv_files.append(category_csv)

        # Summary statistics CSV
        summary_data = self._prepare_summary_data(analysis)
        summary_csv = self.storage_dir / f"{filename_base}_summary.csv"
        pd.DataFrame(summary_data).to_csv(
            summary_csv, index=False, encoding="utf-8-sig"
        )
        csv_files.append(summary_csv)

        # All tickets CSV
        tickets_csv = self.storage_dir / f"{filename_base}_all_tickets.csv"
        categorized_df.to_csv(tickets_csv, index=False, encoding="utf-8-sig")
        csv_files.append(tickets_csv)

        logger.info(f"Exported {len(csv_files)} CSV files")
        return csv_files

    def _export_text_report(self, analysis: Dict[str, Any], filename_base: str) -> Path:
        """Export text summary report."""
        text_file = self.storage_dir / f"{filename_base}_summary.txt"

        with open(text_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("LANGCHAIN CATEGORIZATION PIPELINE - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            metadata = analysis["metadata"]
            f.write(f"Analysis Date: {metadata['analysis_timestamp']}\n")
            f.write(f"Pipeline Version: {metadata['pipeline_version']}\n")
            f.write(
                f"Total Tickets Analyzed: {metadata['data_summary']['total_tickets']:,}\n"
            )
            f.write(
                f"Unique Categories Found: {metadata['data_summary']['unique_categories']}\n\n"
            )

            # Key statistics
            stats = analysis["statistical_summary"]
            f.write("KEY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Categories: {stats['unique_categories']}\n")
            f.write(
                f"Mean Tickets per Category: {stats['category_statistics']['mean_tickets_per_category']}\n"
            )
            f.write(
                f"Median Tickets per Category: {stats['category_statistics']['median_tickets_per_category']}\n\n"
            )

            # Top categories
            f.write("TOP 10 CATEGORIES:\n")
            f.write("-" * 50 + "\n")
            for i, category_info in enumerate(
                analysis["category_analysis"]["top_categories"], 1
            ):
                f.write(
                    f"{i:2d}. {category_info['category']:<35} {category_info['count']:>8,} ({category_info['percentage']:>5.1f}%)\n"
                )

            # Quality metrics
            quality = analysis["quality_metrics"]
            f.write(f"\nQUALITY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Data Completeness: {quality['data_completeness']['completeness_rate']}%\n"
            )
            f.write(
                f"Valid Categorizations: {quality['data_completeness']['valid_categorizations']:,}\n"
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated by TicketReportGenerator\n")
            f.write("Integrated with LangChain Categorization Pipeline\n")

        logger.info(f"Text report exported: {text_file}")
        return text_file

    def _prepare_summary_data(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare summary data for export."""
        metadata = analysis["metadata"]
        stats = analysis["statistical_summary"]
        quality = analysis["quality_metrics"]

        return [
            {"Metric": "Analysis Date", "Value": metadata["analysis_timestamp"]},
            {
                "Metric": "Total Tickets",
                "Value": metadata["data_summary"]["total_tickets"],
            },
            {"Metric": "Unique Categories", "Value": stats["unique_categories"]},
            {
                "Metric": "Mean Tickets per Category",
                "Value": stats["category_statistics"]["mean_tickets_per_category"],
            },
            {
                "Metric": "Data Completeness Rate",
                "Value": f"{quality['data_completeness']['completeness_rate']}%",
            },
            {
                "Metric": "Pipeline Components Used",
                "Value": len(
                    analysis["pipeline_performance"]["processing_efficiency"][
                        "pipeline_components_used"
                    ]
                ),
            },
        ]

    def _prepare_quality_metrics_data(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare quality metrics data for export."""
        quality = analysis["quality_metrics"]

        return [
            {
                "Metric": "Total Tickets",
                "Value": quality["data_completeness"]["total_tickets"],
            },
            {
                "Metric": "Valid Categorizations",
                "Value": quality["data_completeness"]["valid_categorizations"],
            },
            {
                "Metric": "Empty Categorizations",
                "Value": quality["data_completeness"]["empty_categorizations"],
            },
            {
                "Metric": "Completeness Rate",
                "Value": f"{quality['data_completeness']['completeness_rate']}%",
            },
            {
                "Metric": "Avg Category Name Length",
                "Value": quality["category_naming_quality"][
                    "average_category_name_length"
                ],
            },
        ]

    def _prepare_performance_data(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare performance data for export."""
        performance = analysis["pipeline_performance"]

        return [
            {
                "Metric": "Total Tickets Processed",
                "Value": performance["processing_efficiency"][
                    "total_tickets_processed"
                ],
            },
            {
                "Metric": "Pipeline Components",
                "Value": ", ".join(
                    performance["processing_efficiency"]["pipeline_components_used"]
                ),
            },
            {
                "Metric": "Cache Enabled",
                "Value": performance["cache_performance"].get(
                    "cache_enabled", "Unknown"
                ),
            },
        ]

    def process_pipeline_results(
        self,
        categorized_file: Optional[Path] = None,
        export_reports: bool = True,
        filename_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main method to process pipeline results and generate reports.

        Args:
            categorized_file: Path to categorized tickets file
            export_reports: Whether to export reports
            filename_base: Base filename for exports

        Returns:
            Complete analysis results and export information
        """
        logger.info("Processing LangChain categorization pipeline results")

        try:
            # Analyze results
            analysis = self.analyze_categorization_results(categorized_file)

            # Load categorized data for export
            if not categorized_file:
                categorized_file = self.database_dir / "categorized_tickets.csv"

            categorized_df = pd.read_csv(
                categorized_file, sep=";", encoding="utf-8-sig"
            )

            result = {
                "analysis_completed": True,
                "analysis_data": analysis,
                "total_tickets": len(categorized_df),
                "total_categories": analysis["statistical_summary"][
                    "unique_categories"
                ],
                "export_info": {"files_exported": []},
            }

            # Export reports if requested
            if export_reports:
                exported_files = self.export_comprehensive_reports(
                    analysis, categorized_df, filename_base
                )

                result["export_info"] = {
                    "files_exported": exported_files,
                    "total_files": sum(len(files) for files in exported_files.values()),
                    "storage_directory": str(self.storage_dir),
                }

            logger.info(
                f"Pipeline analysis completed successfully. Processed {result['total_tickets']:,} tickets."
            )
            return result

        except Exception as e:
            logger.error(f"Error processing pipeline results: {str(e)}")
            raise


# Import numpy for statistical calculations
import numpy as np
