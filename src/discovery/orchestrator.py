"""
Two-Phase Orchestration System

This module implements the main orchestrator that coordinates the complete
Opção D pipeline: discovery → application with cost monitoring and validation.
"""

import logging
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Import base processor with fallback for testing
try:
    from ..base_processor import BaseProcessor
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from base_processor import BaseProcessor

# Import discovery components
try:
    from .intelligent_sampler import IntelligentSampler
    from .category_discoverer import CategoryDiscoverer
    from .fast_classifier import FastClassifier
except ImportError:
    # Fallback for testing
    import sys

    sys.path.append(str(Path(__file__).parent))
    from intelligent_sampler import IntelligentSampler
    from category_discoverer import CategoryDiscoverer
    from fast_classifier import FastClassifier


@dataclass
class OrchestrationMetrics:
    """Metrics for the complete orchestration process"""

    total_tickets: int
    discovery_sample_size: int
    categories_discovered: int
    total_processing_time: float
    discovery_time: float
    application_time: float
    total_cost_usd: float
    cost_per_1k_tickets: float
    avg_confidence: float
    classification_rate: float
    meets_cost_target: bool
    meets_confidence_target: bool


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration process"""

    # Discovery phase
    sample_rate: float = 0.15
    sampling_strategy: str = "hybrid"
    min_confidence: float = 0.85

    # Application phase
    batch_size: int = 100
    max_workers: int = 4

    # Cost monitoring
    cost_target_per_1k: float = 0.20
    cost_alert_threshold: float = 10.00

    # Quality validation
    confidence_threshold: float = 0.85
    max_categories_per_ticket: int = 3

    # File management
    categories_filename: str = "discovered_categories.json"
    results_filename: str = "final_categorized_tickets.csv"
    metrics_filename: str = "orchestration_metrics.json"


class OrchestrationError(Exception):
    """Custom exception for orchestration errors"""

    pass


class TwoPhaseOrchestrator(BaseProcessor):
    """
    Main orchestrator for the Two-Phase Discovery-Application pipeline.

    Coordinates the complete Opção D workflow:
    1. Discovery Phase: Sample → Discover Categories
    2. Application Phase: Classify All Tickets
    3. Validation and Monitoring
    """

    def __init__(
        self,
        api_key: str,
        database_dir: Optional[Path] = None,
        config: Optional[OrchestrationConfig] = None,
    ):
        """
        Initialize the Two-Phase Orchestrator.

        Args:
            api_key: Google API key for Gemini
            database_dir: Directory for data and outputs
            config: Orchestration configuration
        """
        # Set default database directory
        if database_dir is None:
            database_dir = Path.cwd() / "database"

        # Initialize parent
        super().__init__(
            api_key=api_key,
            database_dir=database_dir,
            max_tickets_per_batch=100,
            max_workers=4,
            use_cache=True,
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config or OrchestrationConfig()

        # Initialize components (lazy loading)
        self.sampler = None
        self.discoverer = None
        self.classifier = None

        # Metrics tracking
        self.start_time = None
        self.discovery_metrics = {}
        self.application_metrics = {}
        self.cost_tracker = {
            "discovery_cost": 0.0,
            "application_cost": 0.0,
            "total_tokens": 0,
        }

        self.logger.info("TwoPhaseOrchestrator initialized with Opção D configuration")

    def execute_complete_pipeline(
        self,
        input_file_path: Path,
        output_dir: Optional[Path] = None,
        force_rediscovery: bool = False,
        force_reclassification: bool = False,
    ) -> OrchestrationMetrics:
        """
        Execute the complete two-phase pipeline.

        Args:
            input_file_path: Path to input CSV file with tickets
            output_dir: Directory for outputs (defaults to database_dir)
            force_rediscovery: Force rediscovery even if categories exist
            force_reclassification: Force reclassification even if results exist

        Returns:
            OrchestrationMetrics with complete pipeline metrics
        """
        self.start_time = time.time()

        # Set output directory
        if output_dir is None:
            output_dir = self.database_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("=== STARTING TWO-PHASE ORCHESTRATION ===")
        self.logger.info(f"Input file: {input_file_path}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Load and validate input data
            tickets_df = self._load_and_validate_input(input_file_path)

            # Phase 1: Discovery
            categories_path = self._execute_discovery_phase(
                tickets_df, output_dir, force_rediscovery
            )

            # Phase 2: Application
            results_path = self._execute_application_phase(
                tickets_df, categories_path, output_dir, force_reclassification
            )

            # Generate final metrics and validation
            metrics = self._generate_final_metrics(
                tickets_df, categories_path, results_path, output_dir
            )

            # Save metrics
            self._save_orchestration_metrics(metrics, output_dir)

            self.logger.info("=== TWO-PHASE ORCHESTRATION COMPLETED ===")
            self._log_final_summary(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise OrchestrationError(f"Pipeline execution failed: {str(e)}") from e

    def _load_and_validate_input(self, input_file_path: Path) -> pd.DataFrame:
        """Load and validate input tickets data."""
        self.logger.info(f"Loading input data from {input_file_path}")

        try:
            # Load CSV data with appropriate encoding
            tickets_df = pd.read_csv(input_file_path, sep=";", encoding="utf-8-sig")

            # Validate required columns
            required_columns = ["ticket_id", "text", "sender", "category"]
            missing_columns = [col for col in required_columns if col not in tickets_df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Filter to TEXT category only
            text_tickets = tickets_df[tickets_df["category"].str.upper() == "TEXT"].copy()

            if text_tickets.empty:
                raise ValueError("No tickets found with category='TEXT'")

            self.logger.info(f"Loaded {len(text_tickets):,} valid TEXT tickets")
            self.logger.info(f"Unique tickets: {text_tickets['ticket_id'].nunique():,}")

            return text_tickets

        except Exception as e:
            raise OrchestrationError(f"Failed to load input data: {str(e)}") from e

    def _execute_discovery_phase(
        self, tickets_df: pd.DataFrame, output_dir: Path, force_rediscovery: bool
    ) -> Path:
        """Execute the discovery phase of the pipeline."""
        self.logger.info("=== PHASE 1: CATEGORY DISCOVERY ===")

        categories_path = output_dir / self.config.categories_filename

        # Check if categories already exist
        if categories_path.exists() and not force_rediscovery:
            self.logger.info(f"Categories already exist at {categories_path}, skipping discovery")
            self.discovery_metrics = {"skipped": True, "reason": "categories_exist"}
            return categories_path

        discovery_start = time.time()

        try:
            # Initialize sampler
            self.sampler = IntelligentSampler(
                strategy=self.config.sampling_strategy,
                random_state=42,  # Use fixed seed for reproducible results
            )

            # Step 1: Intelligent sampling
            self.logger.info(
                f"Sampling {self.config.sample_rate:.1%} of tickets using {self.config.sampling_strategy} strategy"
            )

            sample_tickets = self.sampler.sample_tickets(
                df=tickets_df,
                sample_size=self.config.sample_rate,
            )

            sample_size = len(sample_tickets)
            self.logger.info(f"Sampled {sample_size:,} tickets for discovery")

            # Step 2: Category discovery
            self.discoverer = CategoryDiscoverer(
                api_key=self.api_key, database_dir=self.database_dir
            )

            self.logger.info("Starting category pattern discovery...")
            discovered_categories = self.discoverer.discover_categories(
                tickets_df=sample_tickets, output_path=categories_path
            )

            # Track discovery metrics
            discovery_time = time.time() - discovery_start
            self.discovery_metrics = {
                "sample_size": sample_size,
                "sample_rate": self.config.sample_rate,
                "strategy": self.config.sampling_strategy,
                "categories_discovered": len(discovered_categories.get("categories", [])),
                "processing_time": discovery_time,
                "cost_estimate": self._estimate_discovery_cost(sample_size),
            }

            self.logger.info(f"Discovery completed in {discovery_time:.1f}s")
            self.logger.info(
                f"Discovered {len(discovered_categories.get('categories', []))} categories"
            )

            return categories_path

        except Exception as e:
            raise OrchestrationError(f"Discovery phase failed: {str(e)}") from e

    def _execute_application_phase(
        self,
        tickets_df: pd.DataFrame,
        categories_path: Path,
        output_dir: Path,
        force_reclassification: bool,
    ) -> Path:
        """Execute the application phase of the pipeline."""
        self.logger.info("=== PHASE 2: CATEGORY APPLICATION ===")

        results_path = output_dir / self.config.results_filename

        # Check if results already exist
        if results_path.exists() and not force_reclassification:
            self.logger.info(f"Results already exist at {results_path}, skipping classification")
            self.application_metrics = {"skipped": True, "reason": "results_exist"}
            return results_path

        application_start = time.time()

        try:
            # Initialize classifier
            self.classifier = FastClassifier(
                api_key=self.api_key,
                database_dir=self.database_dir,
                batch_size=self.config.batch_size,
                max_workers=self.config.max_workers,
            )

            # Load discovered categories
            self.logger.info(f"Loading categories from {categories_path}")
            self.classifier.load_categories(categories_path)

            # Classify all tickets
            self.logger.info(f"Classifying {len(tickets_df):,} tickets...")

            results_df = self.classifier.classify_all_tickets(
                tickets_df=tickets_df,
                output_path=results_path,
                force_reclassify=force_reclassification,
            )

            # Track application metrics
            application_time = time.time() - application_start
            classification_stats = self.classifier.get_classification_stats(results_df)

            self.application_metrics = {
                "total_tickets": len(tickets_df),
                "classified_tickets": len(results_df),
                "processing_time": application_time,
                "classification_stats": classification_stats,
                "cost_estimate": self._estimate_application_cost(len(tickets_df)),
            }

            self.logger.info(f"Application completed in {application_time:.1f}s")
            self.logger.info(f"Classified {len(results_df):,} tickets")

            return results_path

        except Exception as e:
            raise OrchestrationError(f"Application phase failed: {str(e)}") from e

    def _generate_final_metrics(
        self,
        tickets_df: pd.DataFrame,
        categories_path: Path,
        results_path: Path,
        output_dir: Path,
    ) -> OrchestrationMetrics:
        """Generate comprehensive final metrics for the orchestration."""
        total_time = time.time() - self.start_time

        # Load results for analysis
        results_df = pd.read_csv(results_path)

        # Load categories for count
        with open(categories_path, "r", encoding="utf-8") as f:
            categories_data = json.load(f)

        # Calculate key metrics
        total_tickets = len(tickets_df)
        discovery_sample_size = self.discovery_metrics.get("sample_size", 0)
        categories_discovered = len(categories_data.get("categories", []))

        # Cost calculations
        discovery_cost = self.discovery_metrics.get("cost_estimate", 0.0)
        application_cost = self.application_metrics.get("cost_estimate", 0.0)
        total_cost = discovery_cost + application_cost
        cost_per_1k = (total_cost / total_tickets) * 1000 if total_tickets > 0 else 0

        # Quality metrics
        confidences = results_df["confidence"].astype(float)
        avg_confidence = confidences.mean()
        classified_count = len(results_df[results_df["category_ids"] != ""])
        classification_rate = classified_count / total_tickets if total_tickets > 0 else 0

        # Target validation
        meets_cost_target = cost_per_1k <= self.config.cost_target_per_1k
        meets_confidence_target = avg_confidence >= self.config.confidence_threshold

        return OrchestrationMetrics(
            total_tickets=total_tickets,
            discovery_sample_size=discovery_sample_size,
            categories_discovered=categories_discovered,
            total_processing_time=total_time,
            discovery_time=self.discovery_metrics.get("processing_time", 0.0),
            application_time=self.application_metrics.get("processing_time", 0.0),
            total_cost_usd=total_cost,
            cost_per_1k_tickets=cost_per_1k,
            avg_confidence=avg_confidence,
            classification_rate=classification_rate,
            meets_cost_target=meets_cost_target,
            meets_confidence_target=meets_confidence_target,
        )

    def _save_orchestration_metrics(self, metrics: OrchestrationMetrics, output_dir: Path):
        """Save orchestration metrics to JSON file."""
        metrics_path = output_dir / self.config.metrics_filename

        metrics_dict = {
            "orchestration_summary": {
                "total_tickets": metrics.total_tickets,
                "discovery_sample_size": metrics.discovery_sample_size,
                "categories_discovered": metrics.categories_discovered,
                "total_processing_time": f"{metrics.total_processing_time:.1f}s",
                "cost_per_1k_tickets": f"${metrics.cost_per_1k_tickets:.4f}",
                "avg_confidence": f"{metrics.avg_confidence:.3f}",
                "classification_rate": f"{metrics.classification_rate:.1%}",
            },
            "phase_breakdown": {
                "discovery": {
                    "time": f"{metrics.discovery_time:.1f}s",
                    "sample_rate": f"{self.config.sample_rate:.1%}",
                    "strategy": self.config.sampling_strategy,
                    "categories_found": metrics.categories_discovered,
                },
                "application": {
                    "time": f"{metrics.application_time:.1f}s",
                    "batch_size": self.config.batch_size,
                    "workers": self.config.max_workers,
                    "classification_rate": f"{metrics.classification_rate:.1%}",
                },
            },
            "target_compliance": {
                "cost_target": f"${self.config.cost_target_per_1k:.2f}",
                "actual_cost": f"${metrics.cost_per_1k_tickets:.4f}",
                "meets_cost_target": metrics.meets_cost_target,
                "confidence_target": f"{self.config.confidence_threshold:.2f}",
                "actual_confidence": f"{metrics.avg_confidence:.3f}",
                "meets_confidence_target": metrics.meets_confidence_target,
            },
            "detailed_metrics": {
                "discovery_metrics": self.discovery_metrics,
                "application_metrics": self.application_metrics,
            },
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "sample_rate": self.config.sample_rate,
                "sampling_strategy": self.config.sampling_strategy,
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "cost_target_per_1k": self.config.cost_target_per_1k,
                "confidence_threshold": self.config.confidence_threshold,
            },
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Orchestration metrics saved to {metrics_path}")

    def _estimate_discovery_cost(self, sample_size: int) -> float:
        """Estimate cost for discovery phase."""
        # Rough estimation based on Opção D metrics
        tokens_per_ticket = 500  # Average tokens per ticket analysis
        total_tokens = sample_size * tokens_per_ticket
        # Gemini 2.5 Flash pricing approximation
        cost_per_1k_tokens = 0.000075
        return (total_tokens / 1000) * cost_per_1k_tokens

    def _estimate_application_cost(self, total_tickets: int) -> float:
        """Estimate cost for application phase."""
        # Rough estimation based on FastClassifier
        tokens_per_ticket = 150  # Input tokens
        output_tokens_per_ticket = 32  # Max output tokens
        total_input_tokens = total_tickets * tokens_per_ticket
        total_output_tokens = total_tickets * output_tokens_per_ticket

        # Gemini 2.5 Flash pricing
        input_cost_per_1k = 0.000075
        output_cost_per_1k = 0.000075

        input_cost = (total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (total_output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost

    def _log_final_summary(self, metrics: OrchestrationMetrics):
        """Log comprehensive final summary."""
        self.logger.info("=== FINAL ORCHESTRATION SUMMARY ===")
        self.logger.info(f"📊 Total tickets processed: {metrics.total_tickets:,}")
        self.logger.info(f"🎯 Categories discovered: {metrics.categories_discovered}")
        self.logger.info(f"⏱️ Total processing time: {metrics.total_processing_time:.1f}s")
        self.logger.info(f"💰 Total cost: ${metrics.total_cost_usd:.4f}")
        self.logger.info(f"💰 Cost per 1K tickets: ${metrics.cost_per_1k_tickets:.4f}")
        self.logger.info(f"📈 Average confidence: {metrics.avg_confidence:.3f}")
        self.logger.info(f"✅ Classification rate: {metrics.classification_rate:.1%}")

        self.logger.info("=== OPÇÃO D COMPLIANCE CHECK ===")
        cost_status = "✅" if metrics.meets_cost_target else "❌"
        confidence_status = "✅" if metrics.meets_confidence_target else "❌"

        self.logger.info(
            f"{cost_status} Cost target: ${metrics.cost_per_1k_tickets:.4f} "
            f"{'<=' if metrics.meets_cost_target else '>'} ${self.config.cost_target_per_1k:.2f}"
        )
        self.logger.info(
            f"{confidence_status} Confidence target: {metrics.avg_confidence:.3f} "
            f"{'>=' if metrics.meets_confidence_target else '<'} {self.config.confidence_threshold:.2f}"
        )

        if metrics.meets_cost_target and metrics.meets_confidence_target:
            self.logger.info("🎉 ALL OPÇÃO D TARGETS MET!")
        else:
            self.logger.warning("⚠️ Some Opção D targets not met - review configuration")


# Utility functions for easy integration
def run_complete_pipeline(
    input_file: str,
    output_dir: str,
    api_key: str,
    config: Optional[Dict[str, Any]] = None,
) -> OrchestrationMetrics:
    """
    Convenience function to run the complete pipeline.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory for outputs
        api_key: Google API key
        config: Configuration overrides

    Returns:
        OrchestrationMetrics with complete results
    """
    # Create configuration
    orchestration_config = OrchestrationConfig()
    if config:
        for key, value in config.items():
            if hasattr(orchestration_config, key):
                setattr(orchestration_config, key, value)

    # Initialize orchestrator
    orchestrator = TwoPhaseOrchestrator(
        api_key=api_key, database_dir=Path(output_dir), config=orchestration_config
    )

    # Execute pipeline
    return orchestrator.execute_complete_pipeline(
        input_file_path=Path(input_file), output_dir=Path(output_dir)
    )


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)

    input_file = "database/sample_tickets.csv"
    output_dir = "database/orchestration_output"

    metrics = run_complete_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        api_key=api_key,
        config={"sample_rate": 0.15, "batch_size": 50, "max_workers": 2},
    )

    print("Pipeline completed successfully!")
    print(f"Cost per 1K tickets: ${metrics.cost_per_1k_tickets:.4f}")
    print(f"Average confidence: {metrics.avg_confidence:.3f}")
    print(f"Processing time: {metrics.total_processing_time:.1f}s")
