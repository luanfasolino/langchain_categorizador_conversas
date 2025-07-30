"""
Fast Classifier for Application Phase

This module implements the optimized classifier for the application phase
that processes all tickets using discovered categories with cost-efficient prompts.
Based on Opção D Map-Reduce architecture.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Import base processor with fallback for testing
try:
    from ..base_processor import BaseProcessor
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from base_processor import BaseProcessor


@dataclass
class ClassificationResult:
    """Classification result for a single ticket"""

    ticket_id: str
    categories: List[int]
    confidence: float
    processing_time: float
    tokens_used: int


@dataclass
class BatchResult:
    """Result for a batch of classifications"""

    batch_id: int
    results: List[ClassificationResult]
    total_tokens: int
    total_time: float
    success_rate: float


class FastClassifier(BaseProcessor):
    """
    Fast Classifier for applying discovered categories to all tickets.

    Based on Opção D Map-Reduce architecture with optimized prompts
    for cost-effective classification of large ticket volumes.
    """

    def __init__(
        self,
        api_key: str,
        database_dir: Optional[Path] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_workers: int = 4,
        batch_size: int = 100,
    ):
        """
        Initialize the Fast Classifier.

        Args:
            api_key: Google API key for Gemini
            database_dir: Directory for data and cache
            model_name: LLM model to use
            temperature: LLM temperature (lower = more consistent)
            max_workers: Maximum concurrent workers
            batch_size: Number of tickets per batch
        """
        # Set default database directory
        if database_dir is None:
            database_dir = Path.cwd() / "database"

        # Initialize parent with correct signature
        super().__init__(
            api_key=api_key,
            database_dir=database_dir,
            max_tickets_per_batch=batch_size,
            max_workers=max_workers,
            use_cache=True,
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Override LLM with custom settings following Opção D
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, google_api_key=api_key
        )

        # Classification-specific configurations following Opção D specs
        self.batch_size = batch_size
        self.max_categories_per_ticket = 3  # From Opção D config
        self.confidence_threshold = 0.85  # From Opção D config

        # Cost optimization settings
        self.classify_max_tokens = 32  # From Opção D config
        self.target_cost_per_1k = 0.20  # From Opção D metrics

        # Initialize categories and chains
        self.categories = None
        self.classification_chain = None

        self.logger.info(
            f"FastClassifier initialized with {model_name}, "
            f"batch_size={batch_size}, max_workers={max_workers}"
        )

    def load_categories(self, categories_path: Path) -> Dict[str, Any]:
        """
        Load discovered categories from JSON file.

        Args:
            categories_path: Path to categories.json file

        Returns:
            Categories dictionary
        """
        try:
            with open(categories_path, "r", encoding="utf-8") as f:
                self.categories = json.load(f)

            # Setup classification chain with loaded categories
            self._setup_classification_chain()

            self.logger.info(
                f"Loaded {len(self.categories.get('categories', []))} "
                f"categories from {categories_path}"
            )
            return self.categories

        except Exception as e:
            self.logger.error(
                f"Failed to load categories from {categories_path}: {str(e)}"
            )
            raise

    def _setup_classification_chain(self):
        """Setup the optimized classification chain following Opção D specs."""

        if not self.categories:
            raise ValueError(
                "Categories must be loaded before setting up classification chain"
            )

        # Build category list for prompt (optimized format)
        category_list = []
        for cat in self.categories.get("categories", []):
            cat_text = f"{cat['id']}. {cat['display_name']}: {cat['description']}"
            # Add key examples for better accuracy
            if cat.get("examples"):
                examples = ", ".join(
                    cat["examples"][:2]
                )  # Max 2 examples for token efficiency
                cat_text += f" (ex: {examples})"
            category_list.append(cat_text)

        categories_text = "\n".join(category_list)

        # Optimized classification template following Opção D principles
        classification_template = f"""
Você é um classificador de tickets de suporte altamente eficiente.

CATEGORIAS DISPONÍVEIS:
{categories_text}

REGRAS DE CLASSIFICAÇÃO:
1. Analise cada ticket e retorne APENAS os IDs das categorias (números)
2. Máximo {self.max_categories_per_ticket} categorias por ticket
3. Use a categoria mais específica quando houver sobreposição
4. Se incerto, escolha a categoria mais provável (>85% confiança)
5. Responda em formato JSON: {{"categories": [id1, id2], "confidence": 0.XX}}

TICKET:
{{ticket_text}}

CLASSIFICAÇÃO:"""

        # Create optimized chain using LCEL (Opção D style)
        self.classification_prompt = ChatPromptTemplate.from_template(
            classification_template
        )
        self.classification_chain = (
            self.classification_prompt | self.llm | StrOutputParser()
        )

        self.logger.info("Classification chain setup completed with optimized prompts")

    def classify_all_tickets(
        self,
        tickets_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        force_reclassify: bool = False,
    ) -> pd.DataFrame:
        """
        Classify all tickets using the discovered categories.

        Args:
            tickets_df: DataFrame with all tickets to classify
            output_path: Path to save classification results
            force_reclassify: Skip cache and force new classification

        Returns:
            DataFrame with classification results
        """
        if self.categories is None:
            raise ValueError("Categories must be loaded before classification")

        if tickets_df.empty:
            raise ValueError("Cannot classify empty dataset")

        # Generate cache key
        tickets_content = tickets_df.to_string()
        cache_key = self._generate_cache_key(tickets_content + "fast_classification")

        # Check cache first
        if not force_reclassify:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info("Loading classification results from cache")
                if output_path:
                    cached_result.to_csv(output_path, index=False)
                return cached_result

        self.logger.info(f"Starting classification of {len(tickets_df)} tickets")

        try:
            # Prepare tickets for classification
            prepared_tickets = self._prepare_tickets_for_classification(tickets_df)

            # Process in optimized batches following Opção D architecture
            batch_results = self._process_classification_batches(prepared_tickets)

            # Consolidate results
            classification_results = self._consolidate_classification_results(
                batch_results
            )

            # Create results DataFrame
            results_df = self._create_results_dataframe(
                classification_results, tickets_df
            )

            # Cache results
            self._save_to_cache(cache_key, results_df)

            # Save to file if requested
            if output_path:
                results_df.to_csv(output_path, index=False)
                self.logger.info(f"Classification results saved to {output_path}")

            # Log performance metrics
            self._log_classification_metrics(batch_results, results_df)

            return results_df

        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            raise

    def _prepare_tickets_for_classification(
        self, tickets_df: pd.DataFrame
    ) -> List[Dict[str, str]]:
        """
        Prepare tickets for classification processing.

        Args:
            tickets_df: Input tickets DataFrame

        Returns:
            List of prepared ticket dictionaries
        """
        prepared_tickets = []

        for ticket_id, group in tickets_df.groupby("ticket_id"):
            # Sort messages chronologically
            messages = (
                group.sort_values("message_sended_at")
                if "message_sended_at" in group.columns
                else group
            )

            # Combine all messages for the ticket (following Opção D approach)
            conversation_parts = []
            for _, msg in messages.iterrows():
                sender = msg.get("sender", "UNKNOWN")
                text = msg.get("text", "").strip()
                if text:  # Only include non-empty messages
                    conversation_parts.append(f"[{sender}]: {text}")

            if conversation_parts:  # Only include tickets with content
                ticket_text = "\n".join(conversation_parts)
                prepared_tickets.append(
                    {"ticket_id": str(ticket_id), "text": ticket_text}
                )

        self.logger.info(f"Prepared {len(prepared_tickets)} tickets for classification")
        return prepared_tickets

    def _process_classification_batches(
        self, prepared_tickets: List[Dict[str, str]]
    ) -> List[BatchResult]:
        """
        Process tickets in optimized batches using parallel execution.

        Args:
            prepared_tickets: List of prepared ticket dictionaries

        Returns:
            List of batch results
        """
        # Create batches
        batches = [
            prepared_tickets[i : i + self.batch_size]
            for i in range(0, len(prepared_tickets), self.batch_size)
        ]

        self.logger.info(
            f"Processing {len(batches)} batches with {self.max_workers} workers"
        )

        batch_results = []

        # Process batches in parallel following Opção D pattern
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    self._process_classification_batch, batch, batch_idx
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            # Collect results with progress tracking
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()
                    batch_results.append(batch_result)

                    # Log progress
                    processed = len(batch_results)
                    total = len(batches)
                    self.logger.info(
                        f"Completed batch {batch_idx + 1}/{total} "
                        f"({processed / total:.1%} done)"
                    )

                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    # Continue with other batches
                    continue

        if not batch_results:
            raise RuntimeError("No successful batch classifications completed")

        self.logger.info(f"Completed {len(batch_results)} batches successfully")
        return batch_results

    def _process_classification_batch(
        self, batch: List[Dict[str, str]], batch_idx: int
    ) -> BatchResult:
        """
        Process a single batch of tickets for classification.

        Args:
            batch: List of tickets in the batch
            batch_idx: Batch index for logging

        Returns:
            BatchResult with classification outcomes
        """
        batch_start_time = time.time()
        results = []
        total_tokens = 0
        successful_classifications = 0

        for ticket in batch:
            try:
                start_time = time.time()

                # Perform classification
                classification_response = self.classification_chain.invoke(
                    {"ticket_text": ticket["text"]}
                )

                # Parse classification result
                classification_data = self._parse_classification_response(
                    classification_response
                )

                # Estimate tokens (following Opção D token estimation)
                tokens_used = (
                    self.estimate_tokens(ticket["text"]) + self.classify_max_tokens
                )
                total_tokens += tokens_used

                # Create result
                result = ClassificationResult(
                    ticket_id=ticket["ticket_id"],
                    categories=classification_data.get("categories", []),
                    confidence=classification_data.get("confidence", 0.0),
                    processing_time=time.time() - start_time,
                    tokens_used=tokens_used,
                )

                results.append(result)
                successful_classifications += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to classify ticket {ticket['ticket_id']} in batch {batch_idx}: {str(e)}"
                )
                # Add failed result
                results.append(
                    ClassificationResult(
                        ticket_id=ticket["ticket_id"],
                        categories=[],
                        confidence=0.0,
                        processing_time=0.0,
                        tokens_used=0,
                    )
                )

        batch_time = time.time() - batch_start_time
        success_rate = successful_classifications / len(batch) if batch else 0.0

        return BatchResult(
            batch_id=batch_idx,
            results=results,
            total_tokens=total_tokens,
            total_time=batch_time,
            success_rate=success_rate,
        )

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM classification response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed classification data
        """
        try:
            # Try to parse as JSON first
            if "{" in response and "}" in response:
                start_idx = response.find("{")
                end_idx = response.rfind("}")
                json_str = response[start_idx : end_idx + 1]
                return json.loads(json_str)

            # Fallback: extract category numbers
            import re

            category_numbers = re.findall(r"\b(\d+)\b", response)
            categories = [
                int(num) for num in category_numbers[: self.max_categories_per_ticket]
            ]

            return {
                "categories": categories,
                "confidence": 0.7,  # Default confidence for fallback parsing
            }

        except Exception as e:
            self.logger.error(f"Failed to parse classification response: {str(e)}")
            return {"categories": [], "confidence": 0.0}

    def _consolidate_classification_results(
        self, batch_results: List[BatchResult]
    ) -> List[ClassificationResult]:
        """
        Consolidate results from all batches.

        Args:
            batch_results: List of batch results

        Returns:
            Consolidated list of classification results
        """
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result.results)

        self.logger.info(f"Consolidated {len(all_results)} classification results")
        return all_results

    def _create_results_dataframe(
        self, results: List[ClassificationResult], original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create results DataFrame with classification outcomes.

        Args:
            results: List of classification results
            original_df: Original tickets DataFrame

        Returns:
            Results DataFrame
        """
        # Create results records
        result_records = []

        for result in results:
            # Get category names
            category_names = []
            for cat_id in result.categories:
                cat_name = self._get_category_name(cat_id)
                if cat_name:
                    category_names.append(cat_name)

            result_records.append(
                {
                    "ticket_id": result.ticket_id,
                    "category_ids": (
                        ",".join(map(str, result.categories))
                        if result.categories
                        else ""
                    ),
                    "category_names": (
                        ",".join(category_names) if category_names else ""
                    ),
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "tokens_used": result.tokens_used,
                }
            )

        results_df = pd.DataFrame(result_records)

        # Add metadata
        results_df["classification_timestamp"] = datetime.now().isoformat()
        results_df["model_used"] = self.model_name

        self.logger.info(
            f"Created results DataFrame with {len(results_df)} classified tickets"
        )
        return results_df

    def _get_category_name(self, category_id: int) -> Optional[str]:
        """
        Get category name by ID.

        Args:
            category_id: Category ID

        Returns:
            Category display name or None
        """
        if not self.categories:
            return None

        for cat in self.categories.get("categories", []):
            if cat.get("id") == category_id:
                return cat.get("display_name", f"Category_{category_id}")

        return None

    def _log_classification_metrics(
        self, batch_results: List[BatchResult], results_df: pd.DataFrame
    ):
        """
        Log comprehensive classification metrics following Opção D monitoring.

        Args:
            batch_results: List of batch results
            results_df: Final results DataFrame
        """
        # Calculate metrics
        total_tickets = len(results_df)
        total_tokens = sum(batch.total_tokens for batch in batch_results)
        total_time = sum(batch.total_time for batch in batch_results)
        avg_success_rate = np.mean([batch.success_rate for batch in batch_results])

        # Cost estimation (following Opção D pricing)
        estimated_cost = self._estimate_cost(total_tokens)
        cost_per_1k = (
            (estimated_cost / total_tickets) * 1000 if total_tickets > 0 else 0
        )

        # Confidence statistics
        confidences = results_df["confidence"].astype(float)
        avg_confidence = confidences.mean()
        high_confidence_ratio = (confidences >= self.confidence_threshold).mean()

        # Log metrics
        self.logger.info("=== CLASSIFICATION METRICS ===")
        self.logger.info(f"Total tickets processed: {total_tickets:,}")
        self.logger.info(f"Total tokens used: {total_tokens:,}")
        self.logger.info(f"Total processing time: {total_time:.1f}s")
        self.logger.info(f"Average success rate: {avg_success_rate:.1%}")
        self.logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        self.logger.info(f"Cost per 1K tickets: ${cost_per_1k:.4f}")
        self.logger.info(f"Average confidence: {avg_confidence:.3f}")
        self.logger.info(f"High confidence ratio: {high_confidence_ratio:.1%}")

        # Check Opção D targets
        if cost_per_1k <= self.target_cost_per_1k:
            self.logger.info(
                f"✅ Cost target met: ${cost_per_1k:.4f} <= ${self.target_cost_per_1k}"
            )
        else:
            self.logger.warning(
                f"❌ Cost target exceeded: ${cost_per_1k:.4f} > ${self.target_cost_per_1k}"
            )

    def _estimate_cost(self, total_tokens: int) -> float:
        """
        Estimate cost based on Gemini pricing.

        Args:
            total_tokens: Total tokens used

        Returns:
            Estimated cost in USD
        """
        # Gemini 2.5 Flash pricing (approximate)
        cost_per_1k_tokens = 0.000075  # $0.075 per 1M tokens
        return (total_tokens / 1000) * cost_per_1k_tokens

    def get_classification_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive classification statistics.

        Args:
            results_df: Classification results DataFrame

        Returns:
            Statistics dictionary
        """
        if results_df.empty:
            return {}

        # Basic stats
        total_tickets = len(results_df)
        classified_tickets = len(results_df[results_df["category_ids"] != ""])
        classification_rate = (
            classified_tickets / total_tickets if total_tickets > 0 else 0
        )

        # Category distribution
        all_categories = []
        for cat_ids in results_df["category_ids"]:
            if cat_ids:
                all_categories.extend([int(x) for x in cat_ids.split(",")])

        category_counts = pd.Series(all_categories).value_counts()

        # Confidence stats
        confidences = results_df["confidence"].astype(float)

        stats = {
            "total_tickets": total_tickets,
            "classified_tickets": classified_tickets,
            "classification_rate": classification_rate,
            "avg_confidence": confidences.mean(),
            "median_confidence": confidences.median(),
            "high_confidence_count": (confidences >= self.confidence_threshold).sum(),
            "category_distribution": category_counts.to_dict(),
            "most_common_category": (
                category_counts.index[0] if len(category_counts) > 0 else None
            ),
            "total_processing_time": results_df["processing_time"].sum(),
            "avg_processing_time": results_df["processing_time"].mean(),
            "total_tokens_used": results_df["tokens_used"].sum(),
        }

        return stats


# Utility functions for integration with Opção D pipeline
def load_and_classify_tickets(
    categories_path: Path, tickets_path: Path, output_path: Path, api_key: str, **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load categories and classify tickets.

    Args:
        categories_path: Path to categories.json
        tickets_path: Path to tickets CSV
        output_path: Path to save results
        api_key: Google API key
        **kwargs: Additional classifier parameters

    Returns:
        Classification results DataFrame
    """
    # Initialize classifier
    classifier = FastClassifier(api_key=api_key, **kwargs)

    # Load categories
    classifier.load_categories(categories_path)

    # Load tickets
    tickets_df = pd.read_csv(tickets_path, sep=";", encoding="utf-8-sig")

    # Classify tickets
    results_df = classifier.classify_all_tickets(tickets_df, output_path)

    return results_df


def validate_classification_accuracy(
    results_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Validate classification accuracy against ground truth.

    Args:
        results_df: Classification results
        ground_truth_df: Ground truth classifications

    Returns:
        Accuracy metrics
    """
    # Implementation for accuracy validation
    # This would compare predicted vs actual categories
    pass
