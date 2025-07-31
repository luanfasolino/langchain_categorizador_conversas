"""
Streaming Processor - Memory-efficient streaming system for large datasets.

This module provides streaming data processing capabilities to handle
datasets from 19K to 500K+ tickets without memory constraints.
"""

import gc
import gzip
import json
import pickle
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming operations."""

    chunk_size_rows: int = 1000
    buffer_size_mb: int = 64
    compression_enabled: bool = True
    memory_limit_mb: int = 512
    temp_dir: Optional[Path] = None
    cleanup_temp_files: bool = True


class MemoryMonitor:
    """Monitor and manage memory usage during streaming."""

    def __init__(self, limit_mb: int = 512):
        self.limit_mb = limit_mb
        self.peak_usage_mb = 0

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        usage_mb = process.memory_info().rss / 1024 / 1024
        self.peak_usage_mb = max(self.peak_usage_mb, usage_mb)
        return usage_mb

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_current_usage_mb()
        return current_usage < self.limit_mb

    def force_cleanup(self):
        """Force garbage collection to free memory."""
        gc.collect()


class ChunkedFileReader:
    """Memory-efficient file reader with configurable buffering."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_mb)

    def read_csv_chunks(
        self, file_path: Path, chunksize: Optional[int] = None, **pandas_kwargs
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Read CSV file in chunks to minimize memory usage.

        Args:
            file_path: Path to CSV file
            chunksize: Number of rows per chunk (uses config default if None)
            **pandas_kwargs: Additional arguments for pd.read_csv

        Yields:
            DataFrame chunks
        """
        chunksize = chunksize or self.config.chunk_size_rows

        # Default pandas arguments optimized for memory efficiency
        default_args = {
            "dtype": str,  # Read everything as string to save memory
            "engine": "c",  # Use C engine for speed
            "low_memory": False,
            "chunksize": chunksize,
        }

        # Merge with user arguments
        pandas_args = {**default_args, **pandas_kwargs}

        logger.info(f"Reading CSV in chunks of {chunksize:,} rows from {file_path}")

        try:
            chunk_count = 0
            for chunk in pd.read_csv(file_path, **pandas_args):
                chunk_count += 1

                # Memory management
                current_memory = self.memory_monitor.get_current_usage_mb()
                logger.debug(
                    f"Chunk {chunk_count}: {len(chunk):,} rows, Memory: {current_memory:.1f}MB"
                )

                # Force cleanup if approaching memory limit
                if not self.memory_monitor.check_memory_limit():
                    logger.warning(
                        f"Memory usage high ({current_memory:.1f}MB), forcing cleanup"
                    )
                    self.memory_monitor.force_cleanup()

                yield chunk

                # Clean up chunk reference to free memory
                del chunk

        except Exception as e:
            logger.error(f"Error reading CSV chunks: {str(e)}")
            raise

    def read_excel_chunks(
        self, file_path: Path, chunksize: Optional[int] = None, **pandas_kwargs
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Read Excel file in chunks (loads full file then yields chunks).

        Note: Excel doesn't support native chunking, so we load and chunk.

        Args:
            file_path: Path to Excel file
            chunksize: Number of rows per chunk
            **pandas_kwargs: Additional arguments for pd.read_excel

        Yields:
            DataFrame chunks
        """
        chunksize = chunksize or self.config.chunk_size_rows

        logger.info(f"Reading Excel file {file_path} (will be chunked after loading)")

        try:
            # Load the full Excel file
            df = pd.read_excel(file_path, dtype=str, **pandas_kwargs)
            total_rows = len(df)

            logger.info(
                f"Excel loaded: {total_rows:,} total rows, chunking into {chunksize:,} row pieces"
            )

            # Yield chunks
            for start_idx in range(0, total_rows, chunksize):
                end_idx = min(start_idx + chunksize, total_rows)
                chunk = df.iloc[start_idx:end_idx].copy()

                chunk_num = (start_idx // chunksize) + 1
                logger.debug(f"Excel chunk {chunk_num}: rows {start_idx:,}-{end_idx:,}")

                yield chunk

                # Clean up chunk reference
                del chunk

                # Memory check
                if not self.memory_monitor.check_memory_limit():
                    self.memory_monitor.force_cleanup()

            # Clean up main dataframe
            del df
            self.memory_monitor.force_cleanup()

        except Exception as e:
            logger.error(f"Error reading Excel chunks: {str(e)}")
            raise

    def read_json_chunks(
        self, file_path: Path, chunksize: Optional[int] = None
    ) -> Generator[List[Dict], None, None]:
        """
        Read JSON file in chunks.

        Args:
            file_path: Path to JSON file
            chunksize: Number of records per chunk

        Yields:
            List of dictionaries (records)
        """
        chunksize = chunksize or self.config.chunk_size_rows

        logger.info(f"Reading JSON in chunks of {chunksize:,} records from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    # Process list in chunks
                    for i in range(0, len(data), chunksize):
                        chunk = data[i : i + chunksize]
                        yield chunk
                        del chunk
                else:
                    # Single object - yield as single chunk
                    yield [data]

        except Exception as e:
            logger.error(f"Error reading JSON chunks: {str(e)}")
            raise


class StreamingDataProcessor:
    """Main streaming data processor for large datasets."""

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.file_reader = ChunkedFileReader(self.config)
        self.temp_files: List[Path] = []

        # Setup temp directory
        if not self.config.temp_dir:
            self.config.temp_dir = Path("database/temp_streaming")
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"StreamingDataProcessor initialized with {self.config.chunk_size_rows:,} row chunks"
        )

    def process_file_streaming(
        self,
        input_file: Path,
        processor_function: callable,
        output_file: Optional[Path] = None,
        nrows: Optional[int] = None,
    ) -> Generator[Any, None, None]:
        """
        Process a file using streaming with a custom processor function.

        Args:
            input_file: Input file path
            processor_function: Function to process each chunk
            output_file: Optional output file for results
            nrows: Limit number of rows to process

        Yields:
            Processed results from each chunk
        """
        logger.info(f"Starting streaming processing of {input_file}")

        total_processed = 0
        chunk_count = 0

        try:
            # Determine file type and get appropriate reader
            if input_file.suffix.lower() == ".csv":
                chunks = self.file_reader.read_csv_chunks(input_file)
            elif input_file.suffix.lower() in [".xlsx", ".xls"]:
                chunks = self.file_reader.read_excel_chunks(input_file)
            elif input_file.suffix.lower() == ".json":
                chunks = self.file_reader.read_json_chunks(input_file)
            else:
                raise ValueError(f"Unsupported file type: {input_file.suffix}")

            # Process each chunk
            for chunk in chunks:
                chunk_count += 1

                # Check row limit
                if nrows and total_processed >= nrows:
                    logger.info(f"Reached row limit ({nrows:,}), stopping processing")
                    break

                # Truncate chunk if needed
                if nrows and total_processed + len(chunk) > nrows:
                    remaining_rows = nrows - total_processed
                    chunk = chunk.iloc[:remaining_rows]

                logger.info(f"Processing chunk {chunk_count}: {len(chunk):,} rows")

                # Process the chunk
                try:
                    processed_chunk = processor_function(chunk, chunk_count)
                    total_processed += len(chunk)

                    # Yield result
                    yield processed_chunk

                    # Save intermediate results if output file specified
                    if output_file:
                        self._save_chunk_result(processed_chunk, chunk_count)

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {str(e)}")
                    # Continue with next chunk instead of failing completely
                    continue

                # Memory cleanup
                del chunk
                if chunk_count % 10 == 0:  # Force cleanup every 10 chunks
                    self.file_reader.memory_monitor.force_cleanup()

            logger.info(
                f"Streaming processing completed: {total_processed:,} rows in {chunk_count} chunks"
            )

        except Exception as e:
            logger.error(f"Error in streaming processing: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.config.cleanup_temp_files:
                self._cleanup_temp_files()

    def merge_streaming_results(
        self,
        result_generator: Generator,
        output_file: Path,
        merge_function: Optional[callable] = None,
    ) -> Path:
        """
        Merge streaming results into a single output file.

        Args:
            result_generator: Generator yielding processed chunks
            output_file: Final output file path
            merge_function: Custom function to merge results (default: concatenate)

        Returns:
            Path to merged output file
        """
        logger.info(f"Merging streaming results to {output_file}")

        if not merge_function:
            merge_function = self._default_merge_function

        try:
            all_results = []
            chunk_count = 0

            for chunk_result in result_generator:
                chunk_count += 1
                all_results.append(chunk_result)

                # Periodic merge to manage memory
                if chunk_count % 50 == 0:
                    logger.info(f"Merging intermediate results (chunk {chunk_count})")
                    merged_partial = merge_function(all_results)

                    # Save intermediate merge
                    temp_file = self._save_intermediate_merge(
                        merged_partial, chunk_count
                    )

                    # Clear results list to free memory
                    all_results = [temp_file]

            # Final merge
            final_result = merge_function(all_results)

            # Save final result
            self._save_final_result(final_result, output_file)

            logger.info(f"Streaming merge completed: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error merging streaming results: {str(e)}")
            raise

    def _default_merge_function(self, results: List[Any]) -> Any:
        """Default function to merge results (concatenate DataFrames or lists)."""
        if not results:
            return None

        # Handle different result types
        first_result = results[0]

        if isinstance(first_result, pd.DataFrame):
            # Concatenate DataFrames
            return pd.concat(results, ignore_index=True)
        elif isinstance(first_result, list):
            # Flatten lists
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        elif isinstance(first_result, dict):
            # Merge dictionaries (assuming they have list values)
            merged = {}
            for result in results:
                for key, value in result.items():
                    if key not in merged:
                        merged[key] = []
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
            return merged
        else:
            # Default: return list of all results
            return results

    def _save_chunk_result(self, result: Any, chunk_number: int):
        """Save individual chunk result to temporary file."""
        temp_file = self.config.temp_dir / f"chunk_{chunk_number}.pkl"

        if self.config.compression_enabled:
            temp_file = temp_file.with_suffix(".pkl.gz")
            with gzip.open(temp_file, "wb") as f:
                pickle.dump(result, f)
        else:
            with open(temp_file, "wb") as f:
                pickle.dump(result, f)

        self.temp_files.append(temp_file)
        return temp_file

    def _save_intermediate_merge(self, result: Any, chunk_number: int) -> Path:
        """Save intermediate merge result."""
        temp_file = self.config.temp_dir / f"intermediate_merge_{chunk_number}.pkl"

        if self.config.compression_enabled:
            temp_file = temp_file.with_suffix(".pkl.gz")
            with gzip.open(temp_file, "wb") as f:
                pickle.dump(result, f)
        else:
            with open(temp_file, "wb") as f:
                pickle.dump(result, f)

        self.temp_files.append(temp_file)
        return temp_file

    def _save_final_result(self, result: Any, output_file: Path):
        """Save final merged result to output file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(result, pd.DataFrame):
            # Save as CSV or Excel based on extension
            if output_file.suffix.lower() == ".csv":
                result.to_csv(output_file, index=False, encoding="utf-8")
            elif output_file.suffix.lower() in [".xlsx", ".xls"]:
                result.to_excel(output_file, index=False)
            else:
                # Default to CSV
                result.to_csv(
                    output_file.with_suffix(".csv"), index=False, encoding="utf-8"
                )
        elif isinstance(result, (list, dict)):
            # Save as JSON
            with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            # Save as pickle
            with open(output_file.with_suffix(".pkl"), "wb") as f:
                pickle.dump(result, f)

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if not self.temp_files:
            return

        logger.info(f"Cleaning up {len(self.temp_files)} temporary files")

        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file}: {str(e)}")

        self.temp_files.clear()

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get statistics about streaming processing."""
        memory_usage = self.file_reader.memory_monitor.get_current_usage_mb()
        peak_memory = self.file_reader.memory_monitor.peak_usage_mb

        return {
            "config": {
                "chunk_size_rows": self.config.chunk_size_rows,
                "buffer_size_mb": self.config.buffer_size_mb,
                "memory_limit_mb": self.config.memory_limit_mb,
                "compression_enabled": self.config.compression_enabled,
            },
            "memory_usage": {
                "current_mb": memory_usage,
                "peak_mb": peak_memory,
                "limit_mb": self.config.memory_limit_mb,
                "usage_percent": (memory_usage / self.config.memory_limit_mb) * 100,
            },
            "temp_files": {
                "count": len(self.temp_files),
                "total_size_mb": sum(
                    f.stat().st_size / 1024 / 1024
                    for f in self.temp_files
                    if f.exists()
                ),
            },
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.config.cleanup_temp_files:
            self._cleanup_temp_files()


class StreamingTicketProcessor:
    """
    Specialized streaming processor for ticket data processing.

    Integrates with existing BaseProcessor but operates in streaming mode.
    """

    def __init__(
        self, base_processor, streaming_config: Optional[StreamingConfig] = None
    ):
        self.base_processor = base_processor
        self.streaming_config = streaming_config or StreamingConfig()
        self.data_processor = StreamingDataProcessor(self.streaming_config)

    def process_tickets_streaming(
        self, input_file: Path, output_file: Path, nrows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process tickets using streaming approach.

        Args:
            input_file: Input CSV/Excel file with ticket data
            output_file: Output file for processed results
            nrows: Limit number of rows to process

        Returns:
            Processing statistics
        """
        logger.info(
            f"Starting streaming ticket processing: {input_file} -> {output_file}"
        )

        def process_ticket_chunk(chunk: pd.DataFrame, chunk_number: int) -> List[Dict]:
            """Process a single chunk of tickets."""
            try:
                # Convert chunk to ticket format expected by base processor
                chunk_tickets = []

                for _, row in chunk.iterrows():
                    ticket = {
                        "ticket_id": str(row.get("ticket_id", "")),
                        "text": str(row.get("text", "")),
                        "sender": str(row.get("sender", "")),
                        "category": str(row.get("category", "")),
                    }
                    chunk_tickets.append(ticket)

                # Apply base processor's data preparation logic
                # Note: This is simplified - in practice, you'd adapt prepare_data for chunks
                filtered_tickets = []
                for ticket in chunk_tickets:
                    # Apply same filtering logic as BaseProcessor
                    if (
                        ticket["category"].lower().strip() == "text"
                        and ticket["text"].strip()
                        and ticket["ticket_id"].strip()
                        and ticket["sender"].lower()
                        not in self.base_processor.AI_SENDER_PATTERNS
                    ):

                        # Clean text
                        ticket["text"] = self.base_processor.clean_text(ticket["text"])
                        filtered_tickets.append(ticket)

                logger.info(
                    f"Chunk {chunk_number}: {len(chunk)} -> {len(filtered_tickets)} tickets after filtering"
                )
                return filtered_tickets

            except Exception as e:
                logger.error(f"Error processing ticket chunk {chunk_number}: {str(e)}")
                return []

        # Process file in streaming mode
        processing_stats = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "start_time": pd.Timestamp.now(),
            "total_chunks": 0,
            "total_input_tickets": 0,
            "total_output_tickets": 0,
        }

        try:
            # Process streaming
            result_generator = self.data_processor.process_file_streaming(
                input_file=input_file,
                processor_function=process_ticket_chunk,
                nrows=nrows,
            )

            # Collect and merge results
            all_results = []
            for chunk_result in result_generator:
                processing_stats["total_chunks"] += 1
                processing_stats["total_output_tickets"] += len(chunk_result)
                all_results.extend(chunk_result)

            # Save final results
            if all_results:
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(output_file, index=False, encoding="utf-8")
                logger.info(
                    f"Saved {len(all_results):,} processed tickets to {output_file}"
                )

            # Update stats
            processing_stats["end_time"] = pd.Timestamp.now()
            processing_stats["duration"] = (
                processing_stats["end_time"] - processing_stats["start_time"]
            )
            processing_stats["streaming_stats"] = (
                self.data_processor.get_streaming_stats()
            )

            return processing_stats

        except Exception as e:
            logger.error(f"Error in streaming ticket processing: {str(e)}")
            raise
