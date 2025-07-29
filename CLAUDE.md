# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Essential Commands
```bash
# Environment Setup
pip install -r requirements.txt            # Install Python dependencies
python -m venv venv                        # Create virtual environment
source venv/bin/activate                   # Activate virtual environment (Linux/Mac)
venv\Scripts\activate                      # Activate virtual environment (Windows)

# Core Processing Operations
python src/main.py --mode categorize       # Run ticket categorization only
python src/main.py --mode summarize        # Run ticket summarization only  
python src/main.py --mode merge            # Merge categorized and summarized results
python src/main.py --mode all              # Run full pipeline (categorize → summarize → merge)

# Data Processing Options
python src/main.py --mode all --nrows 1000     # Process only first 1000 rows
python src/main.py --mode all --no-cache       # Disable caching (process everything fresh)
python src/main.py --mode all --workers 8      # Set number of parallel workers
python src/main.py --mode all --input-file "database/custom_file.csv"  # Specify input file

# Cache Management
rm -rf database/cache/*                    # Clear all cached results
```

## Architecture Overview

### Core Application Structure
This is a LangChain-based conversation categorization and summarization system that processes customer support tickets using Google's Gemini AI models.

**Project Structure**:
```
src/
├── main.py                 # Entry point and CLI interface
├── base_processor.py       # Base class with common functionality
├── categorizer.py          # Ticket categorization using map-reduce
├── summarizer.py           # Ticket summarization and bullet extraction
└── merger.py               # Combines categorization and summarization results
database/
├── cache/                  # Pickle cache files for processed data
├── *.csv                   # Input CSV files with ticket data
└── [output files]          # Generated results
DOC/                        # Project requirements and documentation
├── PRD_*.md               # Product requirement documents
└── README_*.md            # Comparison and analysis documents
```

**Data Processing Pipeline**:
1. **Data Preparation**: Filters tickets by category='TEXT', removes AI messages, validates conversation flow
2. **Categorization**: Map-reduce approach with chunk processing and parallel execution
3. **Summarization**: Multi-phase summarization with bullet point extraction
4. **Merging**: Combines results into final analysis file

### Critical Architecture Patterns

**Map-Reduce Processing**:
- Large datasets split into manageable chunks using `TokenTextSplitter`
- Map phase: Analyzes individual chunks in parallel
- Reduce phase: Consolidates partial results
- Final phase: Applies categorization/summarization to processed data

**Caching System**:
- Intelligent cache using SHA-256 hashes of input data
- Cached results stored as pickle files in `database/cache/`
- Cache keys based on file content, processing parameters, and method type
- Automatic cache invalidation when input data changes

**Error Handling and Resilience**:
- Comprehensive retry logic with exponential backoff
- Detailed error logging with context preservation
- Graceful degradation when individual chunks fail
- Rate limiting compliance for API calls

**Text Processing**:
- Extensive text cleaning and normalization
- Unicode character replacement and ASCII conversion
- URL sanitization and special character handling
- Preserves conversation structure while removing noise

### Key Business Logic Patterns

**Ticket Validation Flow**:
1. Filter by category='TEXT' (case insensitive)
2. Remove AI-generated messages
3. Require minimum 2 USER and 2 AGENT/HELPDESK_INTEGRATION messages per ticket
4. Aggregate all messages per ticket_id
5. Apply comprehensive text cleaning

**Categorization Process** (Map-Reduce Pattern):
1. **Map Phase**: Analyze text chunks and identify category patterns
2. **Combine Phase**: Consolidate partial analyses into unified view
3. **Categorization Phase**: Apply categories to individual tickets using consolidated analysis
4. **Validation**: Ensure JSON format compliance and category consistency

**Summarization Process** (Map-Reduce Pattern):
1. **Map Phase**: Generate concise summaries of text chunks
2. **Combine Phase**: Merge partial summaries into consolidated analysis
3. **Bullet Extraction**: Generate exactly 15 actionable bullet points plus general summary
4. **Validation**: Ensure proper JSON structure and content requirements

### Development Guidelines

**Environment Configuration**:
- Requires `GOOGLE_API_KEY` environment variable for Gemini AI access
- Uses `.env` file for environment variable management
- Python 3.8+ required with specific dependency versions in `requirements.txt`

**Data Input Requirements**:
- CSV files with columns: `ticket_id`, `category`, `text`, `sender`, `ticket_created_at`
- Supports Excel files (.xlsx, .xls) with same column structure
- Uses semicolon (;) as CSV separator
- UTF-8-SIG encoding for proper character handling

**Processing Configuration**:
- Default model: `gemini-2.5-flash` with temperature=0.3
- Chunk sizes: 1M tokens for categorization, 900K for summarization
- Overlap: 10K tokens for categorization, 90K for summarization
- Parallel processing with configurable worker limits (default: min(cpu_count, 4))

**Output Formats**:
- Categorization: `categorized_tickets.csv` (ticket_id, categoria)
- Summarization: `summarized_tickets.csv` (bullets) + `summarized_tickets_resumo.txt` (summary)
- Final merge: `final_analysis.csv` (combined results)

**Error Handling Patterns**:
- All errors logged with timestamps and full context to `error_log_*.log` files
- Cache automatically handles corrupted or invalid cache files
- Batch processing continues even if individual batches fail
- Detailed token usage tracking and cost estimation

**Performance Optimization**:
- Concurrent processing using ThreadPoolExecutor
- Intelligent batching based on token limits
- Progress bars using tqdm for long-running operations
- Memory-efficient chunk processing

### Testing and Validation

**Data Validation**:
- Automatic validation of ticket conversation structure
- JSON response format verification
- Category consistency checking
- Token usage monitoring and cost estimation

**Processing Verification**:
- Detailed logging of processing statistics
- Input/output token counting per phase
- Success/failure tracking per batch
- Cache hit/miss reporting

### Implementation Notes

**LangChain Integration**:
- Uses modern LangChain expression language (LCEL) with pipe operators
- Runnable chains for composable processing pipelines
- Document splitting with TokenTextSplitter
- Template-based prompt engineering

**Google Gemini AI**:
- Configured for optimal performance with specific temperature settings
- Token estimation using character-based approximation (1 token ≈ 4 characters)
- Rate limiting awareness with retry mechanisms
- Cost tracking with per-token pricing calculations

**File Handling**:
- Robust CSV/Excel reading with error recovery using pandas
- UTF-8 encoding consistency throughout
- Path validation and file existence checking
- Interactive file selection for user convenience

**BaseProcessor Class**:
- Provides common functionality for text cleaning, caching, and parallel processing
- Implements intelligent cache management with automatic invalidation
- Handles token estimation and cost tracking
- Manages concurrent execution with configurable worker pools

This system is designed for processing large volumes of customer support conversations with high reliability, consistent categorization, and actionable insights generation. The map-reduce architecture ensures scalability while the comprehensive caching system optimizes performance for iterative development and testing.