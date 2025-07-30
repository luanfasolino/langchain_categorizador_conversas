"""
Category Pattern Discovery Engine

This module implements the discovery phase processing system that analyzes 
sample tickets to identify and extract category patterns using map-reduce architecture.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Import base processor with fallback for testing
try:
    from ..base_processor import BaseProcessor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_processor import BaseProcessor


class CategoryDiscoverer(BaseProcessor):
    """
    Category Pattern Discovery Engine using Map-Reduce architecture.
    
    Implements a three-phase discovery process:
    1. MAP: Analyze chunks of sample tickets to identify patterns
    2. COMBINE: Consolidate patterns from all chunks  
    3. EXTRACT: Generate final category taxonomy
    """
    
    def __init__(self, 
                 api_key: str,
                 database_dir: Optional[Path] = None,
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1,
                 max_workers: int = 4):
        """
        Initialize the Category Discovery Engine.
        
        Args:
            api_key: Google API key for Gemini
            database_dir: Directory for data and cache
            model_name: LLM model to use
            temperature: LLM temperature (lower = more consistent)
            max_workers: Maximum concurrent workers
        """
        # Set default database directory
        if database_dir is None:
            database_dir = Path.cwd() / "database"
        
        # Initialize parent with correct signature
        super().__init__(
            api_key=api_key,
            database_dir=database_dir,
            max_tickets_per_batch=50,
            max_workers=max_workers,
            use_cache=True
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Override LLM with custom settings
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )
        
        # Discovery-specific configurations
        self.chunk_size = 800_000  # 800K tokens for discovery phase
        self.overlap = 240_000     # 30% overlap for better context
        self.min_categories = 5    # Minimum categories to discover
        self.max_categories = 25   # Maximum categories to prevent over-fragmentation
        
        # Initialize text splitter for discovery
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=self.estimate_tokens,  # Use inherited method
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True
        )
        
        # Initialize discovery chains
        self._setup_discovery_chains()
        
        self.logger.info(f"CategoryDiscoverer initialized with {model_name}, "
                        f"chunk_size={self.chunk_size}, overlap={self.overlap}")
    
    def _setup_discovery_chains(self):
        """Setup the map-reduce chains for category discovery."""
        
        # MAP PHASE: Pattern analysis chain
        map_template = """
Você é um especialista em análise de atendimento ao cliente.

Analise as conversas de tickets de suporte abaixo e identifique:

1. **PADRÕES DE PROBLEMAS**: Que tipos de problemas/solicitações aparecem?
2. **TEMAS RECORRENTES**: Quais assuntos são mencionados frequentemente?
3. **CATEGORIAS EMERGENTES**: Agrupe problemas similares em categorias lógicas
4. **PALAVRAS-CHAVE**: Identifique termos técnicos, produtos ou ações específicas

Para cada padrão identificado, forneça:
- Nome descritivo do padrão
- Descrição clara (1-2 frases)
- Palavras-chave associadas
- Exemplos representativos

IMPORTANTE:
- Foque em PADRÕES, não em tickets individuais
- Seja específico mas não excessivamente granular
- Considere o contexto de atendimento ao cliente
- Identifique tanto problemas técnicos quanto solicitações comerciais

CONVERSAS:
{context}

Retorne sua análise de padrões em formato JSON estruturado:
"""

        combine_template = """
Você é um especialista em taxonomias de atendimento ao cliente.

Analise todas as análises de padrões abaixo e consolide em uma visão unificada:

1. **IDENTIFIQUE PADRÕES COMUNS**: Que padrões aparecem em múltiplas análises?
2. **RESOLVA DUPLICAÇÕES**: Agrupe padrões similares
3. **HIERARQUIZE**: Organize padrões em categorias principais e subcategorias
4. **NORMALIZE**: Padronize nomenclatura e descrições

ANÁLISES DE PADRÕES:
{docs}

Consolide em uma análise unificada que servirá para gerar a taxonomia final.
Retorne em formato JSON estruturado com padrões consolidados:
"""

        extract_template = """
Você é um especialista em criação de taxonomias de categorização.

Com base na análise consolidada abaixo, crie uma taxonomia completa de categorias para classificação de tickets de suporte.

ANÁLISE CONSOLIDADA:
{consolidated_analysis}

Crie uma taxonomia com:

1. **CATEGORIAS PRINCIPAIS** (5-15 categorias)
2. **SUBCATEGORIAS** quando apropriado (máximo 3 níveis)
3. **DEFINIÇÕES CLARAS** para cada categoria
4. **PALAVRAS-CHAVE** associadas
5. **EXEMPLOS** representativos

CRITÉRIOS:
- Categorias mutuamente exclusivas quando possível
- Cobertura abrangente dos padrões identificados
- Nomenclatura técnica padronizada (snake_case) 
- Nomes de exibição amigáveis
- Balanceamento entre especificidade e generalização

Retorne EXATAMENTE no formato JSON especificado:

{{
  "version": "1.0",
  "generated_at": "{timestamp}",
  "discovery_stats": {{
    "total_patterns_analyzed": <número>,
    "categories_created": <número>,
    "confidence_level": <0.0-1.0>
  }},
  "categories": [
    {{
      "id": 1,
      "technical_name": "payment_issues",
      "display_name": "Problemas de Pagamento",
      "description": "Falhas em transações, cartões recusados, cobranças indevidas",
      "keywords": ["pagamento", "cartão", "cobrança", "transação", "recusado"],
      "examples": ["Meu cartão foi recusado", "Cobrança duplicada"],
      "subcategories": [
        {{
          "id": 11,
          "technical_name": "card_declined",
          "display_name": "Cartão Recusado",
          "description": "Transações recusadas por problemas com cartão",
          "keywords": ["cartão recusado", "pagamento negado"]
        }}
      ]
    }}
  ],
  "metadata": {{
    "llm_model": "{model_name}",
    "discovery_method": "map_reduce_pattern_analysis",
    "chunk_size": {chunk_size},
    "overlap_tokens": {overlap}
  }}
}}
"""

        # Create prompts
        self.map_prompt = ChatPromptTemplate.from_template(map_template)
        self.combine_prompt = ChatPromptTemplate.from_template(combine_template)
        self.extract_prompt = ChatPromptTemplate.from_template(extract_template)
        
        # Create chains using LCEL
        self.map_chain = self.map_prompt | self.llm | StrOutputParser()
        self.combine_chain = self.combine_prompt | self.llm | StrOutputParser()
        self.extract_chain = self.extract_prompt | self.llm | StrOutputParser()
        
        self.logger.info("Discovery chains initialized successfully")
    
    def discover_categories(self, 
                          tickets_df: pd.DataFrame,
                          output_path: Optional[Path] = None,
                          force_rediscovery: bool = False) -> Dict[str, Any]:
        """
        Discover categories from sample tickets using map-reduce pattern.
        
        Args:
            tickets_df: DataFrame with sampled tickets
            output_path: Path to save categories.json
            force_rediscovery: Skip cache and force new discovery
            
        Returns:
            Dictionary with discovered categories
        """
        if tickets_df.empty:
            raise ValueError("Cannot discover categories from empty dataset")
        
        # Generate cache key based on tickets content
        tickets_content = tickets_df.to_string()
        cache_key = self._generate_cache_key(tickets_content + "category_discovery")
        
        # Check cache first
        if not force_rediscovery:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info("Loading categories from cache")
                if output_path:
                    self._save_categories_json(cached_result, output_path)
                return cached_result
        
        self.logger.info(f"Starting category discovery on {len(tickets_df)} sample tickets")
        
        try:
            # Phase 1: MAP - Analyze patterns in chunks
            self.logger.info("Phase 1: MAP - Analyzing patterns in chunks")
            pattern_analyses = self._map_pattern_analysis(tickets_df)
            
            # Phase 2: COMBINE - Consolidate all patterns
            self.logger.info("Phase 2: COMBINE - Consolidating patterns")
            consolidated_analysis = self._combine_patterns(pattern_analyses)
            
            # Phase 3: EXTRACT - Generate final taxonomy
            self.logger.info("Phase 3: EXTRACT - Generating category taxonomy")
            categories = self._extract_categories(consolidated_analysis)
            
            # Validate and enhance categories
            categories = self._validate_and_enhance_categories(categories, tickets_df)
            
            # Cache results
            self._save_to_cache(cache_key, categories)
            
            # Save to file if requested
            if output_path:
                self._save_categories_json(categories, output_path)
            
            self.logger.info(f"Category discovery completed: {categories['discovery_stats']['categories_created']} categories found")
            return categories
            
        except Exception as e:
            self.logger.error(f"Category discovery failed: {str(e)}")
            raise
    
    def _map_pattern_analysis(self, tickets_df: pd.DataFrame) -> List[str]:
        """
        Phase 1: Map pattern analysis across ticket chunks.
        
        Args:
            tickets_df: Sample tickets DataFrame
            
        Returns:
            List of pattern analysis results
        """
        # Prepare text for analysis
        tickets_text = self._prepare_tickets_text(tickets_df)
        
        # Create chunks
        chunks = self._create_discovery_chunks(tickets_text)
        self.logger.info(f"Created {len(chunks)} chunks for pattern analysis")
        
        # Analyze patterns in parallel
        pattern_analyses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self._analyze_chunk_patterns, chunk, i): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    analysis = future.result()
                    pattern_analyses.append(analysis)
                    self.logger.debug(f"Completed pattern analysis for chunk {chunk_idx}")
                except Exception as e:
                    self.logger.error(f"Pattern analysis failed for chunk {chunk_idx}: {str(e)}")
                    continue
        
        if not pattern_analyses:
            raise RuntimeError("No successful pattern analyses completed")
        
        self.logger.info(f"MAP phase completed: {len(pattern_analyses)} pattern analyses")
        return pattern_analyses
    
    def _combine_patterns(self, pattern_analyses: List[str]) -> str:
        """
        Phase 2: Combine all pattern analyses into consolidated view.
        
        Args:
            pattern_analyses: List of individual pattern analyses
            
        Returns:
            Consolidated analysis string
        """
        # Combine all analyses into single document
        combined_doc = "\n\n---ANÁLISE---\n\n".join(pattern_analyses)
        
        # Generate consolidated analysis
        try:
            consolidated = self.combine_chain.invoke({"docs": combined_doc})
            self.logger.info("COMBINE phase completed successfully")
            return consolidated
        except Exception as e:
            self.logger.error(f"Pattern combination failed: {str(e)}")
            raise
    
    def _extract_categories(self, consolidated_analysis: str) -> Dict[str, Any]:
        """
        Phase 3: Extract final category taxonomy from consolidated analysis.
        
        Args:
            consolidated_analysis: Consolidated pattern analysis
            
        Returns:
            Categories dictionary
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Generate categories
            categories_str = self.extract_chain.invoke({
                "consolidated_analysis": consolidated_analysis,
                "timestamp": timestamp,
                "model_name": self.model_name,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            })
            
            # Parse JSON response
            categories = json.loads(categories_str)
            
            self.logger.info("EXTRACT phase completed successfully")
            return categories
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse categories JSON: {str(e)}")
            # Try to extract JSON from response
            categories_str = self._extract_json_from_response(categories_str)
            try:
                categories = json.loads(categories_str)
                return categories
            except:
                raise RuntimeError("Could not parse categories response as valid JSON")
        except Exception as e:
            self.logger.error(f"Category extraction failed: {str(e)}")
            raise
    
    def _analyze_chunk_patterns(self, chunk: str, chunk_idx: int) -> str:
        """
        Analyze patterns in a single chunk.
        
        Args:
            chunk: Text chunk to analyze
            chunk_idx: Chunk index for logging
            
        Returns:
            Pattern analysis string
        """
        try:
            analysis = self.map_chain.invoke({"context": chunk})
            return analysis
        except Exception as e:
            self.logger.error(f"Chunk {chunk_idx} analysis failed: {str(e)}")
            raise
    
    def _prepare_tickets_text(self, tickets_df: pd.DataFrame) -> str:
        """
        Prepare tickets text for discovery analysis.
        
        Args:
            tickets_df: Tickets DataFrame
            
        Returns:
            Formatted text string
        """
        ticket_texts = []
        
        for ticket_id, group in tickets_df.groupby('ticket_id'):
            # Sort messages chronologically
            messages = group.sort_values('message_sended_at') if 'message_sended_at' in group.columns else group
            
            # Format conversation
            conversation = f"TICKET {ticket_id}:\n"
            for _, msg in messages.iterrows():
                sender = msg.get('sender', 'UNKNOWN')
                text = msg.get('text', '')
                conversation += f"[{sender}]: {text}\n"
            
            conversation += "\n"
            ticket_texts.append(conversation)
        
        combined_text = "\n".join(ticket_texts)
        self.logger.info(f"Prepared text for {len(ticket_texts)} tickets, {len(combined_text)} characters")
        return combined_text
    
    def _create_discovery_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for pattern discovery.
        
        Args:
            text: Combined tickets text
            
        Returns:
            List of text chunks
        """
        # Create documents and split
        doc = Document(page_content=text)
        chunks = self.text_splitter.split_documents([doc])
        
        # Extract text content
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        self.logger.info(f"Created {len(chunk_texts)} discovery chunks")
        return chunk_texts
    
    def _validate_and_enhance_categories(self, 
                                       categories: Dict[str, Any], 
                                       tickets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate and enhance discovered categories.
        
        Args:
            categories: Raw categories from extraction
            tickets_df: Original sample tickets
            
        Returns:
            Enhanced categories dictionary
        """
        if "categories" not in categories:
            raise ValueError("Invalid categories structure: missing 'categories' key")
        
        category_list = categories["categories"]
        
        # Validate category count
        if len(category_list) < self.min_categories:
            self.logger.warning(f"Only {len(category_list)} categories found, minimum is {self.min_categories}")
        elif len(category_list) > self.max_categories:
            self.logger.warning(f"{len(category_list)} categories found, maximum is {self.max_categories}")
        
        # Enhance with statistics
        categories["discovery_stats"]["total_tickets_analyzed"] = len(tickets_df)
        categories["discovery_stats"]["unique_tickets"] = tickets_df['ticket_id'].nunique()
        categories["discovery_stats"]["categories_created"] = len(category_list)
        
        # Add quality metrics
        avg_keywords = np.mean([len(cat.get('keywords', [])) for cat in category_list])
        has_subcategories = sum(1 for cat in category_list if cat.get('subcategories'))
        
        categories["discovery_stats"]["avg_keywords_per_category"] = round(avg_keywords, 1)
        categories["discovery_stats"]["categories_with_subcategories"] = has_subcategories
        categories["discovery_stats"]["confidence_level"] = min(0.95, 0.6 + (len(category_list) * 0.02))
        
        return categories
    
    def _save_categories_json(self, categories: Dict[str, Any], output_path: Path):
        """
        Save categories to JSON file.
        
        Args:
            categories: Categories dictionary
            output_path: Path to save file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(categories, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Categories saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save categories to {output_path}: {str(e)}")
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response that might have extra text.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Find JSON boundaries
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        
        raise ValueError("No valid JSON found in response")
    
    def load_categories(self, categories_path: Path) -> Dict[str, Any]:
        """
        Load categories from JSON file.
        
        Args:
            categories_path: Path to categories.json file
            
        Returns:
            Categories dictionary
        """
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            self.logger.info(f"Loaded {len(categories.get('categories', []))} categories from {categories_path}")
            return categories
            
        except Exception as e:
            self.logger.error(f"Failed to load categories from {categories_path}: {str(e)}")
            raise
    
    def get_discovery_stats(self, categories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive discovery statistics.
        
        Args:
            categories: Categories dictionary
            
        Returns:
            Statistics dictionary
        """
        if "categories" not in categories:
            return {}
        
        category_list = categories["categories"]
        
        stats = {
            "total_categories": len(category_list),
            "categories_with_subcategories": sum(1 for cat in category_list if cat.get('subcategories')),
            "total_subcategories": sum(len(cat.get('subcategories', [])) for cat in category_list),
            "avg_keywords_per_category": np.mean([len(cat.get('keywords', [])) for cat in category_list]),
            "avg_examples_per_category": np.mean([len(cat.get('examples', [])) for cat in category_list]),
            "categories_by_complexity": {}
        }
        
        # Analyze complexity
        for cat in category_list:
            complexity = len(cat.get('subcategories', []))
            complexity_level = "simple" if complexity == 0 else "complex" if complexity > 2 else "medium"
            stats["categories_by_complexity"][complexity_level] = stats["categories_by_complexity"].get(complexity_level, 0) + 1
        
        return stats


# Utility functions for category operations
def merge_categories(categories1: Dict[str, Any], categories2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two category dictionaries.
    
    Args:
        categories1: First categories dict
        categories2: Second categories dict
        
    Returns:
        Merged categories dict
    """
    # Implementation for merging categories
    # This is a placeholder - would need sophisticated logic to handle conflicts
    pass


def validate_categories_schema(categories: Dict[str, Any]) -> bool:
    """
    Validate categories dictionary against expected schema.
    
    Args:
        categories: Categories to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["version", "generated_at", "discovery_stats", "categories", "metadata"]
    
    if not all(key in categories for key in required_keys):
        return False
    
    # Validate categories structure
    for cat in categories.get("categories", []):
        if not all(key in cat for key in ["id", "technical_name", "display_name", "description"]):
            return False
    
    return True