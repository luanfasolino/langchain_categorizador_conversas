# PRD - Opção D: Pipeline Map-Reduce Hierárquico

## 1. VISÃO GERAL

### 1.1 Objetivo

Implementar um pipeline de categorização hierárquico (map → reduce → classify) que processa tickets de suporte com máxima qualidade e consistência nas categorias.

### 1.2 Benefícios Principais

- **Categorias 100% consistentes** através de dicionário canônico
- **Processamento escalável** com chunks otimizados
- **Contexto preservado** com overlap inteligente
- **Qualidade máxima** através de múltiplas fases de processamento

### 1.3 Métricas de Sucesso

- Zero categorias duplicadas/similares
- Precisão de categorização > 95%
- Custo por ticket < $0.15
- Tempo de processamento < 3 horas para 1M registros

## 2. ARQUITETURA TÉCNICA

### 2.1 Fluxo de Dados

```
CSV Input → Chunker → Map (LLM) → Reduce (LLM) → Classify (LLM) → Normalize → Output
                ↓                                        ↑
            categories.yaml ←────────────────────────────┘
```

### 2.2 Componentes

#### 2.2.1 Chunker Inteligente

```python
def split_by_tokens(text: str, max_tokens: int = 512, overlap: int = 128):
    """
    - Divide texto preservando frases completas
    - Overlap de 25% (128/512)
    - Usa mesmo tokenizer do modelo
    """
```

#### 2.2.2 Map-Summary (LLM Call 1)

```python
# Para cada chunk
prompt = """
Em EXATAMENTE 1 frase (máximo 25 palavras), resuma o assunto principal
tratado neste trecho de conversa de suporte.
"""
# max_output_tokens = 32
```

#### 2.2.3 Reduce-Summary (LLM Call 2)

```python
# Agrupa summaries por ticket
prompt = """
Combine as frases abaixo em um único parágrafo conciso de no máximo
50 palavras, sem repetir ideias, mantendo apenas informações essenciais.
"""
```

#### 2.2.4 Classify (LLM Call 3)

```python
# Com categories.yaml predefinido
prompt = """
Você é um classificador preciso. Use APENAS estas categorias:

{categories_yaml_content}

Para o ticket abaixo, retorne JSON:
{
  "categoria": "<canonical_name>",
  "confidence": <0-1>,
  "motivo": "<breve justificativa>"
}

Ticket: {ticket_summary}
"""
# temperature=0, top_p=0.1
```

### 2.3 Estrutura categories.yaml

```yaml
categories:
  - canonical_name: "problemas_pagamento"
    display_name: "Problemas de Pagamento"
    description: "Falhas em transações, cartão recusado, estorno"
    synonyms:
      - "pagamento com problema"
      - "erro no pagamento"
      - "problema com cartão"
    keywords: ["pagar", "cartão", "estorno", "cobrança", "transação"]

  - canonical_name: "alteracao_reserva"
    display_name: "Alteração de Reserva"
    description: "Mudança de datas, horários, passageiros"
    synonyms:
      - "mudar reserva"
      - "trocar data"
      - "remarcar"
    keywords: ["alterar", "mudar", "trocar", "remarcar", "data"]
```

## 3. IMPLEMENTAÇÃO DETALHADA

### 3.1 Fase 0: Geração Automática de Categorias

```python
def generate_categories(sample_tickets: list, output_path: str):
    """
    1. Processa amostra de 5% dos tickets
    2. Extrai até 50 categorias potenciais
    3. Agrupa similares
    4. Gera categories.yaml inicial
    """
    prompt = """
    Analise estes tickets e liste até 30 categorias principais que
    descrevem os problemas. Para cada categoria, forneça:
    - Nome canônico (snake_case)
    - Nome de exibição
    - Descrição breve
    - 3-5 sinônimos
    - 5-10 palavras-chave

    Formato: YAML
    """
```

### 3.2 Classe Principal

```python
class MapReduceCategorizer(BaseProcessor):
    def __init__(self, api_key: str, categories_path: str):
        self.categories = self.load_categories(categories_path)
        self.chunk_size = 512
        self.overlap = 128

    def process_pipeline(self, tickets: list):
        # 1. Map phase
        summaries = self.map_summarize(tickets)

        # 2. Reduce phase
        consolidated = self.reduce_summaries(summaries)

        # 3. Classify phase
        results = self.classify_tickets(consolidated)

        # 4. Normalize phase
        normalized = self.normalize_results(results)

        return normalized
```

### 3.3 Otimizações de Performance

#### 3.3.1 Processamento Paralelo

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class ParallelProcessor:
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers)
        self.rate_limiter = RateLimiter(calls_per_second=50)
```

#### 3.3.2 Cache Inteligente

```python
class SmartCache:
    def __init__(self):
        self.summary_cache = {}  # ticket_id -> summary
        self.category_cache = {}  # summary_hash -> category
```

### 3.4 Tratamento de Erros

```python
@retry(max_attempts=3, backoff_factor=2)
def call_llm_with_retry(prompt: str, **kwargs):
    try:
        response = llm.invoke(prompt, **kwargs)
        return response
    except RateLimitError:
        time.sleep(60)
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise
```

## 4. ESTIMATIVAS

### 4.1 Custo Detalhado (41.7K tickets válidos)

```
Map Phase:
- Input: 41.7K × 500 tokens = 20.8M tokens
- Output: 41.7K × 25 tokens = 1M tokens

Reduce Phase:
- Input: 41.7K × 100 tokens = 4.2M tokens
- Output: 41.7K × 50 tokens = 2.1M tokens

Classify Phase:
- Input: 41.7K × 80 tokens = 3.3M tokens
- Output: 41.7K × 20 tokens = 0.8M tokens

Total: ~$6.50 com Gemini 2.5 Flash
```

### 4.2 Tempo de Execução

- Geração de categorias: 30 minutos
- Map phase: 40 minutos
- Reduce phase: 20 minutos
- Classify phase: 30 minutos
- **Total: ~2 horas**

### 4.3 Recursos Necessários

- CPU: 4+ cores
- RAM: 8GB mínimo
- Disco: 2GB para cache
- API Rate Limit: 50 req/s

## 5. PLANO DE IMPLEMENTAÇÃO

### Dia 1: Infraestrutura Base

- [ ] Criar chunker com preservação de contexto
- [ ] Implementar gerador automático de categories.yaml
- [ ] Estruturar classes base do pipeline
- [ ] Setup de logging e monitoramento

### Dia 2: Pipeline Core

- [ ] Implementar Map phase com paralelização
- [ ] Implementar Reduce phase com compressão iterativa
- [ ] Implementar Classify phase com validação
- [ ] Adicionar sistema de cache

### Dia 3: Refinamentos e Testes

- [ ] Implementar normalização pós-processamento
- [ ] Adicionar retry logic e error handling
- [ ] Criar suite de testes
- [ ] Documentação e exemplos

## 6. RISCOS E MITIGAÇÕES

| Risco                     | Probabilidade | Impacto | Mitigação                        |
| ------------------------- | ------------- | ------- | -------------------------------- |
| Rate limits da API        | Alta          | Alto    | Implementar backoff exponencial  |
| Categorias inconsistentes | Média         | Alto    | Validação cruzada com embeddings |
| Memória insuficiente      | Baixa         | Médio   | Processamento em batches menores |
| Perda de contexto         | Média         | Alto    | Aumentar overlap para 30%        |

## 7. CRITÉRIOS DE ACEITAÇÃO

1. **Funcionalidade**

   - Pipeline processa 100% dos tickets válidos
   - Categorias geradas são mutuamente exclusivas
   - Confiança média > 0.85

2. **Performance**

   - Processamento < 3h para dataset completo
   - Uso de memória < 8GB
   - Custo < $0.20 por 1000 tickets

3. **Qualidade**
   - Zero categorias duplicadas
   - Cobertura de 95%+ dos casos
   - Logs detalhados de todo processo

## 8. EXTENSÕES FUTURAS

1. **Active Learning**

   - Interface para revisão humana
   - Auto-atualização de categories.yaml
   - Métricas de drift

2. **Embeddings para Validação**

   - Usar text-embedding-3-small
   - Clustering para descobrir novas categorias
   - Detecção de outliers

3. **Dashboard Analytics**
   - Visualização de categorias mais comuns
   - Tendências temporais
   - Alertas de anomalias
