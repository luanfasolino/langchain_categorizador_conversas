# PRD - Opção E: Pipeline Descoberta + Aplicação

## 1. VISÃO GERAL

### 1.1 Objetivo

Implementar um sistema de categorização em duas fases distintas: primeiro DESCOBRE categorias com amostragem inteligente, depois APLICA essas categorias consistentemente em todo o dataset.

### 1.2 Benefícios Principais

- **Custo 60% menor** que abordagem tradicional
- **Categorias 100% consistentes** sem fragmentação
- **Descoberta automática** sem conhecimento prévio do negócio
- **Flexibilidade** para ajustar categorias entre fases

### 1.3 Métricas de Sucesso

- Descoberta de 95%+ das categorias relevantes
- Zero inconsistências na aplicação
- Custo total < $1.50 para 20K registros
- Tempo total < 30 minutos

## 2. ARQUITETURA TÉCNICA

### 2.1 Visão Geral do Fluxo

```
FASE 1: DESCOBERTA (10-20% dos dados)
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│   Amostra   │ -> │   Análise   │ -> │  Descoberta  │ -> │ categories.json│
│ Inteligente │    │   Profunda  │    │  Categorias  │    │   (output)     │
└─────────────┘    └─────────────┘    └──────────────┘    └────────────────┘
                                                                    │
FASE 2: APLICAÇÃO (100% dos dados)                                │
┌─────────────┐    ┌─────────────┐    ┌──────────────┐           │
│  Todos os   │ -> │ Classificação│ <- │  Categorias  │ <─────────┘
│   Tickets   │    │   Rápida    │    │    Fixas     │
└─────────────┘    └─────────────┘    └──────────────┘
                           │
                           ↓
                   ┌──────────────┐
                   │ Resultado    │
                   │ Final (.csv) │
                   └──────────────┘
```

### 2.2 Fase 1: Descoberta de Categorias

#### 2.2.1 Amostragem Inteligente

```python
class IntelligentSampler:
    def __init__(self, strategy: str = "stratified"):
        self.strategy = strategy

    def sample_tickets(self, df: pd.DataFrame, sample_size: float = 0.15):
        """
        Estratégias de amostragem:
        - stratified: Mantém proporção temporal
        - diversity: Maximiza diversidade textual
        - hybrid: Combina temporal + diversidade
        """
        if self.strategy == "stratified":
            # Amostra proporcional por mês
            return df.groupby(pd.Grouper(key='ticket_created_at', freq='M'))\
                     .apply(lambda x: x.sample(frac=sample_size))

        elif self.strategy == "diversity":
            # Usa TF-IDF para selecionar tickets diversos
            vectorizer = TfidfVectorizer(max_features=1000)
            vectors = vectorizer.fit_transform(df['text'])

            # K-means para encontrar clusters
            kmeans = KMeans(n_clusters=int(len(df) * sample_size))
            clusters = kmeans.fit_predict(vectors)

            # Pega um representante de cada cluster
            sampled_indices = []
            for cluster_id in range(kmeans.n_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                # Pega o mais próximo ao centroide
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(vectors[cluster_indices] - center, axis=1)
                sampled_indices.append(cluster_indices[np.argmin(distances)])

            return df.iloc[sampled_indices]
```

#### 2.2.2 Análise Profunda com Contexto Completo

```python
class DeepAnalyzer:
    def __init__(self, llm, overlap_percent: float = 0.30):
        self.llm = llm
        self.chunk_size = 800000  # 800K tokens
        self.overlap = int(chunk_size * overlap_percent)

    def analyze_for_categories(self, tickets: list):
        """
        Análise em 3 níveis para máxima descoberta
        """
        # Nível 1: Análise individual
        individual_insights = self.analyze_individual(tickets)

        # Nível 2: Análise por grupos
        group_insights = self.analyze_groups(tickets)

        # Nível 3: Análise global
        global_insights = self.analyze_global(individual_insights, group_insights)

        return self.extract_categories(global_insights)
```

#### 2.2.3 Template de Descoberta

```python
discovery_template = """
Você é um especialista em análise de atendimento ao cliente.

Analise as conversas abaixo e:

1. IDENTIFIQUE todos os tipos de problemas/solicitações
2. AGRUPE problemas similares
3. CRIE categorias hierárquicas quando apropriado
4. SUGIRA nomenclatura padronizada

Para cada categoria encontrada, forneça:
- Nome técnico (snake_case)
- Nome de exibição amigável
- Descrição clara (1 frase)
- Palavras-chave associadas
- Exemplos de tickets

IMPORTANTE:
- Seja exaustivo - é melhor ter categorias demais que de menos
- Mantenha granularidade apropriada (nem muito genérico, nem muito específico)
- Considere categorias compostas quando um ticket tem múltiplos problemas

CONVERSAS:
{sample_tickets}

Retorne um JSON estruturado com as categorias descobertas.
"""
```

### 2.3 Fase 2: Aplicação Eficiente

#### 2.3.1 Classificador Otimizado

```python
class FastClassifier:
    def __init__(self, categories_path: str):
        self.categories = self.load_categories(categories_path)
        self.prompt_template = self.build_optimized_prompt()

    def build_optimized_prompt(self):
        """
        Prompt minimalista para classificação rápida
        """
        cat_list = "\n".join([
            f"{i+1}. {cat['display_name']}: {cat['description']}"
            for i, cat in enumerate(self.categories)
        ])

        return f"""
Categorias disponíveis:
{cat_list}

Ticket: {{ticket_text}}

Retorne APENAS o número da categoria principal (1-{len(self.categories)}).
Se múltiplas categorias se aplicam, retorne os números separados por vírgula (ex: 1,3,5).
Resposta:"""
```

#### 2.3.2 Processamento em Batch Ultra-Eficiente

```python
class BatchProcessor:
    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size

    def process_with_categories(self, tickets: list, categories: dict):
        """
        Processa tickets em batches grandes com prompt otimizado
        """
        # Cria prompt batch que classifica múltiplos tickets de uma vez
        batch_prompt = """
Classifique os tickets abaixo usando APENAS os números das categorias:

{categories_list}

TICKETS:
{tickets_batch}

Retorne JSON: {"ticket_id": [categoria_ids], ...}
"""
```

### 2.4 Estrutura do categories.json

```json
{
  "version": "1.0",
  "generated_at": "2024-01-10T10:00:00Z",
  "stats": {
    "total_tickets_analyzed": 5000,
    "total_categories": 18,
    "confidence_threshold": 0.85
  },
  "categories": [
    {
      "id": 1,
      "technical_name": "payment_issues",
      "display_name": "Problemas de Pagamento",
      "description": "Falhas em transações, cartões recusados, cobranças indevidas",
      "keywords": ["pagamento", "cartão", "cobrança", "transação", "recusado"],
      "examples": ["Meu cartão foi recusado", "Cobrança duplicada"],
      "frequency": 0.23,
      "subcategories": [
        {
          "id": 11,
          "name": "cartao_recusado",
          "display": "Cartão Recusado"
        },
        {
          "id": 12,
          "name": "cobranca_duplicada",
          "display": "Cobrança Duplicada"
        }
      ]
    },
    {
      "id": 2,
      "technical_name": "booking_changes",
      "display_name": "Alterações de Reserva",
      "description": "Mudanças de data, horário, passageiros ou cancelamentos",
      "keywords": ["alterar", "mudar", "remarcar", "cancelar", "trocar"],
      "examples": ["Preciso mudar a data", "Quero cancelar minha reserva"],
      "frequency": 0.18
    }
  ],
  "metadata": {
    "llm_model": "gemini-2.5-flash",
    "sampling_strategy": "stratified",
    "sample_size": 0.15,
    "overlap_percentage": 0.3
  }
}
```

## 3. IMPLEMENTAÇÃO DETALHADA

### 3.1 CLI Interface

```bash
# Fase 1: Descoberta
python main.py --mode discover \
    --sample-rate 0.15 \
    --sampling-strategy hybrid \
    --output categories.json \
    --min-confidence 0.85

# Fase 2: Aplicação
python main.py --mode apply \
    --categories categories.json \
    --batch-size 500 \
    --workers 8

# Modo completo (executa ambas as fases)
python main.py --mode smart-categorize \
    --sample-rate 0.15 \
    --auto-apply true
```

### 3.2 Classe Principal

```python
class SmartCategorizer:
    def __init__(self, api_key: str, database_dir: Path):
        self.api_key = api_key
        self.database_dir = database_dir
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Baixa para consistência
            google_api_key=api_key
        )

    def discover_categories(self,
                          input_file: Path,
                          sample_rate: float = 0.15,
                          strategy: str = "hybrid") -> Path:
        """
        Fase 1: Descobre categorias automaticamente
        """
        # 1. Carrega e prepara dados
        tickets = self.prepare_data(input_file)

        # 2. Amostragem inteligente
        sampler = IntelligentSampler(strategy)
        sample = sampler.sample_tickets(tickets, sample_rate)

        # 3. Análise profunda
        analyzer = DeepAnalyzer(self.llm)
        categories = analyzer.analyze_for_categories(sample)

        # 4. Salva categorias
        output_path = self.database_dir / "categories.json"
        self.save_categories(categories, output_path)

        return output_path

    def apply_categories(self,
                        input_file: Path,
                        categories_file: Path,
                        batch_size: int = 500) -> Path:
        """
        Fase 2: Aplica categorias em todo dataset
        """
        # 1. Carrega categorias
        categories = self.load_categories(categories_file)

        # 2. Prepara classificador otimizado
        classifier = FastClassifier(categories)

        # 3. Processa em batches
        processor = BatchProcessor(batch_size)
        results = processor.process_all(input_file, classifier)

        # 4. Salva resultados
        output_path = self.database_dir / "categorized_tickets_final.csv"
        self.save_results(results, output_path)

        return output_path
```

### 3.3 Otimizações Específicas

#### 3.3.1 Cache Entre Fases

```python
class PhaseCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.discovery_cache = cache_dir / "discovery_cache.pkl"
        self.ticket_embeddings = cache_dir / "embeddings.npy"

    def cache_discovery_insights(self, insights: dict):
        """Salva insights intermediários para reuso"""
        with open(self.discovery_cache, 'wb') as f:
            pickle.dump(insights, f)
```

#### 3.3.2 Validação de Cobertura

```python
def validate_category_coverage(categories: list, validation_sample: list):
    """
    Valida se as categorias cobrem adequadamente os casos
    """
    uncategorized = []
    low_confidence = []

    for ticket in validation_sample:
        result = classify_single(ticket, categories)
        if result['confidence'] < 0.7:
            low_confidence.append(ticket)
        if result['category'] == 'unknown':
            uncategorized.append(ticket)

    coverage = 1 - (len(uncategorized) / len(validation_sample))
    avg_confidence = np.mean([r['confidence'] for r in results])

    return {
        'coverage': coverage,
        'avg_confidence': avg_confidence,
        'uncategorized_count': len(uncategorized),
        'low_confidence_count': len(low_confidence)
    }
```

## 4. ESTIMATIVAS DETALHADAS

### 4.1 Custos (19.251 tickets válidos)

#### Fase 1: Descoberta (15% = 2.9K tickets)

```
Análise profunda:
- Input: 2.9K × 500 tokens = 1.45M tokens
- Output: 2.9K × 100 tokens = 0.29M tokens
- Custo: ~$0.37

Geração de categorias:
- Input: 0.29M tokens (summaries)
- Output: 10K tokens (categories)
- Custo: ~$0.05

Subtotal Fase 1: ~$0.42
```

#### Fase 2: Aplicação (100% = 19.251 tickets)

```
Classificação rápida:
- Input: 19.3K × 150 tokens = 2.9M tokens
- Output: 19.3K × 5 tokens = 0.1M tokens
- Custo: ~$0.50

Total Geral: ~$0.92
```

### 4.2 Tempo de Execução

```
Fase 1:
- Amostragem: 1 minuto
- Análise: 8 minutos
- Geração categorias: 2 minutos
Subtotal: 11 minutos

Fase 2:
- Carregamento: 1 minuto
- Classificação: 12 minutos
- Salvamento: 1 minuto
Subtotal: 14 minutos

TOTAL: ~25 minutos
```

### 4.3 Comparação com Outras Abordagens

| Métrica      | Opção E | Opção D | Atual  |
| ------------ | ------- | ------- | ------ |
| Custo        | $0.92   | $3.32   | $2.31  |
| Tempo        | 25 min  | 60 min  | 65 min |
| Consistência | 100%    | 100%    | 70%    |
| Complexidade | Média   | Alta    | Baixa  |

## 5. PLANO DE IMPLEMENTAÇÃO

### Dia 1: Implementação Core

#### Manhã (4h)

- [ ] Implementar IntelligentSampler com 3 estratégias
- [ ] Criar DeepAnalyzer com análise multi-nível
- [ ] Desenvolver gerador de categories.json

#### Tarde (4h)

- [ ] Implementar FastClassifier otimizado
- [ ] Criar BatchProcessor com paralelização
- [ ] Integrar ambas as fases

### Dia 2: Refinamentos (se necessário)

- [ ] Adicionar validação de cobertura
- [ ] Implementar cache entre fases
- [ ] Criar relatórios de qualidade
- [ ] Testes end-to-end

## 6. VANTAGENS ÚNICAS DESTA ABORDAGEM

### 6.1 Descoberta Automática Inteligente

- Não requer conhecimento do negócio
- Encontra categorias emergentes
- Adapta-se a diferentes domínios

### 6.2 Economia Significativa

- 60% mais barato que chunking tradicional
- 70% mais barato que map-reduce completo
- ROI positivo em primeira execução

### 6.3 Qualidade Superior

- Zero fragmentação de categorias
- Nomenclatura 100% consistente
- Possibilidade de revisão entre fases

### 6.4 Flexibilidade

- Pode rodar fases independentemente
- Categories.json reutilizável
- Fácil adicionar novas categorias

## 7. CASOS DE USO AVANÇADOS

### 7.1 Análise Temporal

```python
# Descobrir categorias por período
python main.py --mode discover \
    --date-from 2024-01 \
    --date-to 2024-03 \
    --output categories_q1.json
```

### 7.2 Merge de Categorias

```python
# Combinar categorias de múltiplas descobertas
python main.py --mode merge-categories \
    --inputs "categories_q1.json,categories_q2.json" \
    --output categories_merged.json
```

### 7.3 Atualização Incremental

```python
# Descobrir novas categorias sem perder as antigas
python main.py --mode discover \
    --existing-categories categories.json \
    --update-mode incremental
```

## 8. MONITORAMENTO E MÉTRICAS

### 8.1 Métricas de Qualidade

```python
{
  "discovery_metrics": {
    "categories_found": 18,
    "avg_confidence": 0.92,
    "coverage": 0.97,
    "processing_time": "18m32s"
  },
  "application_metrics": {
    "tickets_processed": 41700,
    "avg_classification_time": "35ms",
    "cache_hit_rate": 0.23,
    "errors": 0
  }
}
```

### 8.2 Alertas Automáticos

- Coverage < 90%: Reexecutar descoberta com amostra maior
- Confidence < 0.80: Revisar categorias manualmente
- Novas categorias emergentes: Sugerir atualização

## 9. CONCLUSÃO

A Opção E oferece o melhor equilíbrio entre:

- **Qualidade**: Categorias 100% consistentes
- **Custo**: 60% mais barato
- **Simplicidade**: Duas fases claras e distintas
- **Flexibilidade**: Adaptável a qualquer domínio

Ideal para cenários onde:

1. Não há conhecimento prévio do domínio
2. Consistência é crítica
3. Orçamento é limitado
4. Precisa de resultados rápidos
