# PRD - Opção D: Pipeline Map-Reduce Hierárquico (v2.0)

## 1. VISÃO GERAL

### 1.1 Objetivo

Implementar um pipeline de categorização hierárquico (map → reduce → classify) que processa tickets de suporte com máxima qualidade e consistência nas categorias.

### 1.2 Benefícios Principais

- **Categorias consistentes** através de dicionário canônico versionado
- **Processamento escalável** com chunks dinâmicos
- **Configuração externa** sem alterar código
- **Observabilidade completa** de custos e performance

### 1.3 Métricas de Sucesso

- **F1-Score ≥ 0.85** em amostra de validação (500 tickets)
- **Categorias duplicadas ≤ 0.5%** (medido por similaridade)
- **Custo por 1000 tickets < $0.20**
- **Tempo de processamento < 1h** para 19.251 registros
- **Taxa de erro < 1%** no pipeline

## 2. ARQUITETURA TÉCNICA

### 2.1 Configuração Externa (config.yaml)

```yaml
# config.yaml - Todas as configurações editáveis
version: "1.0"
model:
  provider: "gemini"
  name: "gemini-2.5-flash" # ou gemini-2.5-flash-lite
  temperature: 0.1

chunking:
  strategy: "dynamic" # dynamic | fixed
  base_size: 512
  max_size: 1024
  overlap_percent: 0.25
  preserve_sentences: true

rate_limits:
  requests_per_minute: 50
  concurrent_workers: 10
  retry_attempts: 3
  backoff_factor: 2.0

prompts:
  map_max_tokens: 48
  reduce_max_tokens: 80
  classify_max_tokens: 32

validation:
  sample_size: 500
  confidence_threshold: 0.85
  max_categories_per_ticket: 3

monitoring:
  log_level: "INFO"
  track_costs: true
  alert_threshold_usd: 10.00
```

### 2.2 Componentes Melhorados

#### 2.2.1 Chunker Dinâmico

```python
class DynamicChunker:
    def __init__(self, config: dict):
        self.config = config
        self.stats_cache = {}

    def calculate_optimal_chunk_size(self, texts: list) -> int:
        """Calcula tamanho ótimo baseado em P90 dos textos"""
        if self.config['chunking']['strategy'] == 'fixed':
            return self.config['chunking']['base_size']

        # Calcula percentil 90 do comprimento
        lengths = [len(text) // 4 for text in texts]  # ~4 chars/token
        p90 = np.percentile(lengths, 90)

        # Chunk size = P90 * 1.2, limitado ao máximo
        optimal = int(p90 * 1.2)
        return min(optimal, self.config['chunking']['max_size'])

    def split_with_overlap(self, text: str, chunk_size: int) -> list:
        """Divide preservando sentenças completas"""
        overlap = int(chunk_size * self.config['chunking']['overlap_percent'])
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sent_size = self.estimate_tokens(sentence)

            if current_size + sent_size > chunk_size and current_chunk:
                # Salva chunk atual
                chunks.append(' '.join(current_chunk))

                # Inicia novo chunk com overlap
                overlap_sentences = []
                overlap_size = 0

                for s in reversed(current_chunk):
                    s_size = self.estimate_tokens(s)
                    if overlap_size + s_size <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sent_size
            else:
                current_chunk.append(sentence)
                current_size += sent_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
```

#### 2.2.2 Map-Summary com Limite Estrito

```python
map_prompt = PromptTemplate(
    template="""
Resuma em UMA frase (máximo 40 palavras) o problema principal neste trecho.
Seja específico sobre o tipo de problema relatado.

Trecho: {text}

Resumo:""",
    max_output_tokens=48  # Configurável via YAML
)
```

#### 2.2.3 Sistema de Rate Limiting Global

```python
class GlobalRateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.calls = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()

            # Remove calls older than 1 minute
            while self.calls and self.calls[0] < now - 60:
                self.calls.popleft()

            # Wait if at limit
            if len(self.calls) >= self.rpm:
                sleep_time = 60 - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
                return await self.acquire()

            self.calls.append(now)
```

### 2.3 Tratamento de Erros Específicos

```python
class LLMErrorHandler:
    def __init__(self, config: dict):
        self.config = config
        self.error_counts = defaultdict(int)

    async def handle_api_call(self, func, *args, **kwargs):
        for attempt in range(self.config['retry_attempts']):
            try:
                return await func(*args, **kwargs)

            except OutputTooLongError as e:
                # Reduz tokens de output em 20%
                if 'max_output_tokens' in kwargs:
                    kwargs['max_output_tokens'] = int(kwargs['max_output_tokens'] * 0.8)
                    logger.warning(f"Reduzindo output tokens para {kwargs['max_output_tokens']}")

            except RateLimitError as e:
                # Backoff exponencial específico para rate limit
                wait_time = (2 ** attempt) * 60  # 1min, 2min, 4min
                logger.warning(f"Rate limit atingido, esperando {wait_time}s")
                await asyncio.sleep(wait_time)

            except InvalidRequestError as e:
                # Log detalhado e não retry (erro de entrada)
                logger.error(f"Request inválido: {e}")
                self.error_counts['invalid_request'] += 1
                raise

            except TimeoutError as e:
                # Aumenta timeout progressivamente
                if 'timeout' in kwargs:
                    kwargs['timeout'] *= 1.5

            except Exception as e:
                logger.error(f"Erro inesperado tentativa {attempt + 1}: {e}")

        raise MaxRetriesExceeded(f"Falhou após {self.config['retry_attempts']} tentativas")
```

### 2.4 Clustering de Categorias (Opcional)

```python
class CategoryClusterer:
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model

    def cluster_and_merge(self, categories: list) -> dict:
        """Agrupa categorias similares usando embeddings"""
        if len(categories) < 20:  # Não vale a pena para poucas categorias
            return {cat: cat for cat in categories}

        # Gera embeddings
        embeddings = self.get_embeddings(categories)

        # HDBSCAN para clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
        clusters = clusterer.fit_predict(embeddings)

        # Merge categorias no mesmo cluster
        category_map = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise points
                continue

            cluster_categories = [cat for cat, c in zip(categories, clusters) if c == cluster_id]
            # Escolhe o mais curto como canônico
            canonical = min(cluster_categories, key=len)

            for cat in cluster_categories:
                category_map[cat] = canonical

        return category_map
```

## 3. MONITORAMENTO E CUSTOS

### 3.1 Token Meter

```python
class TokenMeter:
    def __init__(self, alert_threshold_usd: float):
        self.alert_threshold = alert_threshold_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.costs_by_phase = defaultdict(float)

    def track_call(self, phase: str, input_tokens: int, output_tokens: int, model: str):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Preços por modelo
        prices = {
            "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40}
        }

        cost = (input_tokens * prices[model]["input"] +
                output_tokens * prices[model]["output"]) / 1_000_000

        self.costs_by_phase[phase] += cost

        # Alerta se passar do threshold
        total_cost = sum(self.costs_by_phase.values())
        if total_cost > self.alert_threshold:
            self.send_alert(total_cost)

    def get_report(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cost_by_phase": dict(self.costs_by_phase),
            "total_cost_usd": sum(self.costs_by_phase.values()),
            "cost_per_1000_tickets": (sum(self.costs_by_phase.values()) / self.tickets_processed) * 1000
        }
```

### 3.2 Métricas de Qualidade

```python
def calculate_f1_score(predictions: list, ground_truth: list) -> float:
    """Calcula F1-Score macro para validação"""
    # Converte para formato sklearn
    y_true = []
    y_pred = []

    for gt, pred in zip(ground_truth, predictions):
        # Multi-label encoding
        for category in gt['categories']:
            y_true.append(category)
            y_pred.append(pred['categories'][0] if pred['categories'] else 'unknown')

    return f1_score(y_true, y_pred, average='macro')

def validate_categories(categories: list) -> dict:
    """Valida duplicatas e consistência"""
    # Detecta duplicatas por similaridade
    duplicates = []
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories[i+1:], i+1):
            similarity = SequenceMatcher(None, cat1.lower(), cat2.lower()).ratio()
            if similarity > 0.85:
                duplicates.append((cat1, cat2, similarity))

    duplicate_rate = len(duplicates) / len(categories) if categories else 0

    return {
        "total_categories": len(categories),
        "unique_categories": len(set(c.lower() for c in categories)),
        "duplicate_pairs": duplicates,
        "duplicate_rate": duplicate_rate,
        "passes_threshold": duplicate_rate <= 0.005  # 0.5%
    }
```

## 4. ESTIMATIVAS ATUALIZADAS

### 4.1 Custo por SKU (19.251 tickets válidos)

| Modelo         | Input (M) | Output (M) | Custo Total | Por 1K tickets |
| -------------- | --------- | ---------- | ----------- | -------------- |
| **Flash**      | 13.1      | 2.3        | **$3.32**   | **$0.173**     |
| **Flash-Lite** | 13.1      | 2.3        | **$2.21**   | **$0.115**     |

### 4.2 Projeções de Escala

```
20K tickets: $3.46 - $2.30
50K tickets: $8.65 - $5.75
100K tickets: $17.30 - $11.50
500K tickets: $86.50 - $57.50
```

## 5. PLANO DE IMPLEMENTAÇÃO ATUALIZADO

### Dia 0: Preparação

- [ ] Solicitar aumento de quota da API (se necessário)
- [ ] Configurar ambiente e dependências
- [ ] Criar config.yaml inicial

### Dia 1: Core + Config

- [ ] Implementar DynamicChunker com config externa
- [ ] Criar GlobalRateLimiter assíncrono
- [ ] Desenvolver LLMErrorHandler completo
- [ ] Setup TokenMeter e alertas

### Dia 2: Pipeline + Validação

- [ ] Implementar pipeline map-reduce com async
- [ ] Adicionar validação F1-Score
- [ ] Criar detector de duplicatas
- [ ] Smoke test com 100 tickets

### Dia 3: Polish + Produção

- [ ] Dashboard de monitoramento (Grafana)
- [ ] Documentação de operação
- [ ] Testes de carga (1K tickets)
- [ ] Deploy e validação final

## 6. NOVOS RISCOS E MITIGAÇÕES

| Risco                 | Probabilidade | Impacto | Mitigação                             |
| --------------------- | ------------- | ------- | ------------------------------------- |
| Drift semântico       | Média         | Alto    | Re-treino mensal das categorias       |
| Custos inesperados    | Média         | Alto    | Token meter + alertas + limite diário |
| Mudança de preços API | Baixa         | Alto    | Config externa permite troca rápida   |
| Categorias emergentes | Alta          | Médio   | Active learning quinzenal             |

## 7. CRITÉRIOS DE ACEITAÇÃO REVISADOS

### Funcionalidade

- [ ] F1-Score ≥ 0.85 em amostra de 500 tickets
- [ ] Taxa de duplicatas ≤ 0.5%
- [ ] Processamento bem-sucedido de 99%+ dos tickets

### Configurabilidade

- [ ] Trocar dataset = alterar apenas config.yaml
- [ ] Ajustar chunks/overlap sem tocar código
- [ ] Mudar modelo (Flash/Lite) via config

### Observabilidade

- [ ] Dashboard mostra custos em tempo real
- [ ] Logs estruturados com trace_id
- [ ] Alertas automáticos de threshold

### Performance

- [ ] < 1h para 20K registros
- [ ] < 8GB RAM em pico
- [ ] Custo < $0.20 por 1K tickets

## 8. VERSIONAMENTO DE CATEGORIAS

```yaml
# categories_v2.yaml
version: "2.0"
generated_at: "2024-01-15T10:00:00Z"
parent_version: "1.0"
changes:
  added: ["categoria_nova_1", "categoria_nova_2"]
  merged: { "categoria_old": "categoria_canonical" }
  removed: ["categoria_obsoleta"]

stats:
  total_categories: 22
  samples_analyzed: 5000
  f1_score: 0.87

categories:
  # ... lista de categorias
```

## 9. CONCLUSÃO

Esta versão 2.0 do PRD incorpora:

- ✅ Métricas objetivas (F1-Score)
- ✅ Configuração 100% externa
- ✅ Tratamento específico de erros
- ✅ Monitoramento de custos em tempo real
- ✅ Versionamento de categorias
- ✅ Critérios de aceitação testáveis

O sistema fica mais robusto, observável e pronto para produção.
