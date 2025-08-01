# 🎯 Categorizador de Conversas com IA

Sistema inteligente de categorização e análise de tickets de suporte ao cliente usando **LangChain** e **Google Gemini AI**. O sistema processa conversas em larga escala e gera categorias precisas seguindo uma arquitetura **Map-Reduce-Classify**.

## 🌟 Funcionalidades

- ✅ **Categorização Inteligente**: Classifica automaticamente tickets de suporte
- ✅ **Análise de Padrões**: Identifica problemas recorrentes e tendências
- ✅ **Resumos Executivos**: Gera bullet points com sugestões de melhoria
- ✅ **Cache Inteligente**: Sistema de cache para otimizar performance e custos
- ✅ **Processamento Paralelo**: Execução otimizada com múltiplos workers
- ✅ **Relatórios Detalhados**: Análises completas com métricas e custos

## 🏗️ Como Funciona a Categorização

Nosso sistema segue uma arquitetura **Map-Reduce-Classify** inspirada em práticas de Big Data:

```
📊 DADOS BRUTOS
      ↓
🔄 FASE MAP (Resumir)
   • Divide conversas em chunks
   • Cada chunk é resumido em paralelo
   • Identifica problemas principais
      ↓
🔄 FASE REDUCE (Consolidar)
   • Combina todos os resumos
   • Cria visão unificada dos problemas
   • Identifica padrões globais
      ↓
🎯 FASE CLASSIFY (Categorizar)
   • Usa análise consolidada
   • Categoriza cada ticket individualmente
   • Gera categorias específicas e precisas
      ↓
📈 RESULTADOS FINAIS
```

### 💡 Por que essa Arquitetura?

1. **🎯 Precisão**: Categorização baseada em análise completa do dataset
2. **⚡ Performance**: Processamento paralelo otimizado
3. **💰 Eficiência**: Cache inteligente reduz custos de API
4. **🔍 Qualidade**: Visão consolidada gera categorias mais consistentes

## 🚀 Início Rápido

### 📋 Pré-requisitos

- Python 3.8+
- Conta Google Cloud com Gemini AI habilitado
- 4GB RAM mínimo (recomendado: 8GB)

### 🔧 Instalação

1. **Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd langchain_categorizador_conversas
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure sua API Key:**
```bash
# Crie um arquivo .env na raiz do projeto
echo "GOOGLE_API_KEY=sua_api_key_aqui" > .env
```

### 📊 Preparação dos Dados

Seu arquivo CSV deve ter estas colunas obrigatórias:
- `ticket_id`: ID único do ticket
- `category`: Tipo de mensagem (use 'TEXT' para conversas)
- `text`: Conteúdo da conversa
- `sender`: Quem enviou (USER, AGENT, HELPDESK_INTEGRATION, etc.)
- `ticket_created_at`: Data de criação

**Exemplo:**
```csv
ticket_id;category;text;sender;ticket_created_at
12345;TEXT;Olá, preciso de ajuda;USER;2024-01-15 10:30:00
12345;TEXT;Como posso ajudá-lo?;AGENT;2024-01-15 10:31:00
```

Coloque seu arquivo na pasta `database/` (será criada automaticamente).

## 🎮 Como Usar

### 🔥 Execução Completa (Recomendado)

Para processar um dataset completo:

```bash
python src/main.py --mode all --cache-control fresh
```

### 🎛️ Execução por Etapas

Se preferir executar etapa por etapa:

```bash
# 1. Apenas categorização
python src/main.py --mode categorize --cache-control fresh

# 2. Apenas resumos e bullet points
python src/main.py --mode summarize --cache-control fresh

# 3. Combinar resultados
python src/main.py --mode merge

# 4. Análise completa e relatórios
python src/main.py --mode analyze
```

### ⚙️ Opções Avançadas

```bash
# Processar apenas primeiros 1000 registros (para testes)
python src/main.py --mode all --nrows 1000

# Usar menos workers (se tiver problemas de rate limit)
python src/main.py --mode all --workers 2

# Especificar arquivo específico
python src/main.py --mode all --input-file "meu_arquivo.csv"

# Usar cache existente (mais rápido)
python src/main.py --mode all --cache-control continue
```

## 🎯 Controle de Cache

O sistema tem **cache inteligente** que acelera execuções subsequentes:

### 🚀 Fresh Start (Máxima Precisão)
```bash
--cache-control fresh
```
- ❌ Apaga cache antigo
- ✅ Processa tudo novamente  
- ✅ Gera novo cache
- **Use quando:** Mudou prompts, dados ou quer máxima precisão

### ⚡ Continue (Máxima Velocidade)  
```bash
--cache-control continue
```
- ✅ Usa cache existente
- ⚡ Processamento muito mais rápido
- **Use quando:** Dados e configurações não mudaram

### 🚫 Sem Cache
```bash
--no-cache
```
- ❌ Não usa cache existente
- ❌ Não salva novo cache
- **Use quando:** Quer sempre reprocessar tudo

## 📁 Estrutura de Arquivos Gerados

Após a execução, você encontrará:

```
database/
├── cache/                              # Cache inteligente (acelera próximas execuções)
└── analysis_reports/                   # 📁 Todos os arquivos de análise organizados aqui
    ├── categorized_tickets.csv         # Tickets categorizados
    ├── summarized_tickets.csv          # Bullet points de melhoria
    ├── summarized_tickets_resumo.txt   # Resumo executivo
    ├── final_analysis.csv              # Análise combinada
    ├── pipeline_analysis_YYYYMMDD_HHMMSS.xlsx     # Relatório Excel completo
    ├── pipeline_analysis_YYYYMMDD_HHMMSS_categories.csv    # Análise por categorias
    ├── pipeline_analysis_YYYYMMDD_HHMMSS_summary.csv       # Resumo estatístico
    └── pipeline_analysis_YYYYMMDD_HHMMSS_summary.txt       # Resumo executivo
```

## 📊 Interpretando os Resultados

### 🏷️ Categorized Tickets
Arquivo com todas as categorias atribuídas:
```csv
ticket_id;categoria
12345;Problema de Login
12345;Recuperação de Senha
12346;Dúvida sobre Cobrança
```

### 💡 Bullet Points
15 sugestões práticas para reduzir volume de contatos:
```csv
bullet
Implementar reset de senha automático via SMS
Criar FAQ sobre problemas de login mais comuns
Melhorar interface de recuperação de conta
```

### 📈 Relatórios de Análise
- **Categorias mais frequentes**
- **Tendências por período**
- **Oportunidades de melhoria**
- **ROI estimado das melhorias**

## ⚡ Performance e Custos

### 🎯 Configurações Recomendadas

| Dataset | Workers | Tempo Estimado | Custo Estimado |
|---------|---------|----------------|----------------|
| 1K tickets | 2 | 10-15 min | $0.20-0.50 |
| 10K tickets | 4 | 1-2 horas | $2.00-5.00 |
| 100K tickets | 4 | 8-12 horas | $20-50 |

### 💰 Otimização de Custos

1. **Use Cache**: Pode reduzir custos em 70-90%
2. **Workers Moderados**: 2-4 workers evitam rate limits
3. **Teste Pequeno**: Use `--nrows 100` para testar configurações

## 🔧 Solução de Problemas

### ❌ Problemas Comuns

**"Rate Limit Exceeded"**
```bash
# Solução: Reduza workers
python src/main.py --mode all --workers 2
```

**"Nenhum ticket válido encontrado"**
- Verifique se tem coluna `category='TEXT'`
- Certifique-se que há mensagens USER e AGENT suficientes
- Veja o relatório de filtragem no início da execução

**"JSON Parsing Error"**
- Normalmente resolvido automaticamente com retry
- Se persistir, reduza `--nrows` para testes

**Cache não funciona**
- Verifique permissões da pasta `database/cache/`
- Use `--cache-control fresh` para recriar cache

### 📞 Suporte

Se tiver problemas:

1. **Verifique os logs**: O sistema mostra logs detalhados
2. **Teste com poucos dados**: Use `--nrows 100` primeiro
3. **Verifique API Key**: Certifique-se que está no arquivo `.env`

## 🧠 Arquitetura Técnica

### 🔧 Componentes Principais

- **LangChain**: Framework para aplicações com LLM
- **Google Gemini 2.5 Flash**: Modelo de IA para categorização
- **Cache Manager**: Sistema inteligente de cache
- **Base Processor**: Processamento paralelo otimizado
- **Cost Tracker**: Monitoramento de custos em tempo real

### 🎯 Características Técnicas

- **Thread-safe**: Processamento paralelo seguro
- **Error Recovery**: Retry automático com backoff exponencial
- **Token Optimization**: Chunking inteligente para máxima eficiência
- **Memory Efficient**: Processa datasets grandes sem explodir memória

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

---

## 🎉 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:

- 🐛 Reportar bugs
- 💡 Sugerir melhorias
- 🔧 Enviar pull requests
- 📚 Melhorar documentação

---

**Desenvolvido com ❤️ usando LangChain e Google Gemini AI**