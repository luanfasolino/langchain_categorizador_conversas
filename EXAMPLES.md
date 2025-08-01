# 📚 Exemplos Práticos de Uso

Este documento contém exemplos reais de como usar o Categorizador de Conversas em diferentes cenários.

## 🎯 Cenários de Uso

### 1. 🧪 Primeiro Teste (Dataset Pequeno)

**Situação**: Você quer testar o sistema pela primeira vez

```bash
# Teste com apenas 100 tickets para validar setup
python src/main.py --mode all --nrows 100 --workers 1 --cache-control fresh

# O que esperar:
# ✅ Processamento rápido (2-5 minutos)
# ✅ Custo baixo ($0.05-0.20)
# ✅ Validação de que tudo funciona
```

**Saída esperada:**
```
📊 Dados brutos carregados: 100 registros
💬 Tickets válidos: 15 tickets, 89 mensagens
🎯 Total de categorias encontradas: 8
💰 Custo Total: $0.12
```

---

### 2. 🚀 Produção (Dataset Completo)

**Situação**: Processar seu dataset completo pela primeira vez

```bash
# Execução completa com cache fresh
python src/main.py --mode all --cache-control fresh --workers 4

# Monitoramento recomendado:
# 👁️ Acompanhe os logs para verificar progresso
# 💰 Observe estimativas de custo
# ⏱️ Tempo estimado aparece no início
```

**Dica**: Para datasets >50K tickets, considere executar em horários de menor uso da API.

---

### 3. ⚡ Re-execução (Com Cache)

**Situação**: Você já processou antes e quer rodar novamente

```bash
# Usa cache existente - muito mais rápido
python src/main.py --mode all --cache-control continue

# Economia típica:
# ⚡ 70-90% mais rápido
# 💰 70-90% mais barato
# 🎯 Mesma qualidade de resultado
```

---

### 4. 🔄 Atualização de Dados

**Situação**: Você adicionou novos tickets ao dataset

```bash
# Opção 1: Fresh completo (mais lento, máxima precisão)
python src/main.py --mode all --cache-control fresh

# Opção 2: Continue (aproveitará cache de dados iguais)
python src/main.py --mode all --cache-control continue

# Recomendação: Use fresh se >30% dos dados mudaram
```

---

### 5. 🎛️ Processamento por Etapas

**Situação**: Você quer controle granular do processo

```bash
# Passo 1: Só categorização (Map-Reduce-Classify)
python src/main.py --mode categorize --cache-control fresh

# Passo 2: Só análise e bullet points
python src/main.py --mode summarize --cache-control fresh

# Passo 3: Combinar resultados
python src/main.py --mode merge

# Passo 4: Gerar relatórios finais
python src/main.py --mode analyze
```

**Vantagem**: Permite analisar resultados intermediários e ajustar se necessário.

---

## 📊 Exemplos de Resultados

### 🏷️ Categorização Típica

**Input**: Ticket sobre problema de login
```
Ticket 12345:
USER: Não consigo fazer login no sistema
AGENT: Qual mensagem de erro aparece?
USER: Diz que senha está incorreta, mas tenho certeza que está certa
AGENT: Vou resetar sua senha. Verifique seu email.
```

**Output**: 
```csv
ticket_id;categoria
12345;Problema de Autenticação
12345;Reset de Senha
12345;Erro de Login
```

### 💡 Bullet Points Típicos

**Exemplo de sugestões geradas:**

```csv
bullet
Implementar sistema de recuperação de senha mais intuitivo
Adicionar verificação em duas etapas para maior segurança
Criar FAQ específico sobre problemas de login mais comuns
Melhorar mensagens de erro para serem mais claras
Implementar bloqueio temporário após tentativas falhas
```

### 📈 Análise de Categorias

**Exemplo de relatório:**

```
🔝 Top 5 Categorias:
1. Problema de Login: 1,245 tickets (23.4%)
2. Dúvida sobre Cobrança: 987 tickets (18.6%)
3. Solicitação de Reembolso: 654 tickets (12.3%)
4. Problema Técnico: 543 tickets (10.2%)
5. Alteração de Dados: 432 tickets (8.1%)
```

---

## 🔧 Configurações Avançadas

### ⚙️ Ajuste de Performance

```bash
# Para sistemas com pouca memória
python src/main.py --mode all --workers 1 --cache-control fresh

# Para máxima velocidade (se API permitir)
python src/main.py --mode all --workers 8 --cache-control continue

# Para datasets muito grandes (>100K tickets)
python src/main.py --mode categorize --workers 2 --cache-control fresh
# ... depois continue com summarize, merge, analyze
```

### 🎯 Controle de Qualidade

```bash
# Máxima precisão (sempre fresh)
python src/main.py --mode all --cache-control fresh --workers 2

# Teste de consistência (compare resultados)
python src/main.py --mode categorize --cache-control fresh --input-file "sample1.csv"
python src/main.py --mode categorize --cache-control fresh --input-file "sample2.csv"
```

### 💰 Controle de Custos

```bash
# Para desenvolvimento/teste
python src/main.py --mode all --nrows 500 --cache-control fresh

# Para validação antes de produção
python src/main.py --mode all --nrows 5000 --cache-control fresh

# Estimativa antes de execução completa
python src/main.py --mode categorize --nrows 1000 --cache-control fresh
# (use os custos mostrados para extrapolar)
```

---

## 🚨 Troubleshooting Detalhado

### ❌ "Rate Limit Exceeded"

**Sintomas:**
```
❌ classify_batch_1 falhou: 429 ResourceExhausted
```

**Soluções (em ordem de preferência):**

1. **Reduzir workers:**
```bash
python src/main.py --mode all --workers 1
```

2. **Aumentar intervalo entre requests** (edite código se necessário)

3. **Aguardar e tentar novamente** (rate limits resetam por tempo)

**Prevenção:**
- Use no máximo 4 workers
- Evite horários de pico da API (horário comercial US)

---

### ❌ "JSON Parsing Error"

**Sintomas:**
```
❌ Problemas na resposta: ['JSON parsing error: Expecting value: line 1']
```

**Causas e Soluções:**

1. **Batch muito grande:**
```bash
# Solução: Código já reduz automaticamente de 50→5 tickets por batch
# Se persistir, reduza manualmente no código (categorizer.py linha 126)
```

2. **Resposta truncada da API:**
```bash
# Geralmente resolve sozinho com retry automático
# Se persistir, teste com --nrows menor primeiro
```

**Prevenção:**
- Sistema já otimizado para evitar isso
- Batch size ajustado automaticamente

---

### ❌ "Nenhum ticket válido encontrado"

**Sintomas:**
```
💬 Tickets válidos: 0 tickets, 0 mensagens
```

**Verificações:**

1. **Formato do CSV:**
```csv
# ✅ Correto:
ticket_id;category;text;sender;ticket_created_at
12345;TEXT;Olá preciso de ajuda;USER;2024-01-15

# ❌ Incorreto:
id,type,message,from,date
12345,text,Olá preciso de ajuda,user,2024-01-15
```

2. **Conteúdo adequado:**
- Cada ticket precisa de ≥2 mensagens USER E ≥2 mensagens AGENT
- Campo `category` deve ser 'TEXT' (case insensitive)
- Mensagens não podem estar vazias

3. **Verificar relatório de filtragem:**
```
📋 RELATÓRIO DE FILTRAGEM DE DADOS
Registros brutos: 1,000
Após filtro category='TEXT': 953 (95.3%)
Após remoção mensagens AI: 712 (71.2%)
Tickets válidos únicos: 37
Taxa de aproveitamento final: 3.7%
```

**Dicas para melhorar aproveitamento:**
- Remova mensagens automáticas/bot antes do processamento
- Certifique-se que conversas são completas (USER + AGENT)
- Verifique se `sender` está padronizado

---

### ❌ Cache não funciona

**Sintomas:**
```
📦 Cache foi utilizado durante o processamento
# Mas pasta database/cache/ continua vazia
```

**Já resolvido**: Problema foi corrigido no código. Cache agora funciona perfeitamente.

**Verificação:**
```bash
ls -la database/cache/
# Deve mostrar arquivos .pkl após execução
```

---

## 📈 Monitoramento em Tempo Real

### 👁️ Logs Importantes para Acompanhar

**Início do processamento:**
```
💰 Projeção de custos: $3.45
⚡ Total estimado de tokens: 2,456,789
🕒 Tempo estimado: ~2.5 horas
```

**Durante Map phase:**
```
🔄 Fase MAP: Resumindo problemas dos chunks...
📊 Cache performance: 85.2% hit rate
Chunk 145/200 processado - Input: 8,136, Output: 1,526
```

**Durante Classify phase:**
```
🔄 Processando batches CLASSIFY: 85%|████████▌ | 34/40 [15:30<02:45, 3.6s/batch]
✅ Batch 34/40 | Tickets: 5 | Cost: 0.05m¢
```

**Conclusão:**
```
💰 RESUMO FINAL:
   • Total Input Tokens: 2,456,789
   • Total Output Tokens: 145,234
   • Custo Total: $3.32
   • Duração: 2.7h
```

### 🎯 Sinais de Sucesso

- ✅ **Cache hit rate > 0%** (se não for primeira execução)
- ✅ **Taxa de sucesso chunks: 100%**
- ✅ **Taxa de sucesso batches: 100%**
- ✅ **Arquivos gerados em database/analysis_reports/**
- ✅ **Custo dentro do esperado**

### 🚨 Sinais de Problema

- ❌ **Muitos errors/retries**
- ❌ **Cache hit rate: 0%** (quando deveria haver cache)
- ❌ **Taxa de sucesso < 95%**
- ❌ **Custo muito acima do projetado**
- ❌ **Tempo muito maior que estimado**

---

## 🎓 Dicas de Especialista

### 🏆 Melhores Práticas

1. **Sempre teste pequeno primeiro:**
```bash
python src/main.py --mode all --nrows 100 --cache-control fresh
```

2. **Use cache inteligentemente:**
   - **Fresh**: Quando dados ou prompts mudaram
   - **Continue**: Para re-execuções ou análises adicionais

3. **Monitore custos:**
   - Projeções aparecem no início
   - Custos reais aparecem no final
   - Para datasets grandes, teste 10% primeiro

4. **Workers otimizados:**
   - 1-2 workers: Desenvolvimento/teste
   - 2-4 workers: Produção normal
   - 4+ workers: Apenas se API permitir

### 💡 Otimizações Avançadas

1. **Pre-processamento de dados:**
```bash
# Remova registros desnecessários antes do processamento
# Padronize campo 'sender' 
# Limpe textos muito longos ou muito curtos
```

2. **Execução em horários otimizados:**
```bash
# APIs têm menos load fora do horário comercial US
# Considere executar à noite ou fim de semana para datasets grandes
```

3. **Backup de resultados:**
```bash
# Sempre faça backup dos arquivos gerados
cp -r database/analysis_reports/ backup_$(date +%Y%m%d_%H%M%S)/
# Ou backup completo incluindo cache:
cp -r database/ backup_complete_$(date +%Y%m%d_%H%M%S)/
```

---

**💪 Com essas práticas, você terá máxima eficiência e qualidade no seu processo de categorização!**