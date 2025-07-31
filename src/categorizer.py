from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.chains.summarize import load_summarize_chain  # Não usado no momento
from langchain.docstore.document import Document
import pandas as pd
from base_processor import BaseProcessor
import json
import time
import concurrent.futures
from tqdm import tqdm
import os
from typing import List


class TicketCategorizer(BaseProcessor):
    def __init__(
        self,
        api_key: str,
        database_dir: Path,
        max_workers: int = None,
        use_cache: bool = True,
    ):
        super().__init__(
            api_key, database_dir, max_workers=max_workers, use_cache=use_cache
        )
        # Configuração otimizada do Gemini 2.5 Flash conforme Task 1.3
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,  # Temperatura otimizada para categorização consistente
            google_api_key=api_key,
            max_output_tokens=8192,  # Permite respostas mais completas
            top_p=0.8,  # Controla diversidade de respostas
            top_k=40,  # Limita tokens considerados
        )

        # Configurações específicas para o modelo
        self.model_config = {
            "name": "gemini-2.5-flash",
            "temperature": 0.3,
            "optimized_for": "categorization_consistency",
            "max_tokens_input": 1000000,  # Limite de entrada
            "max_tokens_output": 8192,  # Limite de saída
            "cost_per_1k_tokens": 0.25,  # Custo estimado por 1K tokens
        }

        print(f"🤖 Modelo configurado: {self.model_config['name']}")
        print(f"   • Temperature: {self.model_config['temperature']}")
        print(f"   • Max tokens output: {self.model_config['max_tokens_output']:,}")
        print(f"   • Otimizado para: {self.model_config['optimized_for']}")

        # Template otimizado para análise inicial dos chunks (MAP phase)
        self.map_template = ChatPromptTemplate.from_template(
            """
            Você é um especialista em análise de conversas de suporte ao cliente. Sua tarefa é identificar padrões e categorizar motivos de contato.

            INSTRUÇÕES:
            1. Analise cuidadosamente as conversas fornecidas
            2. Identifique os principais padrões e motivos subjacentes dos contatos
            3. Foque nas causas raiz dos problemas, não apenas nos sintomas
            4. Considere contextos específicos como: pagamentos, sistema, reservas, antifraude, etc.
            5. Mantenha consistência terminológica

            CONVERSAS PARA ANÁLISE:
            {text}

            FORMATO DE RESPOSTA:
            Forneça uma análise concisa em um parágrafo único, destacando as principais categorias identificadas e seus padrões recorrentes.
        """
        )

        # Template otimizado para combinar análises parciais (COMBINE phase)
        self.combine_template = ChatPromptTemplate.from_template(
            """
            Você é um especialista em consolidação de análises de suporte ao cliente. Sua tarefa é sintetizar múltiplas análises parciais em uma visão unificada.

            INSTRUÇÕES:
            1. Combine as análises parciais fornecidas em uma visão consolidada
            2. Identifique padrões mais frequentes e motivos subjacentes recorrentes
            3. Estabeleça um vocabulário consistente para as categorias
            4. Priorize categorias específicas sobre genéricas
            5. Mantenha foco nas causas raiz dos contatos

            ANÁLISES PARCIAIS PARA CONSOLIDAÇÃO:
            {text}

            FORMATO DE RESPOSTA:
            Forneça uma análise consolidada em um parágrafo único, destacando as principais categorias padronizadas e seus motivos subjacentes.
            Exemplo de categorias esperadas: "Problema com Pagamento", "Erro no Sistema", "Dúvida de Reserva", "Questão Antifraude", etc.
        """
        )

        # Template otimizado para categorização final (REDUCE phase)
        self.categorize_template = ChatPromptTemplate.from_template(
            """
            Você é um especialista em categorização de tickets de suporte ao cliente. Use a análise consolidada para categorizar cada ticket de forma precisa e consistente.

            REGRAS DE CATEGORIZAÇÃO:
            1. Use EXATAMENTE o ticket_id que aparece após "Ticket"
            2. Atribua 1-3 categorias por ticket, em ordem de relevância
            3. Priorize categorias específicas sobre genéricas
            4. Mantenha consistência com o vocabulário da análise consolidada
            5. Evite termos genéricos: "Problemas Gerais", "Outros", "Diversos"
            6. Cada categoria: máximo 50 caracteres, clara e específica
            7. Para contextos específicos (antifraude, pagamento, sistema), use a categoria mais específica

            FORMATO JSON OBRIGATÓRIO (RESPONDA APENAS COM O JSON):
            {{
              "cat": [
                {{"id": "123", "cat": ["Categoria Específica 1", "Categoria 2"]}},
                {{"id": "456", "cat": ["Categoria Única"]}}
              ]
            }}

            ANÁLISE CONSOLIDADA:
            {analysis}

            TICKETS PARA CATEGORIZAÇÃO:
            {tickets}
        """
        )

        # Text Splitter otimizado com melhores práticas Context7
        # Usando RecursiveCharacterTextSplitter + tiktoken conforme latest docs
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",  # Modelo de referência para tokenização
            chunk_size=100000,  # Otimizado: menor que 1M para melhor precisão
            chunk_overlap=20000,  # Overlap mais substancial (20% do chunk)
            add_start_index=True,  # Rastreamento de posição conforme best practices
        )

        # Configurações de batching para otimização
        self.optimal_batch_size = 220  # Tamanho otimizado baseado em testes
        self.token_buffer_size = 50000  # Buffer para evitar overflow de tokens

    def create_optimized_chunks(self, full_text: str) -> List[Document]:
        """
        Cria chunks otimizados usando melhores práticas Context7 + Task 1.1.
        Usa RecursiveCharacterTextSplitter com tiktoken para precisão máxima.
        """
        print(
            "🔧 Criando chunks com RecursiveCharacterTextSplitter + tiktoken (Context7)..."
        )

        # Cria documento inicial
        initial_doc = Document(page_content=full_text)

        # Divide usando split_documents para preservar metadados e add_start_index
        docs = self.text_splitter.split_documents([initial_doc])
        print(f"📊 Texto dividido em {len(docs)} chunks com precisão tiktoken")

        # Adiciona metadados extras para rastreamento
        total_tokens = 0
        for i, doc in enumerate(docs):
            tokens = self.estimate_tokens(doc.page_content)
            total_tokens += tokens

            # Preserva start_index do RecursiveCharacterTextSplitter e adiciona nossos metadados
            doc.metadata.update(
                {
                    "chunk_id": i + 1,
                    "estimated_tokens": tokens,
                    "processing_phase": "map",
                    "splitter_type": "RecursiveCharacterTextSplitter",
                    "tiktoken_based": True,
                }
            )

        print(f"⚡ Total estimado de tokens: {total_tokens:,}")
        if len(docs) > 0:
            print(f"📈 Média de tokens por chunk: {total_tokens // len(docs):,}")
            print(
                f"🎯 Start index tracking: {'Ativo' if docs[0].metadata.get('start_index') is not None else 'Inativo'}"
            )
        else:
            print(
                "⚠️  Nenhum chunk foi criado - verifique se há dados válidos para processar"
            )

        return docs

    def setup_parallel_executor(self) -> dict:
        """
        Configura o executor paralelo com métricas de performance.
        Implementa especificação da Task 1.2.
        """
        executor_config = {
            "max_workers": self.max_workers,
            "worker_type": "ThreadPoolExecutor",
            "monitoring": True,
            "graceful_shutdown": True,
            "performance_metrics": {
                "total_workers": self.max_workers,
                "cpu_count": os.cpu_count(),
                "recommended_workers": min(os.cpu_count(), 4),
            },
        }

        print("🚀 Configurando executor paralelo:")
        print(f"   • Workers: {executor_config['max_workers']}")
        print(
            f"   • CPUs disponíveis: {executor_config['performance_metrics']['cpu_count']}"
        )
        print(
            f"   • Limite recomendado: {executor_config['performance_metrics']['recommended_workers']}"
        )

        return executor_config

    def validate_model_response(
        self, response: str, expected_format: str = "json"
    ) -> dict:
        """
        Valida respostas do modelo Gemini 2.5 Flash.
        Implementa validação conforme Task 1.3.
        """
        validation_result = {
            "is_valid": False,
            "response_length": len(response),
            "format_type": expected_format,
            "issues": [],
            "parsed_data": None,
        }

        try:
            if expected_format == "json":
                # Valida formato JSON
                json_str = self.extract_json(response)
                parsed_data = json.loads(json_str)
                validation_result["parsed_data"] = parsed_data
                validation_result["is_valid"] = True

                # Validações específicas para categorização
                if "cat" in parsed_data and isinstance(parsed_data["cat"], list):
                    for item in parsed_data["cat"]:
                        if (
                            not isinstance(item, dict)
                            or "id" not in item
                            or "cat" not in item
                        ):
                            validation_result["issues"].append(
                                "Invalid item structure in cat array"
                            )
                        elif not isinstance(item["cat"], list):
                            validation_result["issues"].append(
                                "Categories must be a list"
                            )
                        elif len(item["cat"]) > 3:
                            validation_result["issues"].append(
                                "More than 3 categories per ticket"
                            )
                else:
                    validation_result["issues"].append("Missing or invalid 'cat' field")
            else:
                # Para respostas de texto (MAP e COMBINE phases)
                if len(response.strip()) < 50:
                    validation_result["issues"].append("Response too short")
                elif len(response.strip()) > 5000:
                    validation_result["issues"].append("Response too long")
                else:
                    validation_result["is_valid"] = True

        except json.JSONDecodeError as e:
            validation_result["issues"].append(f"JSON parsing error: {str(e)}")
        except Exception as e:
            validation_result["issues"].append(f"Validation error: {str(e)}")

        return validation_result

    def setup_retry_strategy(self) -> dict:
        """
        Configura estratégia de retry com exponential backoff.
        Implementa especificação da Task 1.4.
        """
        retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,  # Delay inicial em segundos
            "max_delay": 60.0,  # Delay máximo em segundos
            "exponential_base": 2.0,  # Base para exponential backoff
            "jitter": True,  # Adiciona aleatoriedade para evitar thundering herd
            "retry_on_errors": [
                "API_QUOTA_EXCEEDED",
                "RATE_LIMIT_EXCEEDED",
                "INTERNAL_ERROR",
                "TIMEOUT",
                "CONNECTION_ERROR",
                "JSON_DECODE_ERROR",
            ],
        }

        print("🔄 Configurando estratégia de retry:")
        print(f"   • Max retries: {retry_config['max_retries']}")
        print(f"   • Base delay: {retry_config['base_delay']}s")
        print(f"   • Max delay: {retry_config['max_delay']}s")
        print(f"   • Exponential base: {retry_config['exponential_base']}")

        return retry_config

    def calculate_retry_delay(
        self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        """
        Calcula delay para retry com exponential backoff e jitter.
        """
        import random

        # Exponential backoff: delay = base_delay * (2 ^ attempt)
        delay = base_delay * (2**attempt)

        # Aplica limite máximo
        delay = min(delay, max_delay)

        # Adiciona jitter (±25% de aleatoriedade)
        jitter = delay * 0.25 * (random.random() - 0.5)
        delay = delay + jitter

        return max(0.1, delay)  # Mínimo de 0.1 segundo

    def execute_with_retry(
        self, operation_func, operation_name: str, max_retries: int = 3, **kwargs
    ) -> dict:
        """
        Executa operação com retry automático e error handling.
        Implementa especificação da Task 1.4.
        """
        retry_config = self.setup_retry_strategy()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = operation_func(**kwargs)
                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "result": result,
                    "attempt": attempt + 1,
                    "execution_time": execution_time,
                    "operation": operation_name,
                }

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

                # Log do erro
                print(
                    f"❌ {operation_name} falhou (tentativa {attempt + 1}/{max_retries + 1}): {str(e)}"
                )

                # Se é o último attempt ou erro não é recuperável, falha
                if attempt >= max_retries or not self._is_retryable_error(error_type):
                    break

                # Calcula delay e aguarda antes do próximo retry
                delay = self.calculate_retry_delay(
                    attempt, retry_config["base_delay"], retry_config["max_delay"]
                )
                print(f"🔄 Aguardando {delay:.2f}s antes do próximo retry...")
                time.sleep(delay)

        # Se chegou aqui, todas as tentativas falharam
        return {
            "success": False,
            "error": str(last_error),
            "error_type": self._classify_error(last_error),
            "attempts": max_retries + 1,
            "operation": operation_name,
        }

    def _classify_error(self, error: Exception) -> str:
        """Classifica o tipo de erro para determinar se é recuperável."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "quota" in error_str or "rate limit" in error_str:
            return "RATE_LIMIT_EXCEEDED"
        elif "timeout" in error_str or "timed out" in error_str:
            return "TIMEOUT"
        elif "connection" in error_str or "network" in error_str:
            return "CONNECTION_ERROR"
        elif "json" in error_str and "decode" in error_str:
            return "JSON_DECODE_ERROR"
        elif "internal" in error_str or "server error" in error_str:
            return "INTERNAL_ERROR"
        elif error_type in ["ValueError", "TypeError"]:
            return "VALIDATION_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def _is_retryable_error(self, error_type: str) -> bool:
        """Determina se um erro justifica retry."""
        retryable_errors = {
            "RATE_LIMIT_EXCEEDED",
            "TIMEOUT",
            "CONNECTION_ERROR",
            "INTERNAL_ERROR",
            "JSON_DECODE_ERROR",
        }
        return error_type in retryable_errors

    def setup_token_tracking_system(self) -> dict:
        """
        Configura sistema abrangente de tracking de tokens e estimativa de custos.
        Implementa especificação da Task 1.5 com melhores práticas Context7.
        """
        tracking_config = {
            "enabled": True,
            "cost_per_1k_input_tokens": 0.00015,  # Gemini 2.5 Flash pricing as of Jul 2025 (USD)
            "cost_per_1k_output_tokens": 0.00060,  # Gemini 2.5 Flash (non-thinking mode) as of Jul 2025 (USD)
            # "cost_per_1k_output_tokens_thinking": 0.00350 # Add if using thinking-mode pricing
            "currency": "USD",
            "tracking_phases": ["map", "combine", "categorize"],
            "budget_monitoring": True,
            "cost_projections": True,
            "detailed_breakdown": True,
        }

        self.token_tracker = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "phase_breakdown": {
                "map": {"input": 0, "output": 0, "cost": 0.0, "chunks_processed": 0},
                "combine": {"input": 0, "output": 0, "cost": 0.0},
                "categorize": {
                    "input": 0,
                    "output": 0,
                    "cost": 0.0,
                    "batches_processed": 0,
                },
            },
            "performance_metrics": {
                "tokens_per_second": 0.0,
                "cost_per_ticket": 0.0,
                "avg_tokens_per_chunk": 0.0,
                "avg_tokens_per_batch": 0.0,
            },
            "session_start_time": time.time(),
            "budget_alerts": [],
        }

        print("💰 Sistema de Token Tracking configurado:")
        print(f"   • Input: ${tracking_config['cost_per_1k_input_tokens']}/1K tokens")
        print(f"   • Output: ${tracking_config['cost_per_1k_output_tokens']}/1K tokens")
        print(
            f"   • Monitoramento de budget: {'Ativo' if tracking_config['budget_monitoring'] else 'Inativo'}"
        )

    def _format_cost_friendly(self, cost_usd: float) -> str:
        """
        Formata custos de forma amigável ao usuário.
        Para valores < $0.10, mostra em centavos.
        """
        if cost_usd < 0.001:
            return f"{cost_usd * 1000:.2f}m¢"  # milicents para valores muito pequenos
        elif cost_usd < 0.10:
            return f"{cost_usd * 100:.1f}¢"  # centavos
        else:
            return f"${cost_usd:.3f}"  # dólares

    def track_token_usage(
        self,
        phase: str,
        input_tokens: int,
        output_tokens: int,
        additional_context: dict = None,
    ) -> dict:
        """
        Registra uso de tokens com detalhamento por fase e cálculo de custos.
        Segue melhores práticas Context7 para tracking preciso.
        """
        if not hasattr(self, "token_tracker"):
            self.setup_token_tracking_system()

        # Calcula custos baseado em pricing Gemini 2.5 Flash
        input_cost = (input_tokens / 1000000) * 0.075
        output_cost = (output_tokens / 1000000) * 0.30
        total_cost = input_cost + output_cost

        # Atualiza tracking global
        self.token_tracker["total_input_tokens"] += input_tokens
        self.token_tracker["total_output_tokens"] += output_tokens
        self.token_tracker["total_cost"] += total_cost

        # Atualiza breakdown por fase
        if phase in self.token_tracker["phase_breakdown"]:
            phase_data = self.token_tracker["phase_breakdown"][phase]
            phase_data["input"] += input_tokens
            phase_data["output"] += output_tokens
            phase_data["cost"] += total_cost

            # Adiciona contexto específico por fase
            if additional_context:
                if phase == "map" and "chunk_processed" in additional_context:
                    phase_data["chunks_processed"] += 1
                elif phase == "categorize" and "batch_processed" in additional_context:
                    phase_data["batches_processed"] += 1

        # Calcula métricas de performance em tempo real
        elapsed_time = time.time() - self.token_tracker["session_start_time"]
        total_tokens = (
            self.token_tracker["total_input_tokens"]
            + self.token_tracker["total_output_tokens"]
        )

        self.token_tracker["performance_metrics"]["tokens_per_second"] = (
            total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        )

        return {
            "phase": phase,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cumulative_cost": self.token_tracker["total_cost"],
        }

    def generate_cost_projection(
        self, dataset_size: int, sample_tokens: int = None
    ) -> dict:
        """
        Gera projeções de custo baseadas no uso atual ou tamanho de amostra.
        Implementa funcionalidade de budget monitoring da Task 1.5.
        """
        if not hasattr(self, "token_tracker"):
            return {"error": "Token tracking não inicializado"}

        # Usa tokens já processados como base ou valor fornecido
        if sample_tokens is None:
            processed_tokens = (
                self.token_tracker["total_input_tokens"]
                + self.token_tracker["total_output_tokens"]
            )
            if processed_tokens == 0:
                return {"error": "Nenhum token processado para projeção"}
        else:
            processed_tokens = sample_tokens

        # Calcula tokens por item processado (ticket)
        # Estima baseado no que já foi processado
        current_cost = self.token_tracker["total_cost"]

        if current_cost > 0:
            cost_per_token = current_cost / processed_tokens
            projected_total_tokens = processed_tokens * (
                dataset_size / max(1, dataset_size)
            )
            projected_cost = cost_per_token * projected_total_tokens
        else:
            # Estimativa conservadora se não há dados
            avg_tokens_per_ticket = processed_tokens or 1000  # Fallback
            projected_total_tokens = avg_tokens_per_ticket * dataset_size
            projected_cost = (projected_total_tokens / 1000) * 0.25  # Estimativa média

        projection = {
            "dataset_size": dataset_size,
            "projected_total_tokens": int(projected_total_tokens),
            "projected_cost": round(projected_cost, 4),
            "cost_breakdown": {
                "conservative_estimate": round(projected_cost * 0.8, 4),
                "realistic_estimate": round(projected_cost, 4),
                "pessimistic_estimate": round(projected_cost * 1.2, 4),
            },
            "budget_alerts": [],
        }

        # Alertas de budget
        if projected_cost > 50:
            projection["budget_alerts"].append("⚠️  Custo projetado alto: >$50")
        if projected_cost > 100:
            projection["budget_alerts"].append("🚨 Custo projetado muito alto: >$100")

        return projection

    def get_comprehensive_token_report(self) -> dict:
        """
        Gera relatório completo de uso de tokens e custos.
        Inclui todas as métricas especificadas na Task 1.5.
        """
        if not hasattr(self, "token_tracker"):
            return {"error": "Token tracking não inicializado"}

        elapsed_time = time.time() - self.token_tracker["session_start_time"]
        total_tokens = (
            self.token_tracker["total_input_tokens"]
            + self.token_tracker["total_output_tokens"]
        )

        # Atualiza métricas finais
        self.token_tracker["performance_metrics"]["tokens_per_second"] = (
            total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        )

        report = {
            "session_summary": {
                "total_input_tokens": self.token_tracker["total_input_tokens"],
                "total_output_tokens": self.token_tracker["total_output_tokens"],
                "total_tokens": total_tokens,
                "total_cost_usd": round(self.token_tracker["total_cost"], 4),
                "session_duration_seconds": round(elapsed_time, 2),
            },
            "phase_breakdown": self.token_tracker["phase_breakdown"],
            "performance_metrics": {
                "tokens_per_second": round(
                    self.token_tracker["performance_metrics"]["tokens_per_second"], 2
                ),
                "cost_efficiency": {
                    "cost_per_1k_tokens": round(
                        (self.token_tracker["total_cost"] / max(total_tokens, 1))
                        * 1000,
                        4,
                    ),
                    "input_output_ratio": round(
                        self.token_tracker["total_input_tokens"]
                        / max(self.token_tracker["total_output_tokens"], 1),
                        2,
                    ),
                },
            },
            "cost_analysis": {
                "input_cost": round(
                    (self.token_tracker["total_input_tokens"] / 1000) * 0.125, 4
                ),
                "output_cost": round(
                    (self.token_tracker["total_output_tokens"] / 1000) * 0.375, 4
                ),
                "cost_distribution": {
                    "map_phase": round(
                        self.token_tracker["phase_breakdown"]["map"]["cost"], 4
                    ),
                    "combine_phase": round(
                        self.token_tracker["phase_breakdown"]["combine"]["cost"], 4
                    ),
                    "categorize_phase": round(
                        self.token_tracker["phase_breakdown"]["categorize"]["cost"], 4
                    ),
                },
            },
            "budget_alerts": self.token_tracker["budget_alerts"],
            "recommendations": self._generate_cost_recommendations(),
        }

        return report

    def _generate_cost_recommendations(self) -> list:
        """
        Gera recomendações para otimização de custos baseadas no uso atual.
        """
        recommendations = []

        if not hasattr(self, "token_tracker"):
            return recommendations

        total_cost = self.token_tracker["total_cost"]
        input_tokens = self.token_tracker["total_input_tokens"]
        output_tokens = self.token_tracker["total_output_tokens"]

        # Recomendações baseadas no padrão de uso
        if input_tokens > output_tokens * 3:
            recommendations.append(
                "💡 Alto ratio input/output - considere chunks menores para reduzir tokens de entrada"
            )

        if total_cost > 10:
            recommendations.append(
                "💰 Custo elevado - considere usar cache mais agressivo ou processamento em lotes maiores"
            )

        # Recomendações por fase
        phases = self.token_tracker["phase_breakdown"]
        map_cost = phases["map"]["cost"]
        combine_cost = phases["combine"]["cost"]
        categorize_cost = phases["categorize"]["cost"]

        if map_cost > combine_cost + categorize_cost:
            recommendations.append(
                "🔄 Fase MAP dominando custos - otimize tamanho de chunks"
            )

        if categorize_cost > map_cost + combine_cost:
            recommendations.append(
                "🎯 Fase CATEGORIZE dominando custos - otimize tamanho de batches"
            )

        if not recommendations:
            recommendations.append(
                "✅ Uso de tokens otimizado - padrão eficiente detectado"
            )

        return recommendations

    def process_tickets(self, input_file: Path, nrows: int = None) -> Path:
        """Processa os tickets usando uma abordagem map-reduce para categorização"""
        tickets = self.prepare_data(input_file, nrows=nrows)
        print(f"\nProcessando arquivo: {input_file}")
        print(f"Total de tickets para processar: {len(tickets)}")

        # Inicializa o sistema de tracking de tokens - Task 1.5
        self.setup_token_tracking_system()

        # Gera projeção de custos inicial
        initial_projection = self.generate_cost_projection(len(tickets))
        if "error" not in initial_projection:
            print(f"💰 Projeção de custos: ${initial_projection['projected_cost']:.2f}")
            print(
                f"   • Tokens estimados: {initial_projection['projected_total_tokens']:,}"
            )
            if initial_projection["budget_alerts"]:
                for alert in initial_projection["budget_alerts"]:
                    print(f"   {alert}")

        # Prepara o texto completo
        full_text = "\n\n".join(
            f"Ticket {ticket['ticket_id']}:\n{ticket['text']}" for ticket in tickets
        )

        # Cria chunks otimizados usando estratégia Map-Reduce Task 1.1
        docs = self.create_optimized_chunks(full_text)

        # Configura chains LCEL otimizadas conforme Context7 best practices
        map_chain = self.map_template | self.llm | StrOutputParser()
        combine_chain = self.combine_template | self.llm | StrOutputParser()

        # Configura executor paralelo Task 1.2
        self.setup_parallel_executor()

        # 1. Map: Analisa cada chunk usando cache inteligente (processamento paralelo otimizado)
        print("\n🔄 Fase MAP: Realizando análise dos chunks em paralelo com cache inteligente...")
        
        # Usa sistema de cache inteligente do BaseProcessor passando a chain diretamente
        try:
            partial_analyses, total_input_tokens, total_output_tokens = self.process_chunks_with_cache(
                docs, map_chain, "map"
            )

            # Estatísticas de performance com cache inteligente
            successful_chunks = len(partial_analyses)
            print(f"✅ Fase MAP concluída: {successful_chunks}/{len(docs)} chunks processados")
            
            # Exibe estatísticas de cache se disponível
            if self.cache_manager:
                cache_stats = self.cache_manager.get_statistics()
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                print(f"📊 Cache performance: {hit_rate:.1f}% hit rate")
                if hit_rate > 0:
                    print(f"⚡ Cache economizou {cache_stats.get('hits', 0)} processamentos de LLM!")
            
            print("\n📊 Estatísticas de Performance MAP:")
            print(f"   • Chunks processados: {successful_chunks}/{len(docs)}")
            print(f"   • Total Input Tokens: {total_input_tokens:,}")
            print(f"   • Total Output Tokens: {total_output_tokens:,}")
            
            # Mostra estatísticas de tokens da fase MAP
            map_phase_data = self.token_tracker["phase_breakdown"]["map"]
            print(
                f"   • Tokens MAP - Input: {map_phase_data['input']:,} | Output: {map_phase_data['output']:,}"
            )
            map_cost_formatted = self._format_cost_friendly(map_phase_data["cost"])
            print(f"   • Custo da fase MAP: {map_cost_formatted}")
            
        except Exception as e:
            print(f"❌ Erro na fase MAP: {str(e)}")
            return None

        # 2. Reduce: Combina as análises parciais
        print("\n🔄 Fase COMBINE: Combinando análises parciais...")
        try:
            combine_input = "\n\n".join(partial_analyses)
            combine_tokens_in = self.estimate_tokens(combine_input)

            consolidated_analysis = combine_chain.invoke({"text": combine_input})

            combine_tokens_out = self.estimate_tokens(consolidated_analysis)

            # Registra no sistema de tracking - Task 1.5
            combine_tracking = self.track_token_usage(
                "combine", combine_tokens_in, combine_tokens_out
            )

            combine_cost_formatted = self._format_cost_friendly(
                combine_tracking["total_cost"]
            )
            print(
                f"💰 Fase COMBINE - Tokens: {combine_tokens_in:,} → {combine_tokens_out:,} | Custo: {combine_cost_formatted}"
            )

        except Exception as e:
            print(f"Erro ao combinar análises: {str(e)}")
            return None

        # 3. Categoriza os tickets usando a análise consolidada, processamento paralelo otimizado
        print("\n🔄 Fase CATEGORIZE: Categorizando tickets em paralelo...")
        try:
            # Chain de summary otimizada com LCEL Context7 best practices
            print("\nResumindo análise consolidada com LCEL otimizado...")
            summary_template = ChatPromptTemplate.from_template(
                """
                Você é um especialista em síntese de análises de suporte. Resuma de forma concisa mantendo apenas informações essenciais sobre categorias e padrões.

                ANÁLISE CONSOLIDADA:
                {text}

                RESUMO CONCISO (mantenha categorias específicas e padrões principais):
                """
            )
            summary_chain = summary_template | self.llm | StrOutputParser()

            analysis_summary = summary_chain.invoke({"text": consolidated_analysis})
            print("Análise resumida com sucesso.")

            # Configure o categorize_chain
            categorize_chain = self.categorize_template | self.llm | StrOutputParser()

            # Usa tamanho de batch otimizado da configuração
            batch_size = self.optimal_batch_size

            # Divida a lista de tickets em batches
            ticket_batches = [
                tickets[i : i + batch_size] for i in range(0, len(tickets), batch_size)
            ]
            print(f"Dividindo os tickets em {len(ticket_batches)} batches.")

            # Função otimizada para processar batch com retry logic Task 1.4
            def process_batch(batch_index, batch):
                def _process_batch_internal():
                    # Construa o texto para este batch
                    batch_text = "\n\n".join(
                        f"Ticket {ticket['ticket_id']}:\n{ticket['text']}"
                        for ticket in batch
                    )

                    categorize_input = {
                        "analysis": analysis_summary,
                        "tickets": batch_text,
                    }

                    # Monitora tokens para este batch
                    batch_input_tokens = sum(
                        self.estimate_tokens(str(v)) for v in categorize_input.values()
                    )

                    response = categorize_chain.invoke(categorize_input)
                    batch_output_tokens = self.estimate_tokens(response)

                    # Registra no sistema de tracking - Task 1.5
                    batch_tracking = self.track_token_usage(
                        "categorize",
                        batch_input_tokens,
                        batch_output_tokens,
                        {"batch_processed": True},
                    )

                    # Valida resposta do modelo conforme Task 1.3
                    validation = self.validate_model_response(response, "json")
                    if not validation["is_valid"]:
                        print(
                            f"⚠️  Problemas na resposta do batch {batch_index}: {validation['issues']}"
                        )

                    json_str = self.extract_json(response)
                    batch_results = json.loads(json_str)

                    if (
                        not isinstance(batch_results, dict)
                        or "cat" not in batch_results
                    ):
                        return {
                            "results": [],
                            "input_tokens": batch_input_tokens,
                            "output_tokens": batch_output_tokens,
                            "success": False,
                            "error": "Formato de resposta inválido",
                            "tracking_result": batch_tracking,
                        }

                    # Valida e normaliza as categorias do batch
                    valid_batch_results = []
                    for item in batch_results["cat"]:
                        if (
                            not isinstance(item, dict)
                            or "id" not in item
                            or "cat" not in item
                        ):
                            continue
                        if not isinstance(item["cat"], list):
                            continue
                        valid_batch_results.append(
                            {
                                "ticket_id": item["id"],
                                "categorias": [
                                    cat.strip()
                                    for cat in item["cat"]
                                    if isinstance(cat, str)
                                ][:3],
                            }
                        )

                    return {
                        "results": valid_batch_results,
                        "input_tokens": batch_input_tokens,
                        "output_tokens": batch_output_tokens,
                        "batch_size": len(batch),
                        "tracking_result": batch_tracking,
                    }

                # Executa com retry automático
                result = self.execute_with_retry(
                    _process_batch_internal,
                    f"process_batch_{batch_index}",
                    max_retries=3,
                )

                if result["success"]:
                    return {
                        **result["result"],
                        "processing_time": result["execution_time"],
                        "retry_attempts": result["attempt"],
                        "success": True,
                    }
                else:
                    print(
                        f"❌ Batch {batch_index} falhou após {result['attempts']} tentativas: {result['error']}"
                    )
                    return {
                        "results": [],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "processing_time": 0,
                        "retry_attempts": result["attempts"],
                        "success": False,
                        "error": result["error"],
                        "error_type": result.get("error_type", "UNKNOWN"),
                    }

            # Processa os batches em paralelo com monitoramento Task 1.2
            all_categorization_results = []
            total_batch_processing_time = 0
            successful_batches = 0

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submete todos os batches para processamento
                future_to_batch = {
                    executor.submit(process_batch, i, batch): (i, batch)
                    for i, batch in enumerate(ticket_batches, 1)
                }

                # Coleta os resultados com monitoramento avançado Task 1.2
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_batch),
                    total=len(ticket_batches),
                    desc="🔄 Processando batches CATEGORIZE",
                ):
                    batch_index, _ = future_to_batch[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            all_categorization_results.extend(result["results"])
                            total_batch_processing_time += result.get(
                                "processing_time", 0
                            )
                            successful_batches += 1

                            retry_info = (
                                f" (retry: {result.get('retry_attempts', 1)})"
                                if result.get("retry_attempts", 1) > 1
                                else ""
                            )
                            tracking_info = result.get("tracking_result", {})
                            cost_info = ""
                            if tracking_info:
                                cost_value = tracking_info.get("total_cost", 0)
                                cost_formatted = self._format_cost_friendly(cost_value)
                                cost_info = f" | Cost: {cost_formatted}"

                            # Log detalhado apenas a cada 10 batches ou no último para reduzir verbosidade
                            if (
                                batch_index % 10 == 0
                                or batch_index == len(ticket_batches)
                                or result.get("retry_attempts", 1) > 1
                            ):
                                print(
                                    f"✅ Batch {batch_index}/{len(ticket_batches)} | "
                                    f"Tickets: {len(result['results'])} | "
                                    f"In: {result['input_tokens']:,} | "
                                    f"Out: {result['output_tokens']:,} | "
                                    f"Time: {result.get('processing_time', 0):.2f}s{cost_info}{retry_info}"
                                )
                        else:
                            print(
                                f"❌ Falha no batch {batch_index}: {result.get('error', 'Erro desconhecido')}"
                            )
                    except Exception as e:
                        print(f"⚠️  Exceção ao processar batch {batch_index}: {str(e)}")

            # Estatísticas de performance dos batches
            avg_batch_time = (
                total_batch_processing_time / successful_batches
                if successful_batches > 0
                else 0
            )
            categorize_phase_data = self.token_tracker["phase_breakdown"]["categorize"]

            print("\n📊 Estatísticas de Performance CATEGORIZE:")
            print(
                f"   • Batches processados: {successful_batches}/{len(ticket_batches)}"
            )
            print(f"   • Tempo médio por batch: {avg_batch_time:.2f}s")
            print(
                f"   • Tempo total de processamento: {total_batch_processing_time:.2f}s"
            )
            print(
                f"   • Tokens CATEGORIZE - Input: {categorize_phase_data['input']:,} | Output: {categorize_phase_data['output']:,}"
            )
            categorize_cost_formatted = self._format_cost_friendly(
                categorize_phase_data["cost"]
            )
            print(f"   • Custo da fase CATEGORIZE: {categorize_cost_formatted}")

            if not all_categorization_results:
                print("\nAviso: Nenhuma categorização válida foi gerada!")
                return None

            print(
                f"\nResultados obtidos: {len(all_categorization_results)} tickets categorizados"
            )
            print("Exemplo do primeiro resultado:", all_categorization_results[0])

            # Prepara os dados para o DataFrame expandido
            expanded_results = []
            for result in all_categorization_results:
                ticket_id = result["ticket_id"]
                for categoria in result["categorias"]:
                    expanded_results.append(
                        {"ticket_id": ticket_id, "categoria": categoria}
                    )

            # Cria e salva o DataFrame expandido
            results_df = pd.DataFrame(expanded_results)
            results_df = results_df.sort_values("ticket_id")

            output_file = self.database_dir / "categorized_tickets.csv"
            results_df.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig")

            print(f"\nResultados salvos em {output_file}")
            print(f"Total de linhas no arquivo: {len(results_df)}")
            print("\nPrimeiras linhas do arquivo:")
            print(results_df.head().to_string())

            # Gera relatório abrangente de tokens e custos - Task 1.5
            comprehensive_report = self.get_comprehensive_token_report()

            print("\n" + "=" * 60)
            print("📊 RELATÓRIO ABRANGENTE DE TOKENS E CUSTOS")
            print("=" * 60)

            # Resumo da sessão
            session_summary = comprehensive_report["session_summary"]
            print("💰 RESUMO FINAL:")
            print(f"   • Total Input Tokens: {session_summary['total_input_tokens']:,}")
            print(
                f"   • Total Output Tokens: {session_summary['total_output_tokens']:,}"
            )
            print(f"   • Total Geral: {session_summary['total_tokens']:,}")
            total_cost_formatted = self._format_cost_friendly(
                session_summary["total_cost_usd"]
            )
            print(f"   • Custo Total: {total_cost_formatted}")
            print(f"   • Duração: {session_summary['session_duration_seconds']:.1f}s")

            # Breakdown por fase
            print("\n🔍 BREAKDOWN POR FASE:")
            phase_breakdown = comprehensive_report["phase_breakdown"]
            for phase, data in phase_breakdown.items():
                phase_name = phase.upper()
                print(
                    f"   • {phase_name}: ${data['cost']:.4f} | Input: {data['input']:,} | Output: {data['output']:,}"
                )
                if phase == "map" and "chunks_processed" in data:
                    print(f"     └─ Chunks processados: {data['chunks_processed']}")
                elif phase == "categorize" and "batches_processed" in data:
                    print(f"     └─ Batches processados: {data['batches_processed']}")

            # Métricas de performance
            performance = comprehensive_report["performance_metrics"]
            print("\n⚡ PERFORMANCE:")
            print(f"   • Tokens por segundo: {performance['tokens_per_second']:.1f}")
            print(
                f"   • Custo por 1K tokens: ${performance['cost_efficiency']['cost_per_1k_tokens']}"
            )
            print(
                f"   • Ratio Input/Output: {performance['cost_efficiency']['input_output_ratio']:.2f}"
            )

            # Recomendações
            print("\n💡 RECOMENDAÇÕES:")
            recommendations = comprehensive_report["recommendations"]
            for rec in recommendations:
                print(f"   {rec}")

            # Alertas de budget
            if comprehensive_report["budget_alerts"]:
                print("\n🚨 ALERTAS DE BUDGET:")
                for alert in comprehensive_report["budget_alerts"]:
                    print(f"   {alert}")

            print("\n=== 🔄 Estatísticas de Error Handling ===")
            print(f"Chunks processados com sucesso: {successful_chunks}/{len(docs)}")
            print(
                f"Batches processados com sucesso: {successful_batches}/{len(ticket_batches)}"
            )
            print(
                f"Taxa de sucesso chunks: {(successful_chunks / len(docs) * 100):.1f}%"
            )
            print(
                f"Taxa de sucesso batches: {(successful_batches / len(ticket_batches) * 100):.1f}%"
            )

            if successful_chunks < len(docs) or successful_batches < len(
                ticket_batches
            ):
                print(
                    "⚠️  Alguns itens falharam após múltiplas tentativas. Verifique logs de erro."
                )

            print("=" * 60)

            return output_file

        except Exception as e:
            print(f"Erro ao categorizar tickets: {str(e)}")
            return None
