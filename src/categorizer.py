from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import TokenTextSplitter

# from langchain.chains.summarize import load_summarize_chain  # Não usado no momento
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnablePassthrough
import pandas as pd
from base_processor import BaseProcessor
import json
import time
import concurrent.futures
from tqdm import tqdm


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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key
        )

        # Template para análise inicial dos chunks
        self.map_template = ChatPromptTemplate.from_template(
            """
            Analise cuidadosamente o seguinte conjunto de conversas de suporte ao cliente.
            Identifique os principais padrões e os motivos subjacentes dos contatos, destacando as categorias
            que melhor representam as causas dos problemas.

            Considere que cada ticket pode conter diversas conversas, mas procure evidenciar os tópicos
            que se repetem e que explicam o motivo principal do contato.

            CONVERSAS:
            {text}

            APRESENTE A ANÁLISE DE CATEGORIAS EM UM ÚNICO PARÁGRAFO:
        """
        )

        # Template para combinar as análises parciais
        self.combine_template = ChatPromptTemplate.from_template(
            """
            Combine as seguintes análises parciais e elabore uma visão consolidada das categorias de suporte.

            Identifique os padrões mais frequentes e os motivos subjacentes que geram os contatos,
            utilizando um vocabulário consistente.

            Forneça um único parágrafo que descreva as categorias consolidadas e os principais motivos
            (por exemplo: "Problema com Pagamento", "Erro no Sistema", "Dúvida de Reserva", etc.).

            ANÁLISES PARCIAIS:
            {text}

            APRESENTE A ANÁLISE CONSOLIDADA EM UM ÚNICO PARÁGRAFO:
        """
        )

        # Template para categorização final
        self.categorize_template = ChatPromptTemplate.from_template(
            """
            Categorize cada ticket fornecido com base na análise consolidada.

            REGRAS:
            1. Utilize EXATAMENTE o mesmo ticket_id que aparece após a palavra "Ticket".
            2. Atribua até 3 categorias por ticket, listadas em ordem de relevância, representando os motivos principais do contato.
            3. Priorize categorias específicas que descrevam de forma clara o motivo subjacente do contato, evitando termos genéricos como "Problemas Gerais" ou "Outros" ou "Outros Assuntos".
            4. Utilize um vocabulário consistente e padronizado, alinhado aos padrões identificados na análise consolidada.
            5. Se o ticket apresentar menções a termos ou contextos específicos (por exemplo, relacionados a antifraude, erros no sistema, problemas com pagamento, etc.), escolha a categoria que melhor represente essa especificidade.
            6. Cada categoria deve ser concisa (máximo 50 caracteres) e refletir claramente o motivo do contato.


            FORMATO JSON OBRIGATÓRIO:
            {{
              "cat": [
                {{"id": "123", "cat": ["Cat1", "Cat2", "Cat3"]}},
                {{"id": "456", "cat": ["Cat1", "Cat2"]}}
              ]
            }}

            ANÁLISE:
            {analysis}

            TICKETS:
            {tickets}

            RESPONDA APENAS COM O JSON.
        """
        )

        # Configura o splitter para chunks grandes com overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=1000000,  # Tamanho máximo de cada chunk
            chunk_overlap=10000,  # Overlap entre chunks
        )

    def process_tickets(self, input_file: Path, nrows: int = None) -> Path:
        """Processa os tickets usando uma abordagem map-reduce para categorização"""
        tickets = self.prepare_data(input_file, nrows=nrows)
        print(f"\nProcessando arquivo: {input_file}")
        print(f"Total de tickets para processar: {len(tickets)}")

        # Contadores de tokens
        total_input_tokens = 0
        total_output_tokens = 0

        # Prepara o texto completo
        full_text = "\n\n".join(
            f"Ticket {ticket['ticket_id']}:\n{ticket['text']}" for ticket in tickets
        )

        # Divide em chunks
        chunks = self.text_splitter.split_text(full_text)
        print(f"\nTexto dividido em {len(chunks)} chunks")

        # Converte chunks para Documents
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Configura as chains
        map_chain = (
            RunnablePassthrough() | self.map_template | self.llm | StrOutputParser()
        )

        combine_chain = (
            RunnablePassthrough() | self.combine_template | self.llm | StrOutputParser()
        )

        # 1. Map: Analisa cada chunk (agora com processamento paralelo)
        print("\nRealizando análise dos chunks em paralelo...")
        partial_analyses = []
        map_tokens_in = 0
        map_tokens_out = 0

        # Função para processar um chunk
        def process_chunk(doc_index, doc):
            try:
                # Conta tokens de entrada
                input_text = doc.page_content
                input_tokens = self.estimate_tokens(input_text)

                analysis = map_chain.invoke({"text": input_text})

                # Conta tokens de saída
                output_tokens = self.estimate_tokens(analysis)

                return {
                    "analysis": analysis,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "success": True,
                }
            except Exception as e:
                print(f"Erro ao processar chunk {doc_index}: {str(e)}")
                return {
                    "analysis": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "success": False,
                    "error": str(e),
                }

        # Processa os chunks em paralelo
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submete todos os chunks para processamento
            future_to_chunk = {
                executor.submit(process_chunk, i, doc): (i, doc)
                for i, doc in enumerate(docs, 1)
            }

            # Coleta os resultados à medida que são concluídos
            for future in tqdm(
                concurrent.futures.as_completed(future_to_chunk),
                total=len(docs),
                desc="Processando chunks",
            ):
                chunk_index, _ = future_to_chunk[future]
                try:
                    result = future.result()
                    if result["success"]:
                        partial_analyses.append(result["analysis"])
                        map_tokens_in += result["input_tokens"]
                        map_tokens_out += result["output_tokens"]
                        print(
                            f"Chunk {chunk_index}/{len(docs)} processado - Input: {result['input_tokens']:,}, Output: {result['output_tokens']:,}"
                        )
                    else:
                        print(
                            f"Falha no chunk {chunk_index}: {result.get('error', 'Erro desconhecido')}"
                        )
                except Exception as e:
                    print(f"Exceção ao processar chunk {chunk_index}: {str(e)}")

        total_input_tokens += map_tokens_in
        total_output_tokens += map_tokens_out
        print(
            f"\nTotal de tokens na fase Map - Input: {map_tokens_in:,}, Output: {map_tokens_out:,}"
        )

        # 2. Reduce: Combina as análises parciais
        print("\nCombinando análises parciais...")
        try:
            combine_input = "\n\n".join(partial_analyses)
            combine_tokens_in = self.estimate_tokens(combine_input)

            consolidated_analysis = combine_chain.invoke({"text": combine_input})

            combine_tokens_out = self.estimate_tokens(consolidated_analysis)
            total_input_tokens += combine_tokens_in
            total_output_tokens += combine_tokens_out

            print(
                f"Tokens na fase Combine - Input: {combine_tokens_in:,}, Output: {combine_tokens_out:,}"
            )

        except Exception as e:
            print(f"Erro ao combinar análises: {str(e)}")
            return None

        # 3. Categoriza os tickets usando a análise consolidada, processando em batches paralelos
        print("\nCategorizando tickets em paralelo...")
        try:
            # Resumo da análise consolidada para reduzir tokens
            print("\nResumindo análise consolidada...")
            summary_chain = (
                ChatPromptTemplate.from_template(
                    """
                Resuma a análise consolidada a seguir de forma concisa, mantendo apenas
                as informações essenciais sobre as principais categorias e padrões identificados:

                {text}

                RESUMO CONCISO:
            """
                )
                | self.llm
                | StrOutputParser()
            )

            analysis_summary = summary_chain.invoke({"text": consolidated_analysis})
            print("Análise resumida com sucesso.")

            # Configure o categorize_chain
            categorize_chain = self.categorize_template | self.llm | StrOutputParser()

            # Defina um tamanho de batch adequado
            batch_size = 220

            # Divida a lista de tickets em batches
            ticket_batches = [
                tickets[i : i + batch_size] for i in range(0, len(tickets), batch_size)
            ]
            print(f"Dividindo os tickets em {len(ticket_batches)} batches.")

            # Função para processar um batch
            def process_batch(batch_index, batch):
                try:
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
                        "success": True,
                    }

                except Exception as e:
                    # Salva log de erro
                    log_file = (
                        self.database_dir
                        / f"error_log_batch_{batch_index}_{int(time.time())}.json"
                    )

                    error_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "batch_index": batch_index,
                        "error_type": str(type(e)),
                        "error_message": str(e),
                        "raw_response": (
                            response if "response" in locals() else "No response"
                        ),
                        "extracted_json": (
                            json_str if "json_str" in locals() else "No JSON"
                        ),
                    }

                    with open(log_file, "w", encoding="utf-8") as f:
                        json.dump(error_data, f, ensure_ascii=False, indent=2)

                    print(f"Log de erro salvo em: {log_file}")

                    return {
                        "results": [],
                        "input_tokens": (
                            batch_input_tokens
                            if "batch_input_tokens" in locals()
                            else 0
                        ),
                        "output_tokens": (
                            batch_output_tokens
                            if "batch_output_tokens" in locals()
                            else 0
                        ),
                        "success": False,
                        "error": str(e),
                    }

            # Processa os batches em paralelo
            all_categorization_results = []
            categorize_tokens_in = 0
            categorize_tokens_out = 0

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submete todos os batches para processamento
                future_to_batch = {
                    executor.submit(process_batch, i, batch): (i, batch)
                    for i, batch in enumerate(ticket_batches, 1)
                }

                # Coleta os resultados à medida que são concluídos
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_batch),
                    total=len(ticket_batches),
                    desc="Processando batches",
                ):
                    batch_index, _ = future_to_batch[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            all_categorization_results.extend(result["results"])
                            categorize_tokens_in += result["input_tokens"]
                            categorize_tokens_out += result["output_tokens"]
                            print(
                                f"Batch {batch_index}/{len(ticket_batches)} processado: {len(result['results'])} tickets categorizados"
                            )
                        else:
                            print(
                                f"Falha no batch {batch_index}: {result.get('error', 'Erro desconhecido')}"
                            )
                    except Exception as e:
                        print(f"Exceção ao processar batch {batch_index}: {str(e)}")

            # Atualiza contadores globais de tokens
            total_input_tokens += categorize_tokens_in
            total_output_tokens += categorize_tokens_out
            print(
                f"\nTotal de tokens na fase Categorize - Input: {categorize_tokens_in:,}, Output: {categorize_tokens_out:,}"
            )

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

            # Exibe o resumo final de tokens
            print("\n=== Resumo do Uso de Tokens ===")
            print(f"Total de Tokens de Entrada: {total_input_tokens:,}")
            print(f"Total de Tokens de Saída: {total_output_tokens:,}")
            print(
                f"Total Geral de Tokens: {(total_input_tokens + total_output_tokens):,}"
            )

            # Estimativa de custo
            estimated_cost = (total_input_tokens + total_output_tokens) * 0.00025 / 1000
            print(f"Custo Estimado: ${estimated_cost:.2f}")

            return output_file

        except Exception as e:
            print(f"Erro ao categorizar tickets: {str(e)}")
            return None
