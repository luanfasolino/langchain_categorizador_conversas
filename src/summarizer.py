from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pandas as pd
from base_processor import BaseProcessor
import json
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnablePassthrough
import concurrent.futures
from tqdm import tqdm

class TicketSummarizer(BaseProcessor):
    def __init__(self, api_key: str, database_dir: Path, max_workers: int = None, use_cache: bool = True):
        super().__init__(api_key, database_dir, max_workers=max_workers, use_cache=use_cache)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        # Template para sumarização dos chunks
        self.map_template = ChatPromptTemplate.from_template("""
            Analise cuidadosamente o seguinte conjunto de conversas de suporte ao cliente.
            Identifique e resuma os principais problemas, dúvidas e padrões de contato observados.
            Mantenha informações relevantes sobre frequência e impacto dos problemas.
            Seja extremamente objetivo e direto.
            
            CONVERSAS:
            {text}
            
            RESUMO (em um único parágrafo):
        """)
        
        # Template para combinar os resumos parciais
        self.combine_template = ChatPromptTemplate.from_template("""
            Combine os seguintes resumos parciais em uma análise consolidada das conversas de suporte.
            Mantenha o foco nos problemas mais frequentes e significativos.
            Preserve informações sobre padrões e tendências importantes.
            O resultado deve ser um único parágrafo conciso e objetivo.
            
            RESUMOS PARCIAIS:
            {text}
            
            ANÁLISE CONSOLIDADA (em um único parágrafo):
        """)
        
        # Template para extração final dos bullet points
        self.bullet_template = ChatPromptTemplate.from_template("""
            Você é um especialista em análise de conversas de suporte ao cliente e em identificar oportunidades de melhoria para reduzir os contatos no call center.
            
            Analise o resumo consolidado abaixo e gere EXATAMENTE:
            1. 15 bullet points com sugestões práticas e diretas para solucionar os problemas identificados
            2. Um resumo geral conciso explicando os motivos centrais dos problemas
            
            REGRAS:
            - Cada bullet point deve ter no máximo 200 caracteres
            - O resumo deve ser um único parágrafo
            - Foque em ações práticas e objetivas
            - IMPORTANTE: Retorne APENAS um objeto JSON no formato especificado abaixo
            
            FORMATO OBRIGATÓRIO DA RESPOSTA:
            {{
              "bullets": [
                {{"bullet": "Primeira sugestão aqui"}},
                {{"bullet": "Segunda sugestão aqui"}},
                {{"bullet": "Terceira sugestão aqui"}},
                ... (exatamente 15 bullets)
              ],
              "resumo": "Seu resumo em um único parágrafo aqui"
            }}

            RESUMO PARA ANALISAR:
            {text}
            
            RESPONDA APENAS COM O JSON NO FORMATO ESPECIFICADO.
        """)
        
        # Configura o splitter para chunks grandes com overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=900000,  # Tamanho máximo de cada chunk
            chunk_overlap=90000  # Overlap entre chunks
        )
        
    def process_tickets(self, input_file: Path, nrows: int = None) -> Path:
        """Processa os tickets usando uma abordagem map-reduce para sumarização"""
        tickets = self.prepare_data(input_file, nrows=nrows)
        print(f"\nProcessando arquivo: {input_file}")
        print(f"Total de tickets para processar: {len(tickets)}")
        
        # Prepara o texto completo
        full_text = "\n\n".join(
            f"Ticket {ticket['ticket_id']}:\n{ticket['text']}"
            for ticket in tickets
        )
        
        # Divide em chunks
        chunks = self.text_splitter.split_text(full_text)
        print(f"\nTexto dividido em {len(chunks)} chunks")
        
        # Converte chunks para Documents
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Configura as chains de sumarização usando o novo formato
        map_chain = (
            RunnablePassthrough() 
            | self.map_template 
            | self.llm 
            | StrOutputParser()
        )
        
        combine_chain = (
            RunnablePassthrough() 
            | self.combine_template 
            | self.llm 
            | StrOutputParser()
        )
        
        # 1. Map: Sumariza cada chunk (agora com processamento paralelo)
        print("\nRealizando sumarização dos chunks em paralelo...")
        partial_summaries = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Função para processar um chunk
        def process_chunk(doc_index, doc):
            try:
                input_text = doc.page_content
                input_tokens = self.estimate_tokens(input_text)
                
                summary = map_chain.invoke({"text": input_text})
                
                output_tokens = self.estimate_tokens(summary)
                
                return {
                    'summary': summary,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'success': True
                }
            except Exception as e:
                print(f"Erro ao processar chunk {doc_index}: {str(e)}")
                return {
                    'summary': None,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Processa os chunks em paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submete todos os chunks para processamento
            future_to_chunk = {
                executor.submit(process_chunk, i, doc): (i, doc) 
                for i, doc in enumerate(docs, 1)
            }
            
            # Coleta os resultados à medida que são concluídos
            for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                              total=len(docs), 
                              desc="Processando chunks"):
                chunk_index, _ = future_to_chunk[future]
                try:
                    result = future.result()
                    if result['success']:
                        partial_summaries.append(result['summary'])
                        total_input_tokens += result['input_tokens']
                        total_output_tokens += result['output_tokens']
                        print(f"Chunk {chunk_index}/{len(docs)} processado - Input: {result['input_tokens']:,}, Output: {result['output_tokens']:,}")
                    else:
                        print(f"Falha no chunk {chunk_index}: {result.get('error', 'Erro desconhecido')}")
                except Exception as e:
                    print(f"Exceção ao processar chunk {chunk_index}: {str(e)}")
        
        # 2. Reduce: Combina os resumos parciais
        print("\nCombinando resumos parciais...")
        try:
            combine_input = "\n\n".join(partial_summaries)
            combine_tokens_in = self.estimate_tokens(combine_input)
            total_input_tokens += combine_tokens_in
            
            consolidated_summary = combine_chain.invoke({"text": combine_input})
            
            combine_tokens_out = self.estimate_tokens(consolidated_summary)
            total_output_tokens += combine_tokens_out
            
            print(f"Tokens na fase Combine - Input: {combine_tokens_in:,}, Output: {combine_tokens_out:,}")
            
        except Exception as e:
            print(f"Erro ao combinar resumos: {str(e)}")
            return None
        
        # 3. Extrai bullet points do resumo consolidado
        print("\nExtraindo bullet points e resumo...")
        try:
            bullet_chain = self.bullet_template | self.llm | StrOutputParser()
            
            bullet_input = {"text": consolidated_summary}
            bullet_tokens_in = self.estimate_tokens(consolidated_summary)
            total_input_tokens += bullet_tokens_in
            
            response = bullet_chain.invoke(bullet_input)
            
            bullet_tokens_out = self.estimate_tokens(response)
            total_output_tokens += bullet_tokens_out
            
            print(f"Tokens na fase Bullet - Input: {bullet_tokens_in:,}, Output: {bullet_tokens_out:,}")
            
            print("\nResposta recebida do modelo:")
            print(response[:500] + "..." if len(response) > 500 else response)  # Debug
            
            json_str = self.extract_json(response)
            print("\nJSON extraído:")
            print(json_str[:500] + "..." if len(json_str) > 500 else json_str)  # Debug
            
            # Validação e normalização do JSON
            results = json.loads(json_str)
            if not isinstance(results, dict) or 'bullets' not in results or 'resumo' not in results:
                print("Erro: Resposta não está no formato esperado")
                return None
            
            # Valida e normaliza os bullet points
            bullets = results['bullets']
            if not isinstance(bullets, list):
                print("Erro: bullets não é uma lista")
                return None
            
            valid_bullets = []
            for item in bullets:
                if not isinstance(item, dict) or 'bullet' not in item:
                    continue
                valid_bullets.append({
                    'bullet': item['bullet'].strip()
                })
            
            if len(valid_bullets) != 15:
                print(f"Aviso: Número de bullet points ({len(valid_bullets)}) diferente do esperado (15)")
            
            if not valid_bullets:
                print("\nAviso: Nenhum bullet point válido foi gerado!")
                return None
            
            # Valida o resumo
            resumo = results['resumo'].strip()
            if not resumo:
                print("\nAviso: Resumo está vazio!")
                return None
            
            print(f"\nResultados obtidos:")
            print(f"- {len(valid_bullets)} bullet points")
            print(f"- Resumo com {len(resumo)} caracteres")
            print("\nExemplo do primeiro bullet point:", valid_bullets[0]['bullet'])
            print("\nResumo:", resumo)
            
            # Exibe o resumo final de tokens
            print("\n=== Resumo do Uso de Tokens ===")
            print(f"Total de Tokens de Entrada: {total_input_tokens:,}")
            print(f"Total de Tokens de Saída: {total_output_tokens:,}")
            print(f"Total Geral de Tokens: {(total_input_tokens + total_output_tokens):,}")
            
            # Estimativa de custo
            estimated_cost = (total_input_tokens + total_output_tokens) * 0.00025 / 1000
            print(f"Custo Estimado: ${estimated_cost:.2f}")
            
            # Salva resultados
            output_data = {
                'bullets': valid_bullets,
                'resumo': resumo
            }
            
            # Salva em dois formatos: CSV para bullets e TXT para o resumo
            output_file_csv = self.database_dir / "summarized_tickets.csv"
            output_file_txt = self.database_dir / "summarized_tickets_resumo.txt"
            
            # Salva bullets em CSV
            bullets_df = pd.DataFrame(valid_bullets)
            bullets_df.to_csv(output_file_csv, sep=';', index=False, encoding='utf-8-sig')
            
            # Salva resumo em TXT
            with open(output_file_txt, 'w', encoding='utf-8') as f:
                f.write(resumo)
            
            print(f"\nResultados salvos em:")
            print(f"- Bullets: {output_file_csv}")
            print(f"- Resumo: {output_file_txt}")
            
            return output_file_csv
            
        except Exception as e:
            print(f"Erro ao extrair bullet points e resumo: {str(e)}")
            return None
        
        return output_file_csv 