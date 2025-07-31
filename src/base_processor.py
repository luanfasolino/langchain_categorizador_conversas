from pathlib import Path
import json
import time
import re
import pandas as pd
from typing import List, Tuple, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
import os
import concurrent.futures
from tqdm import tqdm
import hashlib
import pickle


class BaseProcessor:
    def __init__(
        self,
        api_key: str,
        database_dir: Path,
        max_tickets_per_batch: int = 50,
        max_workers: int = None,
        use_cache: bool = True,
    ):
        self.database_dir = database_dir
        self.max_tickets_per_batch = max_tickets_per_batch
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.max_workers = max_workers or min(
            os.cpu_count(), 4
        )  # Limita ao número de CPUs ou 4, o que for menor
        self.use_cache = use_cache

        # Cria diretório de cache se não existir
        self.cache_dir = self.database_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Inicializa o modelo
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key
        )

    def estimate_tokens(self, text: str) -> int:
        """Estima o número de tokens assumindo ~1 token a cada 4 caracteres."""
        return len(text) // 4

    def clean_text(self, text: str) -> str:
        """Limpa o texto removendo ou substituindo caracteres que podem causar problemas."""
        # Remove mensagens padrão de atendimento
        text = re.sub(
            r"Estamos te conectando a um especialista humano.*?:?\)\s*", "", text
        )
        text = re.sub(
            r"Olá,?\s+[^,]+,\s+sou\s+[^,]+\s+e\s+já\s+estou\s+aqui\.?\s*", "", text
        )

        # Remove asteriscos usados para ênfase
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)

        # Remove caracteres de controle e caracteres especiais problemáticos
        text = "".join(char for char in text if ord(char) >= 32 or char == "\n")

        # Substitui aspas tipográficas por aspas simples
        text = (
            text.replace('"', '"').replace('"', '"').replace(""", "'").replace(""", "'")
        )

        # Trata URLs - substitui por uma representação mais segura
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            lambda m: "[URL: " + m.group(0).split("/")[-1] + "]",
            text,
        )

        # Remove múltiplas linhas em branco
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remove espaços em branco no início e fim de cada linha
        text = "\n".join(line.strip() for line in text.split("\n"))

        # Remove linhas vazias no início e fim do texto
        text = text.strip()

        # Aplica as substituições existentes
        replacements = {
            "\\": " ",
            "\r": " ",
            "\t": " ",
            "|": " ",
            "\u2028": " ",  # line separator
            "\u2029": " ",  # paragraph separator
            "…": "...",  # reticências
            "—": "-",  # travessão
            "–": "-",  # hífen
            '"': '"',  # aspas curvas
            '"': '"',  # aspas curvas
            """: "'",       # aspas simples
            """: "'",  # aspas simples
            "´": "'",  # acento agudo
            "`": "'",  # crase
            "•": "*",  # bullet point
            "○": "*",  # círculo
            "●": "*",  # círculo preenchido
            "□": "*",  # quadrado
            "■": "*",  # quadrado preenchido
            "►": ">",  # seta
            "▼": "v",  # seta
            "▲": "^",  # seta
            "◄": "<",  # seta
            "*[": "[",  # remove asterisco antes de colchete
            "]*": "]",  # remove asterisco depois de colchete
            "‎": "",  # caractere invisível de formatação (LEFT-TO-RIGHT MARK)
            "‏": "",  # caractere invisível de formatação (RIGHT-TO-LEFT MARK)
            "️": "",  # variation selector
            "⃣": "",  # combining enclosing keycap
            "⭐": "*",  # estrela
            "✨": "*",  # brilhos
            "✅": "v",  # check mark
            "❌": "x",  # x mark
            "❗": "!",  # exclamação
            "❓": "?",  # interrogação
            "【": "[",  # colchetes estilizados
            "】": "]",  # colchetes estilizados
            "［": "[",  # colchetes largos
            "］": "]",  # colchetes largos
            "（": "(",  # parênteses estilizados
            "）": ")",  # parênteses estilizados
            "﻿": "",  # BOM (Byte Order Mark)
            "\u200e": "",  # LEFT-TO-RIGHT MARK
            "\u200f": "",  # RIGHT-TO-LEFT MARK
            "\u202a": "",  # LEFT-TO-RIGHT EMBEDDING
            "\u202b": "",  # RIGHT-TO-LEFT EMBEDDING
            "\u202c": "",  # POP DIRECTIONAL FORMATTING
            "\u202d": "",  # LEFT-TO-RIGHT OVERRIDE
            "\u202e": "",  # RIGHT-TO-LEFT OVERRIDE
            "＋": "+",  # FULLWIDTH PLUS
            "．": ".",  # FULLWIDTH STOP
            "，": ",",  # FULLWIDTH COMMA
            "：": ":",  # FULLWIDTH COLON
            "／": "/",  # FULLWIDTH SOLIDUS
            "（": "(",  # FULLWIDTH LEFT PARENTHESIS
            "）": ")",  # FULLWIDTH RIGHT PARENTHESIS
            "［": "[",  # FULLWIDTH LEFT SQUARE BRACKET
            "］": "]",  # FULLWIDTH RIGHT SQUARE BRACKET
            "＊": "*",  # FULLWIDTH ASTERISK
            "＿": "_",  # FULLWIDTH LOW LINE
            "～": "~",  # FULLWIDTH TILDE
            "！": "!",  # FULLWIDTH EXCLAMATION MARK
            "？": "?",  # FULLWIDTH QUESTION MARK
            "；": ";",  # FULLWIDTH SEMICOLON
            "\u2000": " ",  # EN QUAD
            "\u2001": " ",  # EM QUAD
            "\u2002": " ",  # EN SPACE
            "\u2003": " ",  # EM SPACE
            "\u2004": " ",  # THREE-PER-EM SPACE
            "\u2005": " ",  # FOUR-PER-EM SPACE
            "\u2006": " ",  # SIX-PER-EM SPACE
            "\u2007": " ",  # FIGURE SPACE
            "\u2008": " ",  # PUNCTUATION SPACE
            "\u2009": " ",  # THIN SPACE
            "\u200a": " ",  # HAIR SPACE
            "\u202f": " ",  # NARROW NO-BREAK SPACE
            "\u205f": " ",  # MEDIUM MATHEMATICAL SPACE
            "\u3000": " ",  # IDEOGRAPHIC SPACE
            "\x00": "",  # NULL
            "\x01": "",  # START OF HEADING
            "\x02": "",  # START OF TEXT
            "\x03": "",  # END OF TEXT
            "\x04": "",  # END OF TRANSMISSION
            "\x05": "",  # ENQUIRY
            "\x06": "",  # ACKNOWLEDGE
            "\x07": "",  # BELL
            "\x08": "",  # BACKSPACE
            "\x0e": "",  # SHIFT OUT
            "\x0f": "",  # SHIFT IN
            "\x10": "",  # DATA LINK ESCAPE
            "\x11": "",  # DEVICE CONTROL ONE
            "\x12": "",  # DEVICE CONTROL TWO
            "\x13": "",  # DEVICE CONTROL THREE
            "\x14": "",  # DEVICE CONTROL FOUR
            "\x15": "",  # NEGATIVE ACKNOWLEDGE
            "\x16": "",  # SYNCHRONOUS IDLE
            "\x17": "",  # END OF TRANSMISSION BLOCK
            "\x18": "",  # CANCEL
            "\x19": "",  # END OF MEDIUM
            "\x1a": "",  # SUBSTITUTE
            "\x1b": "",  # ESCAPE
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Remove caracteres não-ASCII que possam ter sobrado
        text = "".join(char for char in text if ord(char) < 128)

        # Remove múltiplos espaços em branco
        text = " ".join(text.split())

        # Remove espaços antes de pontuação
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text

    def extract_json(self, response: str) -> str:
        """Extrai o JSON válido da resposta."""
        # Remove markdown se presente
        if "```json" in response:
            response = response.split("```json")[-1]
            if "```" in response:
                response = response.split("```")[0]

        # Tenta encontrar o objeto JSON completo
        try:
            # Procura por { no início e } no fim
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start : end + 1]
                # Tenta validar se é JSON válido
                json.loads(json_str)
                return json_str
        except json.JSONDecodeError:
            # Se falhou, tenta limpar caracteres problemáticos
            cleaned = response.replace("\n", " ").replace("\r", " ")
            cleaned = re.sub(r"\s+", " ", cleaned)

            # Tenta novamente encontrar JSON válido
            try:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1:
                    json_str = cleaned[start : end + 1]
                    json.loads(json_str)  # Valida
                    return json_str
            except json.JSONDecodeError:
                pass

        # Se não conseguiu extrair JSON válido, retorna a resposta limpa
        return response.strip()

    def process_batches(
        self, chain, tickets: list, token_limit: int = 50000
    ) -> Tuple[list, int, int]:
        """Processa os tickets em batches."""
        results = []
        batch_count = 0

        # Buffer para acumular tickets
        batch_tickets = []
        batch_tokens = 0
        header = "Tickets:\n"
        header_tokens = self.estimate_tokens(header)
        batch_tokens += header_tokens

        # Estima total de batches
        estimated_total_batches = (
            sum(len(t["text"]) // 4 for t in tickets) // token_limit
        ) + 1
        print(f"\nTotal estimado de batches: {estimated_total_batches}")

        def process_batch(tickets_text, retry_count=0):
            max_retries = 3
            try:
                input_tokens = self.estimate_tokens(tickets_text)
                self.total_input_tokens += input_tokens

                response = chain.invoke({"tickets_text": tickets_text})
                output_tokens = self.estimate_tokens(response)
                self.total_output_tokens += output_tokens

                print(
                    f"Tokens estimados neste batch - Input: {input_tokens:,}, Output: {output_tokens:,}"
                )

                json_str = self.extract_json(response)
                try:
                    batch_result = json.loads(json_str)

                    # Adiciona logs para rastrear tickets processados vs retornados
                    tickets_no_batch = (
                        len(tickets_text.split("Ticket ")) - 1
                    )  # -1 para compensar o header
                    tickets_processados = len(batch_result)

                    if tickets_no_batch != tickets_processados:
                        print(
                            f"\nAtenção: Batch processou {tickets_processados} de {tickets_no_batch} tickets"
                        )
                        # Identifica quais tickets não foram processados
                        ticket_ids_input = re.findall(r"Ticket (\d+):", tickets_text)
                        ticket_ids_output = [r.get("ticket_id") for r in batch_result]
                        missing_tickets = set(ticket_ids_input) - set(ticket_ids_output)
                        print(f"Tickets não processados neste batch: {missing_tickets}")

                    # Valida a estrutura do resultado
                    valid_results = []
                    for item in batch_result:
                        if not isinstance(item, dict):
                            print(f"Aviso: Item inválido no resultado: {item}")
                            continue
                        if "ticket_id" not in item:
                            print(f"Aviso: Item sem ticket_id: {item}")
                            continue
                        if not item.get("categorias") or not item.get("resumo"):
                            print(f"Aviso: Item sem categorias ou resumo: {item}")
                            continue
                        valid_results.append(item)

                    if len(valid_results) != len(batch_result):
                        print(
                            f"Aviso: {len(batch_result) - len(valid_results)} resultados foram filtrados por serem inválidos"
                        )

                    return valid_results

                except json.JSONDecodeError as json_error:
                    # Identifica o contexto do erro
                    error_position = json_error.pos
                    context_start = max(0, error_position - 100)
                    context_end = min(len(json_str), error_position + 100)
                    error_context = json_str[context_start:context_end]

                    # Salva o log com informações detalhadas
                    log_file = self.database_dir / f"error_log_{int(time.time())}.log"
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("=== Erro de Processamento ===\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Tentativa: {retry_count + 1} de {max_retries}\n")
                        f.write(f"Erro: {str(json_error)}\n")
                        f.write("\n=== Contexto do Erro ===\n")
                        f.write(f"Posição do erro: {error_position}\n")
                        f.write(f"Trecho problemático:\n{error_context}\n")
                        f.write("\n=== Resposta Completa da LLM ===\n")
                        f.write(response)
                        f.write("\n\n=== JSON Extraído ===\n")
                        f.write(json_str)
                        f.write("\n\n=== Input do Batch ===\n")
                        f.write(tickets_text)

                    print(f"Log detalhado do erro salvo em: {log_file}")
                    raise

            except Exception as e:
                # Salva o log quando houver erro
                log_file = self.database_dir / f"error_log_{int(time.time())}.log"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("=== Erro de Processamento ===\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Tentativa: {retry_count + 1} de {max_retries}\n")
                    f.write(f"Erro: {str(e)}\n")
                    f.write(f"Tipo do erro: {type(e)}\n")

                    if isinstance(e, json.JSONDecodeError):
                        f.write("\n=== Detalhes do Erro JSON ===\n")
                        f.write(f"Posição do erro: {e.pos}\n")
                        f.write(f"Linha: {e.lineno}, Coluna: {e.colno}\n")
                        context_start = max(0, e.pos - 100)
                        context_end = min(len(e.doc), e.pos + 100)
                        f.write(
                            f"Contexto do erro:\n{e.doc[context_start:context_end]}\n"
                        )

                    f.write("\n=== Resposta da LLM ===\n")
                    if "response" in locals():
                        f.write(str(response))
                    else:
                        f.write("(Erro ocorreu antes de obter resposta)")

                    f.write("\n\n=== Input do Batch ===\n")
                    f.write(tickets_text)

                print(f"Log detalhado do erro salvo em: {log_file}")

                if retry_count < max_retries:
                    print(
                        f"Erro no processamento, tentativa {retry_count + 1} de {max_retries}"
                    )
                    time.sleep(1)
                    return process_batch(tickets_text, retry_count + 1)
                else:
                    print(f"Erro após {max_retries} tentativas:", str(e))
                    return []

        for ticket in tickets:
            batch_tickets.append(ticket["text"])
            batch_tokens += self.estimate_tokens(ticket["text"])

            if (
                batch_tokens >= token_limit
                or len(batch_tickets) >= self.max_tickets_per_batch
            ):
                batch_result = process_batch(" ".join(batch_tickets))
                if batch_result:
                    results.append(batch_result)
                batch_tickets = []
                batch_tokens = 0
                batch_count += 1
                print(f"Batch {batch_count} processado")

        if batch_tickets:
            batch_result = process_batch(" ".join(batch_tickets))
            if batch_result:
                results.append(batch_result)
            batch_count += 1
            print(f"Batch {batch_count} processado")

        return results, batch_count, self.total_input_tokens, self.total_output_tokens

    def _generate_cache_key(self, data: Any) -> str:
        """Gera uma chave de cache baseada nos dados de entrada."""
        # Converte os dados para string e gera um hash
        if isinstance(data, dict):
            # Ordena as chaves para garantir consistência
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            # Para listas, converte cada item para string e junta
            data_str = json.dumps([str(item) for item in data], sort_keys=True)
        else:
            # Para outros tipos, converte diretamente para string
            data_str = str(data)

        # Gera o hash SHA-256
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Any:
        """Recupera dados do cache se existirem."""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar cache: {str(e)}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Salva dados no cache."""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Erro ao salvar cache: {str(e)}")

    def prepare_data(self, input_file: Path, nrows: int = None) -> List[dict]:
        """Prepara os dados do arquivo de entrada com suporte a cache e validação aprimorada."""
        # Gera chave de cache baseada no arquivo e número de linhas
        cache_key = self._generate_cache_key(
            {"file": str(input_file), "nrows": nrows, "method": "prepare_data"}
        )

        # Tenta recuperar do cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            print(f"Usando dados em cache para {input_file}")
            return cached_data

        # Se não estiver em cache, processa normalmente
        print(f"Processando dados de {input_file} (não encontrado em cache)")

        # Carrega dados com tratamento robusto de arquivos
        df = self._load_file_robust(input_file, nrows)
        raw_count = len(df)
        print(f"📊 Dados brutos carregados: {raw_count:,} registros")

        # Validação e filtragem melhorada por category
        df = self._filter_by_category(df)
        category_filtered_count = len(df)
        print(f"✅ Após filtro category='TEXT': {category_filtered_count:,} registros")

        # Limpeza e preparação de campos
        df = self._prepare_fields(df)

        # Filtragem aprimorada de mensagens AI
        df_filtered = self._filter_ai_messages(df)
        ai_filtered_count = len(df_filtered)
        print(f"🤖 Após remover mensagens AI: {ai_filtered_count:,} registros")

        # Validação de contagem mínima de mensagens
        df_filtered, valid_tickets = self._validate_message_counts(df_filtered)
        final_count = len(df_filtered)
        print(
            f"💬 Tickets válidos (2+ USER e 2+ AGENT): {len(valid_tickets):,} tickets, {final_count:,} mensagens"
        )

        # Agrupa e finaliza
        grouped = self._group_by_ticket(df_filtered)
        result = grouped.to_dict(orient="records")

        # Gera relatório de filtragem
        self._generate_filtering_report(
            raw_count,
            category_filtered_count,
            ai_filtered_count,
            final_count,
            len(valid_tickets),
            len(result),
        )

        # Salva no cache
        self._save_to_cache(cache_key, result)

        return result

    def _load_file_robust(self, input_file: Path, nrows: int = None) -> pd.DataFrame:
        """Carrega arquivo com tratamento robusto para diferentes formatos e encodings."""
        if input_file.suffix.lower() == ".csv":
            print(
                "📄 Detectado arquivo CSV, tentando carregar com diferentes configurações..."
            )

            # Lista de configurações para tentar
            csv_configs = [
                {"encoding": "utf-8-sig", "sep": ";"},
                {"encoding": "utf-8", "sep": ";"},
                {"encoding": "latin-1", "sep": ";"},
                {"encoding": "utf-8-sig", "sep": ","},
                {"encoding": "utf-8", "sep": ","},
                {"encoding": "latin-1", "sep": ","},
            ]

            for i, config in enumerate(csv_configs):
                try:
                    df = pd.read_csv(
                        input_file,
                        nrows=nrows,
                        quotechar='"',
                        escapechar="\\",
                        on_bad_lines="skip",
                        low_memory=False,
                        dtype=str,
                        **config,
                    )
                    print(
                        f"✅ CSV carregado com sucesso: encoding={config['encoding']}, sep='{config['sep']}'"
                    )
                    print(f"   Colunas detectadas: {list(df.columns)}")
                    return df
                except Exception as e:
                    print(f"   Tentativa {i+1} falhou ({config}): {str(e)}")
                    if i == len(csv_configs) - 1:
                        raise Exception(
                            f"Não foi possível carregar o arquivo CSV após {len(csv_configs)} tentativas"
                        )
        else:
            print("📊 Detectado arquivo Excel, usando pd.read_excel()")
            try:
                df = pd.read_excel(input_file, nrows=nrows, dtype=str)
                print("✅ Excel carregado com sucesso")
                print(f"   Colunas detectadas: {list(df.columns)}")
                return df
            except Exception as e:
                raise Exception(f"Erro ao carregar arquivo Excel: {str(e)}")

    def _filter_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra registros por category='TEXT' com validação robusta case-insensitive."""
        if "category" not in df.columns:
            raise ValueError("Coluna 'category' não encontrada no arquivo")

        # Converte para string e trata valores nulos
        df["category"] = df["category"].fillna("").astype(str)

        # Filtragem case-insensitive mais robusta
        df_filtered = df[df["category"].str.lower().str.strip() == "text"]

        # Log das categorias encontradas para debug
        categories_found = df["category"].value_counts().head(10)
        print("📋 Top 10 categorias encontradas:")
        for cat, count in categories_found.items():
            print(f"   '{cat}': {count:,} registros")

        return df_filtered

    def _prepare_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara e limpa campos necessários."""
        required_fields = ["text", "ticket_id", "sender"]
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Campos obrigatórios não encontrados: {missing_fields}")

        # Converte e limpa campos
        df["text"] = df["text"].fillna("").astype(str).apply(self.clean_text)
        df["ticket_id"] = df["ticket_id"].fillna("").astype(str)
        df["sender"] = df["sender"].fillna("").astype(str)

        # Remove registros com campos vazios
        initial_count = len(df)
        df = df[
            (df["text"].str.strip() != "")
            & (df["ticket_id"].str.strip() != "")
            & (df["sender"].str.strip() != "")
        ]
        final_count = len(df)

        if initial_count != final_count:
            print(
                f"⚠️  Removidos {initial_count - final_count} registros com campos vazios"
            )

        return df

    def _filter_ai_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra mensagens AI com detecção aprimorada de padrões."""
        # Padrões de sender que indicam mensagens AI
        ai_patterns = [
            "ai",
            "AI",
            "Ai",
            "aI",
            "bot",
            "BOT",
            "Bot",
            "BoT",
            "assistant",
            "ASSISTANT",
            "Assistant",
            "ai_assistant",
            "AI_ASSISTANT",
            "chatbot",
            "CHATBOT",
            "ChatBot",
            "automated",
            "AUTOMATED",
            "Automated",
            "system",
            "SYSTEM",
            "System",
            "auto",
            "AUTO",
            "Auto",
        ]

        # Log das categorias de sender encontradas
        sender_counts = df["sender"].value_counts()
        print("📨 Tipos de sender encontrados:")
        for sender, count in sender_counts.items():
            print(f"   '{sender}': {count:,} mensagens")

        # Filtra usando padrões mais robustos
        ai_mask = (
            df["sender"].str.lower().str.strip().isin([p.lower() for p in ai_patterns])
        )
        ai_messages_count = ai_mask.sum()

        df_filtered = df[~ai_mask]

        print(f"🤖 Mensagens AI removidas: {ai_messages_count:,}")
        print(f"   Padrões detectados: {df[ai_mask]['sender'].unique().tolist()}")

        return df_filtered

    def _validate_message_counts(self, df: pd.DataFrame) -> tuple:
        """Valida contagem mínima de mensagens por ticket com relatórios detalhados."""
        # Conta mensagens por tipo de sender
        user_messages = df[df["sender"] == "USER"]["ticket_id"].value_counts()
        agent_messages = df[df["sender"].isin(["AGENT", "HELPDESK_INTEGRATION"])][
            "ticket_id"
        ].value_counts()

        # Estatísticas detalhadas
        print("📊 Estatísticas de mensagens por ticket:")
        if len(user_messages) > 0:
            print(
                f"   USER - Média: {user_messages.mean():.1f}, Mediana: {user_messages.median():.1f}, Máx: {user_messages.max()}"
            )
        if len(agent_messages) > 0:
            print(
                f"   AGENT - Média: {agent_messages.mean():.1f}, Mediana: {agent_messages.median():.1f}, Máx: {agent_messages.max()}"
            )

        # Identifica tickets válidos (pelo menos 2 mensagens de cada tipo)
        valid_user_tickets = set(user_messages[user_messages >= 2].index)
        valid_agent_tickets = set(agent_messages[agent_messages >= 2].index)
        valid_tickets = valid_user_tickets.intersection(valid_agent_tickets)

        print(f"✅ Tickets com 2+ mensagens USER: {len(valid_user_tickets):,}")
        print(f"✅ Tickets com 2+ mensagens AGENT: {len(valid_agent_tickets):,}")
        print(f"✅ Tickets válidos (ambos critérios): {len(valid_tickets):,}")

        # Filtra DataFrame
        df_filtered = df[df["ticket_id"].isin(valid_tickets)]

        return df_filtered, valid_tickets

    def _group_by_ticket(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrupa mensagens por ticket com preservação de metadados."""
        grouped = (
            df.groupby("ticket_id")
            .agg(
                {
                    "text": lambda x: " ".join(filter(None, x)),
                    "ticket_created_at": "first",
                    "sender": lambda x: f"Total: {len(x)} mensagens ({x.value_counts().to_dict()})",
                }
            )
            .reset_index()
        )

        # Remove tickets sem texto
        initial_count = len(grouped)
        grouped = grouped[grouped["text"].str.strip() != ""]
        final_count = len(grouped)

        if initial_count != final_count:
            print(
                f"⚠️  Removidos {initial_count - final_count} tickets sem texto válido"
            )

        # Estatísticas finais
        if len(grouped) > 0:
            text_lengths = grouped["text"].str.len()
            print("📝 Estatísticas de texto por ticket:")
            print(f"   Comprimento médio: {text_lengths.mean():.0f} caracteres")
            print(f"   Comprimento mediano: {text_lengths.median():.0f} caracteres")
            print(f"   Comprimento máximo: {text_lengths.max():,} caracteres")

        return grouped

    def _generate_filtering_report(
        self,
        raw_count: int,
        category_count: int,
        ai_count: int,
        final_count: int,
        valid_tickets: int,
        grouped_count: int,
    ):
        """Gera relatório detalhado do processo de filtragem."""
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO DE FILTRAGEM DE DADOS")
        print("=" * 60)
        print(f"Registros brutos:                    {raw_count:,}")
        print(
            f"Após filtro category='TEXT':         {category_count:,} ({category_count/raw_count*100:.1f}%)"
        )
        print(
            f"Após remoção mensagens AI:           {ai_count:,} ({ai_count/raw_count*100:.1f}%)"
        )
        print(
            f"Após validação contagem mensagens:   {final_count:,} ({final_count/raw_count*100:.1f}%)"
        )
        print(f"Tickets válidos únicos:              {valid_tickets:,}")
        print(f"Tickets finais agrupados:            {grouped_count:,}")
        print(
            f"\nTaxa de aproveitamento final:        {grouped_count/raw_count*100:.1f}%"
        )
        print("=" * 60)

    def generate_data_quality_report(
        self, tickets: List[dict], output_file: Path = None
    ) -> dict:
        """Gera relatório detalhado de qualidade dos dados processados."""
        if not tickets:
            print("⚠️  Nenhum ticket para analisar")
            return {}

        # Converte para DataFrame para análise
        df = pd.DataFrame(tickets)

        # Calcula estatísticas básicas
        ticket_count = len(df)

        # Estatísticas de texto
        text_lengths = df["text"].str.len()
        text_word_counts = df["text"].str.split().str.len()

        # Estatísticas por características do ticket
        has_date = "ticket_created_at" in df.columns
        date_stats = {}
        if has_date:
            df["ticket_created_at"] = pd.to_datetime(
                df["ticket_created_at"], errors="coerce"
            )
            date_stats = {
                "data_mais_antiga": df["ticket_created_at"].min(),
                "data_mais_recente": df["ticket_created_at"].max(),
                "periodo_dias": (
                    (df["ticket_created_at"].max() - df["ticket_created_at"].min()).days
                    if pd.notna(df["ticket_created_at"].min())
                    else 0
                ),
            }

        # Compila relatório
        quality_report = {
            "resumo_geral": {
                "total_tickets": ticket_count,
                "tickets_validos": ticket_count,
                "taxa_sucesso": 100.0,
            },
            "estatisticas_texto": {
                "comprimento_caracteres": {
                    "media": float(text_lengths.mean()),
                    "mediana": float(text_lengths.median()),
                    "minimo": int(text_lengths.min()),
                    "maximo": int(text_lengths.max()),
                    "desvio_padrao": float(text_lengths.std()),
                },
                "contagem_palavras": {
                    "media": float(text_word_counts.mean()),
                    "mediana": float(text_word_counts.median()),
                    "minimo": int(text_word_counts.min()),
                    "maximo": int(text_word_counts.max()),
                    "desvio_padrao": float(text_word_counts.std()),
                },
            },
            "distribuicao_mensagens": self._analyze_message_distribution(df),
            "qualidade_dados": self._assess_data_quality(df),
            "periodo_temporal": date_stats,
        }

        # Gera relatório visual
        self._print_quality_report(quality_report)

        # Salva em arquivo se especificado
        if output_file:
            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 Relatório salvo em: {output_file}")

        return quality_report

    def _analyze_message_distribution(self, df: pd.DataFrame) -> dict:
        """Analisa distribuição de mensagens por ticket."""
        if "sender" in df.columns:
            # Extrai informações de contagem de mensagens do campo sender
            sender_info = df["sender"].str.extract(r"Total: (\d+) mensagens \((.+)\)")
            message_counts = pd.to_numeric(sender_info[0], errors="coerce").dropna()

            if len(message_counts) > 0:
                return {
                    "mensagens_por_ticket": {
                        "media": float(message_counts.mean()),
                        "mediana": float(message_counts.median()),
                        "minimo": int(message_counts.min()),
                        "maximo": int(message_counts.max()),
                        "desvio_padrao": float(message_counts.std()),
                    },
                    "distribuicao_quartis": {
                        "q1": float(message_counts.quantile(0.25)),
                        "q2": float(message_counts.quantile(0.5)),
                        "q3": float(message_counts.quantile(0.75)),
                    },
                }

        return {"erro": "Informações de distribuição não disponíveis"}

    def _assess_data_quality(self, df: pd.DataFrame) -> dict:
        """Avalia qualidade geral dos dados."""
        quality_metrics = {
            "completude": {
                "tickets_com_texto": (df["text"].str.strip() != "").sum()
                / len(df)
                * 100,
                "tickets_com_id": (df["ticket_id"].str.strip() != "").sum()
                / len(df)
                * 100,
            },
            "consistencia": {
                "ids_unicos": df["ticket_id"].nunique() == len(df),
                "textos_nao_vazios": (df["text"].str.len() > 10).sum() / len(df) * 100,
            },
            "outliers": {
                "textos_muito_longos": (
                    df["text"].str.len() > df["text"].str.len().quantile(0.95)
                ).sum(),
                "textos_muito_curtos": (
                    df["text"].str.len() < df["text"].str.len().quantile(0.05)
                ).sum(),
            },
        }

        return quality_metrics

    def _print_quality_report(self, report: dict):
        """Imprime relatório de qualidade de forma organizada."""
        print("\n" + "=" * 80)
        print("📊 RELATÓRIO DE QUALIDADE DOS DADOS")
        print("=" * 80)

        # Resumo geral
        resumo = report["resumo_geral"]
        print(f"Total de tickets processados:        {resumo['total_tickets']:,}")
        print(f"Tickets válidos:                     {resumo['tickets_validos']:,}")
        print(f"Taxa de sucesso:                     {resumo['taxa_sucesso']:.1f}%")

        # Estatísticas de texto
        print("\n📝 ESTATÍSTICAS DE TEXTO:")
        texto = report["estatisticas_texto"]
        chars = texto["comprimento_caracteres"]
        words = texto["contagem_palavras"]

        print(
            f"Caracteres - Média: {chars['media']:.0f}, Mediana: {chars['mediana']:.0f}, Max: {chars['maximo']:,}"
        )
        print(
            f"Palavras   - Média: {words['media']:.1f}, Mediana: {words['mediana']:.1f}, Max: {words['maximo']:,}"
        )

        # Distribuição de mensagens
        if "mensagens_por_ticket" in report["distribuicao_mensagens"]:
            print("\n💬 DISTRIBUIÇÃO DE MENSAGENS:")
            msgs = report["distribuicao_mensagens"]["mensagens_por_ticket"]
            print(
                f"Mensagens por ticket - Média: {msgs['media']:.1f}, Mediana: {msgs['mediana']:.1f}, Max: {msgs['maximo']}"
            )

        # Qualidade dos dados
        print("\n✅ QUALIDADE DOS DADOS:")
        quality = report["qualidade_dados"]
        print(
            f"Completude texto:                    {quality['completude']['tickets_com_texto']:.1f}%"
        )
        print(
            f"Consistência IDs:                    {'✓' if quality['consistencia']['ids_unicos'] else '✗'}"
        )
        print(
            f"Textos adequados (>10 chars):        {quality['consistencia']['textos_nao_vazios']:.1f}%"
        )

        # Período temporal
        if report["periodo_temporal"]:
            periodo = report["periodo_temporal"]
            print("\n📅 PERÍODO TEMPORAL:")
            print(
                f"Período analisado:                   {periodo.get('periodo_dias', 0)} dias"
            )
            if "data_mais_antiga" in periodo:
                print(
                    f"Data mais antiga:                    {periodo['data_mais_antiga']}"
                )
                print(
                    f"Data mais recente:                   {periodo['data_mais_recente']}"
                )

        print("=" * 80)

    def process_batches_parallel(
        self, chain, tickets: list, token_limit: int = 50000
    ) -> Tuple[list, int, int, int]:
        """Processa os tickets em batches de forma paralela."""
        # Divide os tickets em batches
        batches = []
        batch_tickets = []
        batch_tokens = 0
        header = "Tickets:\n"
        header_tokens = self.estimate_tokens(header)
        batch_tokens += header_tokens

        # Estima total de batches
        estimated_total_batches = (
            sum(len(t["text"]) // 4 for t in tickets) // token_limit
        ) + 1
        print(f"\nTotal estimado de batches: {estimated_total_batches}")

        # Prepara os batches
        for ticket in tickets:
            ticket_tokens = self.estimate_tokens(ticket["text"])

            # Se adicionar este ticket exceder o limite, finalize o batch atual
            if (
                batch_tokens + ticket_tokens >= token_limit
                or len(batch_tickets) >= self.max_tickets_per_batch
            ):
                if batch_tickets:  # Só adiciona se o batch não estiver vazio
                    batches.append(batch_tickets.copy())
                batch_tickets = []
                batch_tokens = header_tokens

            batch_tickets.append(ticket)
            batch_tokens += ticket_tokens

        # Adiciona o último batch se não estiver vazio
        if batch_tickets:
            batches.append(batch_tickets)

        print(f"Dividido em {len(batches)} batches para processamento paralelo")

        # Função para processar um único batch
        def process_single_batch(batch_index, batch):
            try:
                # Constrói o texto para este batch
                batch_text = "\n\n".join(
                    f"Ticket {ticket['ticket_id']}:\n{ticket['text']}"
                    for ticket in batch
                )

                input_tokens = self.estimate_tokens(batch_text)

                # Processa o batch
                response = chain.invoke({"tickets_text": batch_text})
                output_tokens = self.estimate_tokens(response)

                # Extrai e valida o JSON
                json_str = self.extract_json(response)
                batch_result = json.loads(json_str)

                # Valida a estrutura do resultado
                valid_results = []
                for item in batch_result:
                    if not isinstance(item, dict):
                        continue
                    if "ticket_id" not in item:
                        continue
                    if not item.get("categorias") or not item.get("resumo"):
                        continue
                    valid_results.append(item)

                return {
                    "results": valid_results,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "success": True,
                }
            except Exception as e:
                # Log do erro
                log_file = (
                    self.database_dir
                    / f"error_log_batch_{batch_index}_{int(time.time())}.log"
                )
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"Erro no batch {batch_index}: {str(e)}\n")
                    f.write(f"Tipo do erro: {type(e)}\n")
                    if "response" in locals():
                        f.write(f"Resposta: {response}\n")
                    if "json_str" in locals():
                        f.write(f"JSON extraído: {json_str}\n")

                return {
                    "results": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "success": False,
                    "error": str(e),
                }

        # Processa os batches em paralelo
        total_input_tokens = 0
        total_output_tokens = 0
        batch_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submete todos os batches para processamento
            future_to_batch = {
                executor.submit(process_single_batch, i, batch): (i, batch)
                for i, batch in enumerate(batches)
            }

            # Coleta os resultados à medida que são concluídos
            for future in tqdm(
                concurrent.futures.as_completed(future_to_batch),
                total=len(batches),
                desc="Processando batches",
            ):
                batch_index, _ = future_to_batch[future]
                try:
                    result = future.result()
                    if result["success"]:
                        batch_results.extend(result["results"])
                        total_input_tokens += result["input_tokens"]
                        total_output_tokens += result["output_tokens"]
                        print(
                            f"Batch {batch_index} processado com sucesso: {len(result['results'])} resultados"
                        )
                    else:
                        print(
                            f"Falha no batch {batch_index}: {result.get('error', 'Erro desconhecido')}"
                        )
                except Exception as e:
                    print(f"Exceção ao processar batch {batch_index}: {str(e)}")

        return batch_results, len(batches), total_input_tokens, total_output_tokens

    def process_chunks_with_cache(
        self, chunks: List[Document], map_chain, processor_name: str
    ) -> Tuple[List[str], int, int]:
        """Processa chunks com suporte a cache."""
        partial_results = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Função para processar um chunk
        def process_chunk(doc_index, doc):
            # Gera chave de cache para este chunk
            chunk_text = doc.page_content
            cache_key = self._generate_cache_key(
                {
                    "chunk": chunk_text[
                        :1000
                    ],  # Usa os primeiros 1000 caracteres para o hash
                    "length": len(chunk_text),
                    "method": f"{processor_name}_chunk",
                }
            )

            # Tenta recuperar do cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                print(f"Usando resultado em cache para chunk {doc_index}")
                return cached_result

            try:
                # Processa o chunk
                input_tokens = self.estimate_tokens(chunk_text)

                result = map_chain.invoke({"text": chunk_text})

                output_tokens = self.estimate_tokens(result)

                # Prepara o resultado
                processed_result = {
                    "result": result,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "success": True,
                }

                # Salva no cache
                self._save_to_cache(cache_key, processed_result)

                return processed_result
            except Exception as e:
                print(f"Erro ao processar chunk {doc_index}: {str(e)}")
                return {
                    "result": None,
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
                for i, doc in enumerate(chunks, 1)
            }

            # Coleta os resultados à medida que são concluídos
            for future in tqdm(
                concurrent.futures.as_completed(future_to_chunk),
                total=len(chunks),
                desc="Processando chunks",
            ):
                chunk_index, _ = future_to_chunk[future]
                try:
                    result = future.result()
                    if result["success"]:
                        partial_results.append(result["result"])
                        total_input_tokens += result["input_tokens"]
                        total_output_tokens += result["output_tokens"]
                        print(
                            f"Chunk {chunk_index}/{len(chunks)} processado - Input: {result['input_tokens']:,}, Output: {result['output_tokens']:,}"
                        )
                    else:
                        print(
                            f"Falha no chunk {chunk_index}: {result.get('error', 'Erro desconhecido')}"
                        )
                except Exception as e:
                    print(f"Exceção ao processar chunk {chunk_index}: {str(e)}")

        return partial_results, total_input_tokens, total_output_tokens

    def process_batches_with_cache(
        self, batches: List[List[dict]], process_func, processor_name: str
    ) -> Tuple[List[dict], int, int]:
        """Processa batches com suporte a cache."""
        all_results = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Processa os batches em paralelo
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submete todos os batches para processamento
            future_to_batch = {
                executor.submit(
                    self._process_batch_with_cache,
                    i,
                    batch,
                    process_func,
                    processor_name,
                ): (i, batch)
                for i, batch in enumerate(batches, 1)
            }

            # Coleta os resultados à medida que são concluídos
            for future in tqdm(
                concurrent.futures.as_completed(future_to_batch),
                total=len(batches),
                desc="Processando batches",
            ):
                batch_index, _ = future_to_batch[future]
                try:
                    result = future.result()
                    if result["success"]:
                        all_results.extend(result["results"])
                        total_input_tokens += result["input_tokens"]
                        total_output_tokens += result["output_tokens"]
                        print(
                            f"Batch {batch_index}/{len(batches)} processado: {len(result['results'])} itens"
                        )
                    else:
                        print(
                            f"Falha no batch {batch_index}: {result.get('error', 'Erro desconhecido')}"
                        )
                except Exception as e:
                    print(f"Exceção ao processar batch {batch_index}: {str(e)}")

        return all_results, total_input_tokens, total_output_tokens

    def _process_batch_with_cache(
        self, batch_index: int, batch: List[dict], process_func, processor_name: str
    ) -> Dict[str, Any]:
        """Processa um único batch com suporte a cache."""
        # Gera chave de cache para este batch
        # Usamos os IDs dos tickets como identificador do batch
        ticket_ids = [ticket["ticket_id"] for ticket in batch]
        cache_key = self._generate_cache_key(
            {"ticket_ids": ticket_ids, "method": f"{processor_name}_batch"}
        )

        # Tenta recuperar do cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            print(f"Usando resultado em cache para batch {batch_index}")
            return cached_result

        try:
            # Processa o batch usando a função fornecida
            result = process_func(batch_index, batch)

            # Salva no cache
            self._save_to_cache(cache_key, result)

            return result
        except Exception as e:
            print(f"Erro ao processar batch {batch_index}: {str(e)}")
            return {
                "results": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False,
                "error": str(e),
            }
