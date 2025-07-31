"""
Sistema de rastreamento de custos e tokens em tempo real.

Implementa tracking abrangente de uso de tokens e cálculos de custo para
operações de processamento de linguagem usando modelos Google Gemini.
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import csv


@dataclass
class TokenUsage:
    """Estrutura para rastreamento de uso de tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = 0.0
    operation_type: str = ""
    phase: str = ""

    @property
    def total_tokens(self) -> int:
        """Total de tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Calcula custo em USD baseado no pricing do Gemini 2.5 Flash."""
        # Pricing Gemini 2.5 Flash: $0.125/1K input, $0.375/1K output
        input_cost = (self.input_tokens / 1000) * 0.125
        output_cost = (self.output_tokens / 1000) * 0.375
        return input_cost + output_cost


@dataclass
class CostSession:
    """Representa uma sessão de processamento com tracking de custos."""

    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    operations_count: int = 0
    dataset_size: int = 0
    processing_mode: str = ""

    @property
    def duration_seconds(self) -> float:
        """Duração da sessão em segundos."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def cost_per_item(self) -> float:
        """Custo por item processado."""
        if self.dataset_size > 0:
            return self.total_cost_usd / self.dataset_size
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Taxa de processamento em tokens por segundo."""
        duration = self.duration_seconds
        if duration > 0:
            return (self.total_input_tokens + self.total_output_tokens) / duration
        return 0.0


class CostTracker:
    """
    Sistema principal de rastreamento de custos e tokens em tempo real.

    Fornece funcionalidades para:
    - Tracking de tokens em tempo real
    - Cálculo de custos por operação
    - Monitoramento de budget
    - Relatórios e analytics
    """

    # Pricing Gemini 2.5 Flash (USD per 1K tokens)
    INPUT_TOKEN_PRICE = 0.125 / 1000  # $0.000125 per token
    OUTPUT_TOKEN_PRICE = 0.375 / 1000  # $0.000375 per token

    def __init__(self, storage_dir: Path = None, budget_limit_usd: float = None):
        """
        Inicializa o CostTracker.

        Args:
            storage_dir: Diretório para armazenar dados de tracking
            budget_limit_usd: Limite de budget em USD (opcional)
        """
        self.storage_dir = storage_dir or Path("database/cost_tracking")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.budget_limit_usd = budget_limit_usd
        self._current_session: Optional[CostSession] = None
        self._usage_history: List[TokenUsage] = []
        self._sessions_history: List[CostSession] = []
        self._operation_stats = defaultdict(lambda: defaultdict(int))

        # Thread safety
        self._lock = threading.RLock()

        # Budget monitoring
        self._budget_alerts_enabled = True
        self._alert_thresholds = [0.5, 0.75, 0.9, 0.95]  # 50%, 75%, 90%, 95%
        self._triggered_alerts = set()

        # Load historical data
        self._load_historical_data()

    def start_session(
        self, session_id: str, dataset_size: int = 0, processing_mode: str = "unknown"
    ) -> str:
        """
        Inicia uma nova sessão de tracking.

        Args:
            session_id: Identificador único da sessão
            dataset_size: Tamanho do dataset sendo processado
            processing_mode: Modo de processamento (categorize, summarize, etc.)

        Returns:
            ID da sessão iniciada
        """
        with self._lock:
            if self._current_session and not self._current_session.end_time:
                # Finaliza sessão anterior se ainda ativa
                self.end_session()

            self._current_session = CostSession(
                session_id=session_id,
                start_time=time.time(),
                dataset_size=dataset_size,
                processing_mode=processing_mode,
            )

            print(f"💰 Sessão de custo iniciada: {session_id}")
            print(f"   Dataset size: {dataset_size:,} items")
            print(f"   Processing mode: {processing_mode}")
            if self.budget_limit_usd:
                print(f"   Budget limit: ${self.budget_limit_usd:.2f}")

            return session_id

    def end_session(self) -> Optional[CostSession]:
        """
        Finaliza a sessão atual e retorna estatísticas.

        Returns:
            Dados da sessão finalizada ou None se não havia sessão ativa
        """
        with self._lock:
            if not self._current_session:
                return None

            # Finaliza sessão
            self._current_session.end_time = time.time()

            # Salva na história
            self._sessions_history.append(self._current_session)

            # Salva dados
            self._save_session_data(self._current_session)

            # Relatório final
            session = self._current_session
            print(f"\n💰 RELATÓRIO FINAL DA SESSÃO: {session.session_id}")
            print(f"   Duração: {session.duration_seconds:.1f}s")
            print(
                f"   Total tokens: {session.total_input_tokens + session.total_output_tokens:,}"
            )
            print(f"   Input tokens: {session.total_input_tokens:,}")
            print(f"   Output tokens: {session.total_output_tokens:,}")
            print(f"   Custo total: ${session.total_cost_usd:.4f}")
            print(f"   Custo por item: ${session.cost_per_item:.6f}")
            print(f"   Throughput: {session.tokens_per_second:.1f} tokens/s")

            completed_session = self._current_session
            self._current_session = None

            return completed_session

    def track_operation(
        self,
        input_tokens: int,
        output_tokens: int,
        operation_type: str = "unknown",
        phase: str = "unknown",
    ) -> TokenUsage:
        """
        Registra uma operação com uso de tokens.

        Args:
            input_tokens: Número de tokens de input
            output_tokens: Número de tokens de output
            operation_type: Tipo da operação (map, reduce, categorize, etc.)
            phase: Fase do processamento

        Returns:
            Objeto TokenUsage com detalhes da operação
        """
        with self._lock:
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=time.time(),
                operation_type=operation_type,
                phase=phase,
            )

            # Adiciona ao histórico
            self._usage_history.append(usage)

            # Atualiza sessão atual
            if self._current_session:
                self._current_session.total_input_tokens += input_tokens
                self._current_session.total_output_tokens += output_tokens
                self._current_session.total_cost_usd += usage.cost_usd
                self._current_session.operations_count += 1

            # Atualiza estatísticas por operação
            self._operation_stats[operation_type]["operations"] += 1
            self._operation_stats[operation_type]["input_tokens"] += input_tokens
            self._operation_stats[operation_type]["output_tokens"] += output_tokens
            self._operation_stats[operation_type]["cost_usd"] += usage.cost_usd

            # Verifica alertas de budget
            if self.budget_limit_usd:
                self._check_budget_alerts()

            # Log da operação
            print(
                f"🔢 Token usage - {operation_type}: "
                f"Input: {input_tokens:,}, Output: {output_tokens:,}, "
                f"Cost: ${usage.cost_usd:.6f}"
            )

            return usage

    def estimate_tokens_from_text(self, text: str) -> int:
        """
        Estima número de tokens baseado no texto.
        Usa aproximação de 1 token ≈ 4 caracteres.

        Args:
            text: Texto para estimar tokens

        Returns:
            Número estimado de tokens
        """
        return len(text) // 4

    def track_text_operation(
        self,
        input_text: str,
        output_text: str,
        operation_type: str = "unknown",
        phase: str = "unknown",
    ) -> TokenUsage:
        """
        Registra operação baseada em textos (estima tokens automaticamente).

        Args:
            input_text: Texto de input
            output_text: Texto de output
            operation_type: Tipo da operação
            phase: Fase do processamento

        Returns:
            Objeto TokenUsage com detalhes da operação
        """
        input_tokens = self.estimate_tokens_from_text(input_text)
        output_tokens = self.estimate_tokens_from_text(output_text)

        return self.track_operation(input_tokens, output_tokens, operation_type, phase)

    def get_current_session_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da sessão atual.

        Returns:
            Dicionário com estatísticas da sessão ou dados vazios se sem sessão
        """
        with self._lock:
            if not self._current_session:
                return {"error": "No active session"}

            session = self._current_session
            current_cost = session.total_cost_usd

            stats = {
                "session_id": session.session_id,
                "duration_seconds": session.duration_seconds,
                "total_tokens": session.total_input_tokens
                + session.total_output_tokens,
                "input_tokens": session.total_input_tokens,
                "output_tokens": session.total_output_tokens,
                "total_cost_usd": current_cost,
                "operations_count": session.operations_count,
                "cost_per_item": session.cost_per_item,
                "tokens_per_second": session.tokens_per_second,
                "dataset_size": session.dataset_size,
                "processing_mode": session.processing_mode,
            }

            # Adiciona informações de budget se configurado
            if self.budget_limit_usd:
                budget_used_percent = (current_cost / self.budget_limit_usd) * 100
                stats.update(
                    {
                        "budget_limit_usd": self.budget_limit_usd,
                        "budget_remaining_usd": self.budget_limit_usd - current_cost,
                        "budget_used_percent": budget_used_percent,
                        "budget_status": self._get_budget_status(budget_used_percent),
                    }
                )

            return stats

    def get_operation_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna breakdown de custos por tipo de operação.

        Returns:
            Dicionário com estatísticas por tipo de operação
        """
        with self._lock:
            breakdown = {}

            for op_type, stats in self._operation_stats.items():
                total_tokens = stats["input_tokens"] + stats["output_tokens"]
                breakdown[op_type] = {
                    "operations": stats["operations"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": total_tokens,
                    "cost_usd": stats["cost_usd"],
                    "avg_tokens_per_operation": total_tokens
                    / max(stats["operations"], 1),
                    "avg_cost_per_operation": stats["cost_usd"]
                    / max(stats["operations"], 1),
                }

            return breakdown

    def get_cost_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Analisa tendências de custo nas últimas horas.

        Args:
            hours_back: Número de horas para análise

        Returns:
            Dicionário com análise de tendências
        """
        cutoff_time = time.time() - (hours_back * 3600)

        recent_usage = [u for u in self._usage_history if u.timestamp >= cutoff_time]
        recent_sessions = [
            s for s in self._sessions_history if s.start_time >= cutoff_time
        ]

        if not recent_usage:
            return {"error": "No data available for the specified period"}

        # Agrupa por hora
        hourly_costs = defaultdict(float)
        hourly_tokens = defaultdict(int)

        for usage in recent_usage:
            hour_key = int(usage.timestamp // 3600)
            hourly_costs[hour_key] += usage.cost_usd
            hourly_tokens[hour_key] += usage.total_tokens

        # Calcula métricas
        total_cost = sum(u.cost_usd for u in recent_usage)
        total_tokens = sum(u.total_tokens for u in recent_usage)

        trends = {
            "period_hours": hours_back,
            "total_operations": len(recent_usage),
            "total_sessions": len(recent_sessions),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_hour": total_cost / hours_back,
            "avg_tokens_per_hour": total_tokens / hours_back,
            "hourly_breakdown": {
                str(hour): {"cost": cost, "tokens": hourly_tokens[hour]}
                for hour, cost in hourly_costs.items()
            },
        }

        # Adiciona projeções se há dados suficientes
        if len(recent_sessions) > 0:
            avg_session_cost = sum(s.total_cost_usd for s in recent_sessions) / len(
                recent_sessions
            )
            avg_cost_per_item = sum(
                s.cost_per_item for s in recent_sessions if s.dataset_size > 0
            ) / max(1, len([s for s in recent_sessions if s.dataset_size > 0]))

            trends.update(
                {
                    "avg_session_cost": avg_session_cost,
                    "avg_cost_per_item": avg_cost_per_item,
                    "projected_daily_cost": total_cost * (24 / hours_back),
                    "projected_weekly_cost": total_cost * (168 / hours_back),
                }
            )

        return trends

    def _check_budget_alerts(self):
        """Verifica e dispara alertas de budget se necessário."""
        if not self.budget_limit_usd or not self._current_session:
            return

        current_cost = self._current_session.total_cost_usd
        budget_used_percent = current_cost / self.budget_limit_usd

        for threshold in self._alert_thresholds:
            if (
                budget_used_percent >= threshold
                and threshold not in self._triggered_alerts
            ):
                self._triggered_alerts.add(threshold)
                self._trigger_budget_alert(threshold, current_cost, budget_used_percent)

    def _trigger_budget_alert(
        self, threshold: float, current_cost: float, budget_used_percent: float
    ):
        """Dispara alerta de budget."""
        print(f"\n🚨 BUDGET ALERT: {threshold*100:.0f}% threshold reached!")
        print(f"   Current cost: ${current_cost:.4f}")
        print(f"   Budget limit: ${self.budget_limit_usd:.2f}")
        print(f"   Usage: {budget_used_percent*100:.1f}%")
        print(f"   Remaining: ${self.budget_limit_usd - current_cost:.4f}")

        # Salva alerta
        alert_data = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "threshold_percent": threshold * 100,
            "current_cost_usd": current_cost,
            "budget_limit_usd": self.budget_limit_usd,
            "budget_used_percent": budget_used_percent * 100,
            "session_id": (
                self._current_session.session_id if self._current_session else None
            ),
        }

        alerts_file = self.storage_dir / "budget_alerts.json"
        try:
            if alerts_file.exists():
                with open(alerts_file, "r", encoding="utf-8") as f:
                    alerts = json.load(f)
            else:
                alerts = []

            alerts.append(alert_data)

            with open(alerts_file, "w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erro ao salvar alerta: {e}")

    def _get_budget_status(self, budget_used_percent: float) -> str:
        """Retorna status do budget baseado na porcentagem usada."""
        if budget_used_percent >= 95:
            return "critical"
        elif budget_used_percent >= 75:
            return "warning"
        elif budget_used_percent >= 50:
            return "caution"
        else:
            return "healthy"

    def _save_session_data(self, session: CostSession):
        """Salva dados da sessão em arquivo."""
        try:
            sessions_file = self.storage_dir / "sessions.json"

            if sessions_file.exists():
                with open(sessions_file, "r", encoding="utf-8") as f:
                    sessions = json.load(f)
            else:
                sessions = []

            sessions.append(asdict(session))

            with open(sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erro ao salvar dados da sessão: {e}")

    def _load_historical_data(self):
        """Carrega dados históricos de sessões."""
        try:
            sessions_file = self.storage_dir / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file, "r", encoding="utf-8") as f:
                    sessions_data = json.load(f)

                for session_dict in sessions_data:
                    session = CostSession(**session_dict)
                    self._sessions_history.append(session)

        except Exception as e:
            print(f"Aviso: Erro ao carregar dados históricos: {e}")

    def export_cost_report_csv(
        self,
        filename: str = None,
        include_sessions: bool = True,
        include_operations: bool = True,
    ) -> Path:
        """
        Exporta relatório de custos em formato CSV.

        Args:
            filename: Nome do arquivo (opcional)
            include_sessions: Incluir dados de sessões
            include_operations: Incluir breakdown por operações

        Returns:
            Path do arquivo gerado
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_report_{timestamp}.csv"

        report_file = self.storage_dir / filename

        with open(report_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header principal
            writer.writerow(["COST TRACKING REPORT"])
            writer.writerow(["Generated:", datetime.now().isoformat()])
            writer.writerow([])

            # Resumo geral
            if self._sessions_history:
                total_cost = sum(s.total_cost_usd for s in self._sessions_history)
                total_tokens = sum(
                    s.total_input_tokens + s.total_output_tokens
                    for s in self._sessions_history
                )
                writer.writerow(["SUMMARY"])
                writer.writerow(["Total Sessions:", len(self._sessions_history)])
                writer.writerow(["Total Cost (USD):", f"{total_cost:.6f}"])
                writer.writerow(["Total Tokens:", f"{total_tokens:,}"])
                writer.writerow(
                    [
                        "Average Cost per Session:",
                        f"{total_cost/len(self._sessions_history):.6f}",
                    ]
                )
                writer.writerow([])

            # Dados de sessões
            if include_sessions and self._sessions_history:
                writer.writerow(["SESSIONS"])
                writer.writerow(
                    [
                        "Session ID",
                        "Start Time",
                        "Duration (s)",
                        "Dataset Size",
                        "Processing Mode",
                        "Input Tokens",
                        "Output Tokens",
                        "Total Tokens",
                        "Total Cost (USD)",
                        "Cost per Item (USD)",
                        "Tokens/Second",
                    ]
                )

                for session in self._sessions_history:
                    writer.writerow(
                        [
                            session.session_id,
                            datetime.fromtimestamp(session.start_time).isoformat(),
                            f"{session.duration_seconds:.1f}",
                            session.dataset_size,
                            session.processing_mode,
                            session.total_input_tokens,
                            session.total_output_tokens,
                            session.total_input_tokens + session.total_output_tokens,
                            f"{session.total_cost_usd:.6f}",
                            f"{session.cost_per_item:.8f}",
                            f"{session.tokens_per_second:.1f}",
                        ]
                    )
                writer.writerow([])

            # Breakdown por operações
            if include_operations and self._operation_stats:
                writer.writerow(["OPERATIONS BREAKDOWN"])
                writer.writerow(
                    [
                        "Operation Type",
                        "Count",
                        "Input Tokens",
                        "Output Tokens",
                        "Total Tokens",
                        "Total Cost (USD)",
                        "Avg Tokens/Op",
                        "Avg Cost/Op (USD)",
                    ]
                )

                breakdown = self.get_operation_breakdown()
                for op_type, stats in breakdown.items():
                    writer.writerow(
                        [
                            op_type,
                            stats["operations"],
                            stats["input_tokens"],
                            stats["output_tokens"],
                            stats["total_tokens"],
                            f"{stats['cost_usd']:.6f}",
                            f"{stats['avg_tokens_per_operation']:.1f}",
                            f"{stats['avg_cost_per_operation']:.6f}",
                        ]
                    )

        print(f"📊 Relatório de custos exportado: {report_file}")
        return report_file

    def reset_budget_alerts(self):
        """Reseta alertas de budget disparados."""
        with self._lock:
            self._triggered_alerts.clear()
            print("🔄 Alertas de budget resetados")

    def get_total_spent(self) -> float:
        """Retorna total gasto histórico em USD."""
        return sum(s.total_cost_usd for s in self._sessions_history)

    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Calcula métricas de eficiência de processamento."""
        if not self._sessions_history:
            return {}

        valid_sessions = [s for s in self._sessions_history if s.dataset_size > 0]
        if not valid_sessions:
            return {}

        costs_per_item = [s.cost_per_item for s in valid_sessions]
        tokens_per_second = [s.tokens_per_second for s in valid_sessions]

        return {
            "avg_cost_per_item": sum(costs_per_item) / len(costs_per_item),
            "min_cost_per_item": min(costs_per_item),
            "max_cost_per_item": max(costs_per_item),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second),
            "efficiency_score": (1 / (sum(costs_per_item) / len(costs_per_item)))
            * 1000,  # Higher is better
        }
