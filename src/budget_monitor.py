"""
Sistema de monitoramento de budget com alertas inteligentes.

Implementa monitoramento contínuo de custos com alertas configuráveis,
suspensão automática e análise preditiva de budget.
"""

import json
import time
import threading
import smtplib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback para versões mais antigas do Python
    MimeText = None
    MimeMultipart = None
from dataclasses import dataclass, asdict
from enum import Enum


class AlertLevel(Enum):
    """Níveis de alerta para diferentes situações de budget."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Tipos de alerta disponíveis."""

    BUDGET_THRESHOLD = "budget_threshold"
    DAILY_LIMIT = "daily_limit"
    HOURLY_SPIKE = "hourly_spike"
    PREDICTIVE_OVERAGE = "predictive_overage"
    EFFICIENCY_DROP = "efficiency_drop"
    COST_ANOMALY = "cost_anomaly"


@dataclass
class AlertConfig:
    """Configuração de alerta."""

    alert_type: AlertType
    alert_level: AlertLevel
    threshold_value: float
    enabled: bool = True
    cooldown_minutes: int = 60  # Tempo mínimo entre alertas do mesmo tipo
    notification_methods: List[str] = None  # email, console, webhook

    def __post_init__(self):
        if self.notification_methods is None:
            self.notification_methods = ["console"]


@dataclass
class BudgetAlert:
    """Representa um alerta de budget."""

    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    timestamp: float
    message: str
    current_value: float
    threshold_value: float
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = None
    acknowledged: bool = False

    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


class BudgetMonitor:
    """
    Sistema principal de monitoramento de budget com alertas inteligentes.

    Funcionalidades:
    - Monitoramento em tempo real de custos
    - Alertas configuráveis por threshold
    - Análise preditiva de overage
    - Suspensão automática em situações críticas
    - Notificações por múltiplos canais
    """

    def __init__(self, storage_dir: Path = None, email_config: Dict = None):
        """
        Inicializa o monitor de budget.

        Args:
            storage_dir: Diretório para armazenar dados de alertas
            email_config: Configuração para notificações por email
        """
        self.storage_dir = storage_dir or Path("database/budget_monitoring")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Configurações
        self.email_config = email_config
        self._monitoring_active = False
        self._monitoring_thread = None
        self._monitoring_interval = 30  # 30 segundos

        # Estado do monitor
        self._alert_configs: Dict[str, AlertConfig] = {}
        self._active_alerts: List[BudgetAlert] = []
        self._alert_history: List[BudgetAlert] = []
        self._last_alert_times: Dict[AlertType, float] = {}

        # Callbacks customizados
        self._custom_alert_handlers: Dict[AlertType, List[Callable]] = {}
        self._suspension_handler: Optional[Callable] = None

        # Dados para análise preditiva
        self._cost_history: List[Dict[str, float]] = []
        self._cost_tracking_window = 3600  # 1 hora

        # Thread safety
        self._lock = threading.RLock()

        # Carrega configurações salvas
        self._load_configurations()
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Configura alertas padrão."""
        default_alerts = [
            AlertConfig(
                alert_type=AlertType.BUDGET_THRESHOLD,
                alert_level=AlertLevel.WARNING,
                threshold_value=0.5,  # 50%
                notification_methods=["console"],
            ),
            AlertConfig(
                alert_type=AlertType.BUDGET_THRESHOLD,
                alert_level=AlertLevel.CRITICAL,
                threshold_value=0.75,  # 75%
                notification_methods=["console", "email"],
            ),
            AlertConfig(
                alert_type=AlertType.BUDGET_THRESHOLD,
                alert_level=AlertLevel.EMERGENCY,
                threshold_value=0.95,  # 95%
                notification_methods=["console", "email"],
            ),
            AlertConfig(
                alert_type=AlertType.DAILY_LIMIT,
                alert_level=AlertLevel.CRITICAL,
                threshold_value=10.0,  # $10/dia
                notification_methods=["console"],
            ),
            AlertConfig(
                alert_type=AlertType.HOURLY_SPIKE,
                alert_level=AlertLevel.WARNING,
                threshold_value=2.0,  # $2/hora
                notification_methods=["console"],
            ),
        ]

        for alert_config in default_alerts:
            self.add_alert_config(alert_config)

    def add_alert_config(self, config: AlertConfig):
        """Adiciona ou atualiza configuração de alerta."""
        with self._lock:
            config_key = f"{config.alert_type.value}_{config.threshold_value}"
            self._alert_configs[config_key] = config
            print(
                f"🔔 Alerta configurado: {config.alert_type.value} @ {config.threshold_value} ({config.alert_level.value})"
            )

    def remove_alert_config(self, alert_type: AlertType, threshold_value: float):
        """Remove configuração de alerta."""
        with self._lock:
            config_key = f"{alert_type.value}_{threshold_value}"
            if config_key in self._alert_configs:
                del self._alert_configs[config_key]
                print(f"🔕 Alerta removido: {alert_type.value} @ {threshold_value}")

    def start_monitoring(self, cost_tracker, budget_limit: float):
        """
        Inicia monitoramento contínuo.

        Args:
            cost_tracker: Instância do CostTracker para monitorar
            budget_limit: Limite de budget em USD
        """
        if self._monitoring_active:
            print("⚠️ Monitoramento já está ativo")
            return

        self.cost_tracker = cost_tracker
        self.budget_limit = budget_limit
        self._monitoring_active = True

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

        print(f"👁️ Monitoramento de budget iniciado")
        print(f"   Budget limit: ${budget_limit:.2f}")
        print(f"   Monitoring interval: {self._monitoring_interval}s")
        print(f"   Alert configs: {len(self._alert_configs)}")

    def stop_monitoring(self):
        """Para o monitoramento contínuo."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        print("🛑 Monitoramento de budget parado")

    def _monitoring_loop(self):
        """Loop principal de monitoramento."""
        while self._monitoring_active:
            try:
                self._check_all_alerts()
                time.sleep(self._monitoring_interval)
            except Exception as e:
                print(f"❌ Erro no monitoramento: {e}")
                time.sleep(self._monitoring_interval * 2)  # Espera mais em caso de erro

    def _check_all_alerts(self):
        """Verifica todos os tipos de alerta configurados."""
        current_stats = self.cost_tracker.get_current_session_stats()

        if "error" in current_stats:
            return  # Sem sessão ativa para monitorar

        # Atualiza histórico de custos
        self._update_cost_history(current_stats)

        # Verifica alertas de threshold de budget
        self._check_budget_threshold_alerts(current_stats)

        # Verifica alertas de limite diário
        self._check_daily_limit_alerts()

        # Verifica spikes horárias
        self._check_hourly_spike_alerts()

        # Verifica alertas preditivos
        self._check_predictive_alerts(current_stats)

        # Verifica drops de eficiência
        self._check_efficiency_alerts(current_stats)

        # Verifica anomalias de custo
        self._check_cost_anomaly_alerts()

    def _update_cost_history(self, current_stats: Dict):
        """Atualiza histórico de custos para análise."""
        current_time = time.time()
        cost_point = {
            "timestamp": current_time,
            "total_cost": current_stats.get("total_cost_usd", 0),
            "cost_per_item": current_stats.get("cost_per_item", 0),
            "tokens_per_second": current_stats.get("tokens_per_second", 0),
        }

        self._cost_history.append(cost_point)

        # Mantém apenas dados da janela de tracking
        cutoff_time = current_time - self._cost_tracking_window
        self._cost_history = [
            point for point in self._cost_history if point["timestamp"] >= cutoff_time
        ]

    def _check_budget_threshold_alerts(self, current_stats: Dict):
        """Verifica alertas de threshold de budget."""
        if not hasattr(self, "budget_limit") or self.budget_limit <= 0:
            return

        current_cost = current_stats.get("total_cost_usd", 0)
        budget_used_percent = current_cost / self.budget_limit

        for config_key, config in self._alert_configs.items():
            if config.alert_type != AlertType.BUDGET_THRESHOLD or not config.enabled:
                continue

            if budget_used_percent >= config.threshold_value:
                if self._should_trigger_alert(
                    config.alert_type, config.threshold_value
                ):
                    self._trigger_alert(
                        alert_type=config.alert_type,
                        alert_level=config.alert_level,
                        message=f"Budget threshold {config.threshold_value*100:.0f}% atingido",
                        current_value=budget_used_percent,
                        threshold_value=config.threshold_value,
                        session_id=current_stats.get("session_id"),
                        config=config,
                    )

                    # Suspensão automática em emergência
                    if config.alert_level == AlertLevel.EMERGENCY:
                        self._trigger_emergency_suspension(current_stats)

    def _check_daily_limit_alerts(self):
        """Verifica alertas de limite diário."""
        # Calcula custo das últimas 24 horas
        current_time = time.time()
        day_ago = current_time - 86400  # 24 horas

        daily_cost = 0
        for point in self._cost_history:
            if point["timestamp"] >= day_ago:
                daily_cost += point.get("total_cost", 0)

        # Se não há dados suficientes, usa projeção baseada na sessão atual
        if not self._cost_history and hasattr(self, "cost_tracker"):
            current_stats = self.cost_tracker.get_current_session_stats()
            if "error" not in current_stats:
                session_duration_hours = current_stats.get("duration_seconds", 0) / 3600
                if session_duration_hours > 0:
                    hourly_rate = (
                        current_stats.get("total_cost_usd", 0) / session_duration_hours
                    )
                    daily_cost = hourly_rate * 24  # Projeção de 24h

        for config_key, config in self._alert_configs.items():
            if config.alert_type != AlertType.DAILY_LIMIT or not config.enabled:
                continue

            if daily_cost >= config.threshold_value:
                if self._should_trigger_alert(
                    config.alert_type, config.threshold_value
                ):
                    self._trigger_alert(
                        alert_type=config.alert_type,
                        alert_level=config.alert_level,
                        message=f"Limite diário de ${config.threshold_value:.2f} atingido",
                        current_value=daily_cost,
                        threshold_value=config.threshold_value,
                        config=config,
                    )

    def _check_hourly_spike_alerts(self):
        """Verifica alertas de spikes de custo horárias."""
        if len(self._cost_history) < 2:
            return

        # Calcula custo da última hora
        current_time = time.time()
        hour_ago = current_time - 3600

        recent_costs = [
            point["total_cost"]
            for point in self._cost_history
            if point["timestamp"] >= hour_ago
        ]

        if len(recent_costs) >= 2:
            hourly_cost = max(recent_costs) - min(recent_costs)

            for config_key, config in self._alert_configs.items():
                if config.alert_type != AlertType.HOURLY_SPIKE or not config.enabled:
                    continue

                if hourly_cost >= config.threshold_value:
                    if self._should_trigger_alert(
                        config.alert_type, config.threshold_value
                    ):
                        self._trigger_alert(
                            alert_type=config.alert_type,
                            alert_level=config.alert_level,
                            message=f"Spike de custo detectado: ${hourly_cost:.4f} na última hora",
                            current_value=hourly_cost,
                            threshold_value=config.threshold_value,
                            config=config,
                        )

    def _check_predictive_alerts(self, current_stats: Dict):
        """Verifica alertas preditivos de overage."""
        if not hasattr(self, "budget_limit") or len(self._cost_history) < 5:
            return

        # Análise de tendência de custo
        recent_costs = [point["total_cost"] for point in self._cost_history[-5:]]
        cost_trend = (recent_costs[-1] - recent_costs[0]) / len(recent_costs)

        # Projeta custo para o final da sessão
        current_cost = current_stats.get("total_cost_usd", 0)
        session_duration = current_stats.get("duration_seconds", 0)
        dataset_size = current_stats.get("dataset_size", 1)
        operations_count = current_stats.get("operations_count", 0)

        if operations_count > 0 and dataset_size > 0:
            progress_percent = min(operations_count / dataset_size, 1.0)
            if progress_percent > 0.1:  # Pelo menos 10% de progresso
                projected_final_cost = current_cost / progress_percent

                if projected_final_cost > self.budget_limit:
                    # Configura alerta preditivo
                    config = AlertConfig(
                        alert_type=AlertType.PREDICTIVE_OVERAGE,
                        alert_level=AlertLevel.WARNING,
                        threshold_value=self.budget_limit,
                        notification_methods=["console"],
                    )

                    if self._should_trigger_alert(
                        AlertType.PREDICTIVE_OVERAGE, self.budget_limit
                    ):
                        self._trigger_alert(
                            alert_type=AlertType.PREDICTIVE_OVERAGE,
                            alert_level=AlertLevel.WARNING,
                            message=f"Projeção indica overage: ${projected_final_cost:.4f} estimado",
                            current_value=projected_final_cost,
                            threshold_value=self.budget_limit,
                            config=config,
                            additional_data={
                                "progress_percent": progress_percent * 100,
                                "cost_trend": cost_trend,
                            },
                        )

    def _check_efficiency_alerts(self, current_stats: Dict):
        """Verifica alertas de queda de eficiência."""
        cost_per_item = current_stats.get("cost_per_item", 0)
        tokens_per_second = current_stats.get("tokens_per_second", 0)

        # Baseline de eficiência
        expected_cost_per_item = 0.05  # $0.05 baseline
        expected_tokens_per_second = 100  # 100 tokens/s baseline

        if cost_per_item > expected_cost_per_item * 2:  # 2x pior que esperado
            config = AlertConfig(
                alert_type=AlertType.EFFICIENCY_DROP,
                alert_level=AlertLevel.WARNING,
                threshold_value=expected_cost_per_item * 2,
                notification_methods=["console"],
            )

            if self._should_trigger_alert(AlertType.EFFICIENCY_DROP, cost_per_item):
                self._trigger_alert(
                    alert_type=AlertType.EFFICIENCY_DROP,
                    alert_level=AlertLevel.WARNING,
                    message=f"Queda de eficiência detectada: ${cost_per_item:.6f} por item",
                    current_value=cost_per_item,
                    threshold_value=expected_cost_per_item,
                    config=config,
                )

    def _check_cost_anomaly_alerts(self):
        """Verifica alertas de anomalias de custo."""
        if len(self._cost_history) < 10:
            return

        # Análise de anomalias baseada em desvio padrão
        recent_costs = [point["total_cost"] for point in self._cost_history[-10:]]
        avg_cost = sum(recent_costs) / len(recent_costs)

        # Calcula desvio padrão
        variance = sum((cost - avg_cost) ** 2 for cost in recent_costs) / len(
            recent_costs
        )
        std_dev = variance**0.5

        latest_cost = recent_costs[-1]

        # Anomalia se estiver 2+ desvios padrão acima da média
        if std_dev > 0 and latest_cost > avg_cost + (2 * std_dev):
            config = AlertConfig(
                alert_type=AlertType.COST_ANOMALY,
                alert_level=AlertLevel.WARNING,
                threshold_value=avg_cost + (2 * std_dev),
                notification_methods=["console"],
            )

            if self._should_trigger_alert(AlertType.COST_ANOMALY, latest_cost):
                self._trigger_alert(
                    alert_type=AlertType.COST_ANOMALY,
                    alert_level=AlertLevel.WARNING,
                    message=f"Anomalia de custo detectada: ${latest_cost:.6f} (${std_dev:.6f} σ)",
                    current_value=latest_cost,
                    threshold_value=avg_cost,
                    config=config,
                    additional_data={
                        "std_deviations": (latest_cost - avg_cost) / std_dev,
                        "avg_cost": avg_cost,
                    },
                )

    def _should_trigger_alert(
        self, alert_type: AlertType, threshold_value: float
    ) -> bool:
        """Verifica se deve disparar alerta considerando cooldown."""
        config_key = f"{alert_type.value}_{threshold_value}"
        if config_key not in self._alert_configs:
            return False

        config = self._alert_configs[config_key]
        last_alert_time = self._last_alert_times.get(alert_type, 0)
        current_time = time.time()

        return current_time - last_alert_time >= (config.cooldown_minutes * 60)

    def _trigger_alert(
        self,
        alert_type: AlertType,
        alert_level: AlertLevel,
        message: str,
        current_value: float,
        threshold_value: float,
        config: AlertConfig,
        session_id: str = None,
        additional_data: Dict = None,
    ):
        """Dispara um alerta."""
        alert_id = f"{alert_type.value}_{int(time.time())}"

        alert = BudgetAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            alert_level=alert_level,
            timestamp=time.time(),
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            session_id=session_id,
            additional_data=additional_data or {},
        )

        with self._lock:
            self._active_alerts.append(alert)
            self._alert_history.append(alert)
            self._last_alert_times[alert_type] = time.time()

        # Processa notificações
        self._process_alert_notifications(alert, config)

        # Executa handlers customizados
        if alert_type in self._custom_alert_handlers:
            for handler in self._custom_alert_handlers[alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"❌ Erro em handler customizado: {e}")

        # Salva alerta
        self._save_alert(alert)

    def _process_alert_notifications(self, alert: BudgetAlert, config: AlertConfig):
        """Processa notificações do alerta."""
        for method in config.notification_methods:
            if method == "console":
                self._send_console_notification(alert)
            elif method == "email" and self.email_config:
                self._send_email_notification(alert)
            elif method == "webhook":
                self._send_webhook_notification(alert)

    def _send_console_notification(self, alert: BudgetAlert):
        """Envia notificação para console."""
        level_icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🆘",
        }

        icon = level_icons.get(alert.alert_level, "🔔")
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")

        print(
            f"\n{icon} BUDGET ALERT [{alert.alert_level.value.upper()}] - {timestamp}"
        )
        print(f"   Type: {alert.alert_type.value}")
        print(f"   Message: {alert.message}")
        print(f"   Current: {alert.current_value:.6f}")
        print(f"   Threshold: {alert.threshold_value:.6f}")
        if alert.session_id:
            print(f"   Session: {alert.session_id}")
        if alert.additional_data:
            for key, value in alert.additional_data.items():
                print(f"   {key.title()}: {value}")
        print()

    def _send_email_notification(self, alert: BudgetAlert):
        """Envia notificação por email."""
        if not self.email_config or not MimeText or not MimeMultipart:
            print("📧 Email notification não disponível (dependências não encontradas)")
            return

        try:
            msg = MimeMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = self.email_config["to_email"]
            msg["Subject"] = (
                f"Budget Alert: {alert.alert_level.value.title()} - {alert.alert_type.value}"
            )

            body = f"""
Budget Alert Notification

Alert Level: {alert.alert_level.value.upper()}
Alert Type: {alert.alert_type.value}
Timestamp: {datetime.fromtimestamp(alert.timestamp).isoformat()}

Message: {alert.message}

Details:
- Current Value: {alert.current_value:.6f}
- Threshold: {alert.threshold_value:.6f}
- Session ID: {alert.session_id or 'N/A'}

Additional Data:
{json.dumps(alert.additional_data, indent=2)}

This is an automated notification from the Budget Monitoring System.
            """

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            )
            if self.email_config.get("use_tls", True):
                server.starttls()
            if "username" in self.email_config:
                server.login(
                    self.email_config["username"], self.email_config["password"]
                )

            server.send_message(msg)
            server.quit()

            print(f"📧 Email notification sent for alert {alert.alert_id}")

        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")

    def _send_webhook_notification(self, alert: BudgetAlert):
        """Envia notificação via webhook."""
        # Implementação de webhook pode ser adicionada aqui
        print(f"🔗 Webhook notification triggered for alert {alert.alert_id}")

    def _trigger_emergency_suspension(self, current_stats: Dict):
        """Dispara suspensão de emergência."""
        print("\n🆘 EMERGENCY SUSPENSION TRIGGERED!")
        print("   Budget limit critically exceeded")
        print("   Processing should be suspended immediately")

        if self._suspension_handler:
            try:
                self._suspension_handler(current_stats)
            except Exception as e:
                print(f"❌ Error in suspension handler: {e}")

        # Para o monitoramento
        self.stop_monitoring()

    def add_custom_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Adiciona handler customizado para tipo de alerta."""
        if alert_type not in self._custom_alert_handlers:
            self._custom_alert_handlers[alert_type] = []
        self._custom_alert_handlers[alert_type].append(handler)
        print(f"🔧 Handler customizado adicionado para {alert_type.value}")

    def set_suspension_handler(self, handler: Callable):
        """Define handler para suspensão de emergência."""
        self._suspension_handler = handler
        print("🛡️ Handler de suspensão configurado")

    def acknowledge_alert(self, alert_id: str):
        """Reconhece um alerta."""
        with self._lock:
            for alert in self._active_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    print(f"✅ Alert {alert_id} acknowledged")
                    return True
        return False

    def clear_acknowledged_alerts(self):
        """Remove alertas reconhecidos da lista ativa."""
        with self._lock:
            initial_count = len(self._active_alerts)
            self._active_alerts = [
                alert for alert in self._active_alerts if not alert.acknowledged
            ]
            cleared_count = initial_count - len(self._active_alerts)
            if cleared_count > 0:
                print(f"🧹 {cleared_count} alertas reconhecidos removidos")

    def get_active_alerts(self) -> List[BudgetAlert]:
        """Retorna alertas ativos."""
        return self._active_alerts.copy()

    def get_alert_history(self, hours_back: int = 24) -> List[BudgetAlert]:
        """Retorna histórico de alertas."""
        cutoff_time = time.time() - (hours_back * 3600)
        return [
            alert for alert in self._alert_history if alert.timestamp >= cutoff_time
        ]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de alertas."""
        with self._lock:
            total_alerts = len(self._alert_history)
            active_alerts = len(self._active_alerts)

            # Agrupa por tipo
            alerts_by_type = {}
            alerts_by_level = {}

            for alert in self._alert_history:
                alert_type = alert.alert_type.value
                alert_level = alert.alert_level.value

                alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
                alerts_by_level[alert_level] = alerts_by_level.get(alert_level, 0) + 1

            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "alerts_by_type": alerts_by_type,
                "alerts_by_level": alerts_by_level,
                "monitoring_active": self._monitoring_active,
                "configured_alerts": len(self._alert_configs),
            }

    def _save_alert(self, alert: BudgetAlert):
        """Salva alerta em arquivo."""
        try:
            alerts_file = self.storage_dir / "alerts.json"

            if alerts_file.exists():
                with open(alerts_file, "r", encoding="utf-8") as f:
                    alerts_data = json.load(f)
            else:
                alerts_data = []

            # Adiciona novo alerta
            alert_dict = asdict(alert)
            alert_dict["alert_type"] = alert.alert_type.value
            alert_dict["alert_level"] = alert.alert_level.value
            alerts_data.append(alert_dict)

            # Mantém apenas os últimos 1000 alertas
            if len(alerts_data) > 1000:
                alerts_data = alerts_data[-1000:]

            with open(alerts_file, "w", encoding="utf-8") as f:
                json.dump(alerts_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error saving alert: {e}")

    def _load_configurations(self):
        """Carrega configurações salvas."""
        try:
            config_file = self.storage_dir / "monitor_config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                for config_dict in config_data.get("alert_configs", []):
                    config = AlertConfig(
                        alert_type=AlertType(config_dict["alert_type"]),
                        alert_level=AlertLevel(config_dict["alert_level"]),
                        threshold_value=config_dict["threshold_value"],
                        enabled=config_dict.get("enabled", True),
                        cooldown_minutes=config_dict.get("cooldown_minutes", 60),
                        notification_methods=config_dict.get(
                            "notification_methods", ["console"]
                        ),
                    )
                    self.add_alert_config(config)

        except Exception as e:
            print(f"⚠️ Warning: Could not load configurations: {e}")

    def save_configurations(self):
        """Salva configurações atuais."""
        try:
            config_data = {
                "alert_configs": [
                    {
                        "alert_type": config.alert_type.value,
                        "alert_level": config.alert_level.value,
                        "threshold_value": config.threshold_value,
                        "enabled": config.enabled,
                        "cooldown_minutes": config.cooldown_minutes,
                        "notification_methods": config.notification_methods,
                    }
                    for config in self._alert_configs.values()
                ],
                "monitoring_interval": self._monitoring_interval,
            }

            config_file = self.storage_dir / "monitor_config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Configurações salvas em {config_file}")

        except Exception as e:
            print(f"❌ Error saving configurations: {e}")

    def export_alerts_report(self, filename: str = None) -> Path:
        """Exporta relatório de alertas."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"budget_alerts_report_{timestamp}.json"

        report_file = self.storage_dir / filename

        report_data = {
            "report_generated": datetime.now().isoformat(),
            "statistics": self.get_alert_statistics(),
            "active_alerts": [asdict(alert) for alert in self._active_alerts],
            "recent_alerts": [asdict(alert) for alert in self.get_alert_history(24)],
            "configurations": [
                asdict(config) for config in self._alert_configs.values()
            ],
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"📊 Relatório de alertas exportado: {report_file}")
        return report_file
