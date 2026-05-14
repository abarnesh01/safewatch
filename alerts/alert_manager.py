"""
SafeWatch — AlertManager
Coordinates between ThreatEngine and Telegram bot with cooldown, routing, and queuing.
"""

import time
import threading
import asyncio
from collections import defaultdict
from typing import Optional
from queue import Queue, Empty
from datetime import datetime

from loguru import logger

from alerts.telegram_bot import SafeWatchTelegramBot
from alerts.snapshot_builder import SnapshotBuilder
from database.incident_logger import IncidentLogger
from threats.threat_engine import ThreatReport


class AlertManager:
    """
    Coordinates threat alerts: builds snapshots, routes to agents,
    implements cooldowns, and manages alert queues.
    """

    def __init__(
        self,
        config: dict,
        telegram_bot: SafeWatchTelegramBot,
        incident_logger: IncidentLogger,
    ):
        self._config = config
        self._telegram = telegram_bot
        self._logger = incident_logger
        self._snapshot_builder = SnapshotBuilder()
        self._lock = threading.Lock()

        telegram_config = config.get("telegram", {})
        self._cooldown_seconds = telegram_config.get("alert_cooldown_seconds", 30)
        self._send_snapshot = telegram_config.get("send_snapshot", True)

        self._cooldowns: dict[str, float] = defaultdict(float)
        self._active_alerts: list[dict] = []
        self._alert_queue: Queue = Queue(maxsize=100)
        self._alert_counter = 0

        # Camera name lookup
        self._camera_names: dict[str, str] = {}
        for cam in config.get("cameras", []):
            self._camera_names[cam["id"]] = cam.get("name", cam["id"])

        # Agent camera mapping
        self._camera_agents: dict[str, list[str]] = defaultdict(list)
        agents = telegram_config.get("agents", {})
        for agent_id, agent_cfg in agents.items():
            for cam_id in agent_cfg.get("cameras", []):
                self._camera_agents[cam_id].append(agent_id)

        # Start alert processing thread
        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_queue_loop,
            name="AlertManager-Queue",
            daemon=True,
        )
        self._process_thread.start()

        logger.info("AlertManager initialized")

    def __repr__(self) -> str:
        return (
            f"AlertManager(active_alerts={len(self._active_alerts)}, "
            f"queue_size={self._alert_queue.qsize()})"
        )

    def process_threat_report(self, threat_report: ThreatReport, frame=None):
        """
        Process a threat report from ThreatEngine.

        Args:
            threat_report: ThreatReport with detected threats
            frame: Original frame for snapshot building
        """
        if not threat_report.threats_detected:
            return

        camera_id = threat_report.camera_id
        camera_name = self._camera_names.get(camera_id, camera_id)

        for threat in threat_report.threats_detected:
            # Check cooldown
            cooldown_key = f"{camera_id}:{threat.threat_type}"
            with self._lock:
                last_time = self._cooldowns.get(cooldown_key, 0)
                now = time.time()
                if now - last_time < self._cooldown_seconds:
                    logger.debug(
                        f"Alert suppressed (cooldown): {threat.threat_type} on {camera_id}"
                    )
                    continue
                self._cooldowns[cooldown_key] = now

            # Build threat event dict
            threat_dict = {
                "threat_type": threat.threat_type,
                "confidence": threat.confidence,
                "severity": threat.severity,
                "persons_involved": threat.persons_involved,
                "description": threat.description,
                "location_bbox": threat.location_bbox,
                "timestamp": threat.timestamp or datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "alert_sent": 0,
            }

            # Build snapshot
            snapshot_bytes = None
            snapshot_path = ""
            if self._send_snapshot and frame is not None:
                snapshot_bytes = self._snapshot_builder.build(
                    frame, threat_dict, camera_id,
                    threat_report.timestamp, camera_name,
                )
                if snapshot_bytes:
                    snapshot_path = self._snapshot_builder.save_snapshot(
                        snapshot_bytes, camera_id, threat_report.timestamp,
                    )

            # Log to database
            incident_id = self._logger.log_threat(
                threat_dict, camera_id,
                snapshot_path=snapshot_path,
            )

            # Queue for sending
            alert_data = {
                "threat_dict": threat_dict,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "snapshot_bytes": snapshot_bytes,
                "incident_id": incident_id,
            }

            try:
                self._alert_queue.put_nowait(alert_data)
            except Exception:
                logger.warning("Alert queue full — dropping alert")

            # Track active alert
            self._alert_counter += 1
            with self._lock:
                self._active_alerts.append({
                    "id": self._alert_counter,
                    "incident_id": incident_id,
                    "threat_type": threat.threat_type,
                    "camera_id": camera_id,
                    "severity": threat.severity,
                    "time": time.time(),
                    "acknowledged": False,
                })

                # Keep only last 50 active alerts
                if len(self._active_alerts) > 50:
                    self._active_alerts = self._active_alerts[-50:]

    def _process_queue_loop(self):
        """Background thread that processes the alert send queue."""
        logger.info("Alert queue processor started")
        while self._running:
            try:
                alert_data = self._alert_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                # Determine target agents
                camera_id = alert_data["camera_id"]
                agents = self._camera_agents.get(camera_id, [])

                if not agents:
                    agents = [None]  # Send to default/all

                for agent_id in agents:
                    self._telegram.send_threat_alert_sync(
                        alert_data["threat_dict"],
                        alert_data["camera_id"],
                        snapshot=alert_data.get("snapshot_bytes"),
                        agent_id=agent_id,
                        camera_name=alert_data.get("camera_name", ""),
                    )

                logger.info(
                    f"Alert sent: {alert_data['threat_dict']['threat_type']} "
                    f"on {alert_data['camera_id']}"
                )

            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

        logger.info("Alert queue processor stopped")

    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        with self._lock:
            for alert in self._active_alerts:
                if alert["id"] == alert_id:
                    alert["acknowledged"] = True
                    if alert.get("incident_id"):
                        self._logger._db.acknowledge_incident(alert["incident_id"])
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
        return False

    def get_active_alerts(self) -> list[dict]:
        """Get list of recent unacknowledged alerts."""
        with self._lock:
            return [
                a for a in self._active_alerts
                if not a["acknowledged"] and time.time() - a["time"] < 3600
            ]

    def stop(self):
        """Stop the alert manager."""
        self._running = False
        if self._process_thread.is_alive():
            self._process_thread.join(timeout=5.0)
        logger.info("AlertManager stopped")
