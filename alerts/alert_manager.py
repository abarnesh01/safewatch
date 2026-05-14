"""
SafeWatch Alert Manager
Orchestrates alert dispatch, snapshots, and cooldown management.
"""

import asyncio
from typing import List, Dict, Optional, Any
from loguru import logger

from threats.fight_detector import ThreatEvent
from alerts.snapshot_builder import SnapshotBuilder
from alerts.telegram_bot import TelegramAlertBot


class AlertManager:
    """Manages the entire alerting lifecycle from detection to notification."""

    def __init__(self, telegram_bot: Optional[TelegramAlertBot],
                 snapshot_builder: SnapshotBuilder) -> None:
        self._bot = telegram_bot
        self._snapshot_builder = snapshot_builder
        logger.info("AlertManager initialized")

    async def handle_threats(self, events: List[ThreatEvent], 
                             frame: Any, 
                             camera_name: str) -> None:
        """Process detected threats and trigger notifications."""
        for event in events:
            # 1. Build Snapshot
            snapshot_path = self._snapshot_builder.build_snapshot(frame, event, camera_name)
            
            # 2. Build Alert Message
            message = self._build_alert_message(event, camera_name)
            
            # 3. Dispatch to Telegram
            if self._bot:
                await self._bot.send_alert(message, snapshot_path)
                logger.info("Alert dispatched for {}: {}", camera_name, event.threat_type)

    def _build_alert_message(self, event: ThreatEvent, camera_name: str) -> str:
        """Format the Telegram alert message."""
        return (
            f"🚨 <b>{event.threat_type.upper()} DETECTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Severity:</b> {event.severity}\n"
            f"<b>Camera:</b> {camera_name}\n"
            f"<b>Confidence:</b> {event.confidence:.2f}\n"
            f"<b>Description:</b> {event.description}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
