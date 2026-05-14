"""
SafeWatch — SafeWatchTelegramBot
Async Telegram bot for sending threat alerts using python-telegram-bot v20+.
"""

import os
import io
import asyncio
import time
import threading
from typing import Optional
from datetime import datetime

from loguru import logger

try:
    from telegram import Bot
    from telegram.error import TelegramError, RetryAfter, TimedOut
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available — Telegram alerts disabled")


class SafeWatchTelegramBot:
    """Async Telegram bot for sending threat alerts with retry logic and rate limiting."""

    def __init__(self, config: dict):
        self._config = config.get("telegram", {})
        self._enabled = self._config.get("enabled", True) and TELEGRAM_AVAILABLE
        self._max_retries = self._config.get("max_retries", 3)
        self._send_snapshot = self._config.get("send_snapshot", True)
        self._agents = self._config.get("agents", {})
        self._bot: Optional["Bot"] = None
        self._lock = threading.Lock()
        self._last_send_time = 0.0
        self._min_send_interval = 0.034  # ~30 msgs/sec Telegram limit

        # Resolve bot token from env
        token_cfg = self._config.get("bot_token", "")
        if token_cfg.startswith("${") and token_cfg.endswith("}"):
            env_key = token_cfg[2:-1]
            self._token = os.environ.get(env_key, "")
        else:
            self._token = token_cfg

        if self._enabled and self._token:
            try:
                self._bot = Bot(token=self._token)
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self._bot = None
                self._enabled = False
        elif self._enabled:
            logger.warning("Telegram bot token not configured — alerts disabled")
            self._enabled = False

        # Resolve agent chat IDs from env
        self._resolved_agents = {}
        for agent_id, agent_cfg in self._agents.items():
            chat_id_cfg = agent_cfg.get("chat_id", "")
            if chat_id_cfg.startswith("${") and chat_id_cfg.endswith("}"):
                env_key = chat_id_cfg[2:-1]
                chat_id = os.environ.get(env_key, "")
            else:
                chat_id = chat_id_cfg

            self._resolved_agents[agent_id] = {
                "chat_id": chat_id,
                "name": agent_cfg.get("name", agent_id),
                "cameras": agent_cfg.get("cameras", []),
            }

    def __repr__(self) -> str:
        return (
            f"SafeWatchTelegramBot(enabled={self._enabled}, "
            f"agents={list(self._resolved_agents.keys())})"
        )

    def _get_agents_for_camera(self, camera_id: str) -> list[str]:
        """Get agent IDs that should receive alerts for a camera."""
        agents = []
        for agent_id, agent_cfg in self._resolved_agents.items():
            if camera_id in agent_cfg["cameras"]:
                agents.append(agent_id)
        if not agents:
            agents = list(self._resolved_agents.keys())
        return agents

    async def send_threat_alert(
        self,
        threat_event: dict,
        camera_id: str,
        snapshot: Optional[bytes] = None,
        agent_id: Optional[str] = None,
        camera_name: str = "",
    ):
        """
        Send a threat alert to the appropriate Telegram agent(s).

        Args:
            threat_event: Dict with threat details
            camera_id: Camera ID
            snapshot: Optional JPEG bytes
            agent_id: Specific agent to alert, or None for auto-routing
            camera_name: Human-readable camera name
        """
        if not self._enabled or self._bot is None:
            logger.debug("Telegram disabled — alert not sent")
            return

        target_agents = [agent_id] if agent_id else self._get_agents_for_camera(camera_id)

        threat_type = threat_event.get("threat_type", "UNKNOWN")
        confidence = threat_event.get("confidence", 0.0)
        severity = threat_event.get("severity", "LOW")
        persons = threat_event.get("persons_involved", [])
        description = threat_event.get("description", "")
        ts = threat_event.get("timestamp", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        severity_emoji = {
            "LOW": "🟡",
            "MEDIUM": "🟠",
            "HIGH": "🔴",
            "CRITICAL": "🚨",
        }.get(severity, "⚠️")

        message = (
            f"🚨 SAFEWATCH ALERT\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Threat: {threat_type}\n"
            f"📍 Camera: {camera_name or camera_id} ({camera_id})\n"
            f"🕐 Time: {ts}\n"
            f"📊 Confidence: {confidence:.0%}\n"
            f"{severity_emoji} Severity: {severity}\n"
            f"👥 Persons Involved: {len(persons)}\n"
            f"📝 Details: {description}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        for aid in target_agents:
            agent = self._resolved_agents.get(aid)
            if agent is None or not agent["chat_id"]:
                logger.warning(f"No chat ID for agent {aid}")
                continue

            chat_id = agent["chat_id"]
            await self._send_with_retry(chat_id, message, snapshot)

    async def _send_with_retry(
        self,
        chat_id: str,
        message: str,
        snapshot: Optional[bytes] = None,
    ):
        """Send message with exponential backoff retry."""
        for attempt in range(self._max_retries):
            try:
                # Rate limiting
                elapsed = time.time() - self._last_send_time
                if elapsed < self._min_send_interval:
                    await asyncio.sleep(self._min_send_interval - elapsed)

                if snapshot and self._send_snapshot:
                    photo = io.BytesIO(snapshot)
                    photo.name = "alert.jpg"
                    await self._bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=message[:1024],  # Telegram caption limit
                    )
                else:
                    await self._bot.send_message(
                        chat_id=chat_id,
                        text=message,
                    )

                self._last_send_time = time.time()
                logger.info(f"Alert sent to chat_id={chat_id}")
                return

            except RetryAfter as e:
                wait = e.retry_after
                logger.warning(f"Rate limited, retrying after {wait}s")
                await asyncio.sleep(wait)

            except TimedOut:
                wait = (2 ** attempt) * 1.0
                logger.warning(f"Telegram timeout, retry {attempt + 1}/{self._max_retries} after {wait}s")
                await asyncio.sleep(wait)

            except TelegramError as e:
                logger.error(f"Telegram error: {e}")
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to send alert after {self._max_retries} attempts")

            except Exception as e:
                logger.error(f"Unexpected error sending Telegram alert: {e}")
                break

    async def send_system_alert(self, message: str, all_agents: bool = True):
        """
        Send a system alert message.

        Args:
            message: Alert message text
            all_agents: If True, send to all agents. Otherwise just the first.
        """
        if not self._enabled or self._bot is None:
            return

        agents = list(self._resolved_agents.values())
        if not all_agents:
            agents = agents[:1]

        formatted = f"ℹ️ SAFEWATCH SYSTEM\n━━━━━━━━━━━━━━━━━━\n{message}"

        for agent in agents:
            if agent["chat_id"]:
                await self._send_with_retry(agent["chat_id"], formatted)

    async def send_daily_summary(self, stats: dict, agent_id: Optional[str] = None):
        """
        Send a daily summary report.

        Args:
            stats: Dict with keys: total, by_type, by_severity, avg_confidence
            agent_id: Specific agent, or None for all
        """
        if not self._enabled or self._bot is None:
            return

        total = stats.get("total", 0)
        by_type = stats.get("by_type", {})
        by_severity = stats.get("by_severity", {})

        type_lines = "\n".join(
            f"  • {t}: {c}" for t, c in sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        ) or "  None"

        severity_lines = "\n".join(
            f"  • {s}: {c}" for s, c in sorted(by_severity.items())
        ) or "  None"

        message = (
            f"📊 SAFEWATCH DAILY SUMMARY\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Date: {datetime.now().strftime('%d/%m/%Y')}\n"
            f"📈 Total Incidents: {total}\n\n"
            f"📋 By Type:\n{type_lines}\n\n"
            f"⚠️ By Severity:\n{severity_lines}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        if agent_id:
            agent = self._resolved_agents.get(agent_id)
            if agent and agent["chat_id"]:
                await self._send_with_retry(agent["chat_id"], message)
        else:
            for agent in self._resolved_agents.values():
                if agent["chat_id"]:
                    await self._send_with_retry(agent["chat_id"], message)

    async def test_connection(self) -> bool:
        """
        Test the Telegram bot connection.

        Returns:
            True if bot is reachable and token is valid.
        """
        if not self._enabled or self._bot is None:
            logger.warning("Telegram bot not enabled or not initialized")
            return False

        try:
            me = await self._bot.get_me()
            logger.info(f"Telegram bot connected: @{me.username} ({me.first_name})")
            return True
        except Exception as e:
            logger.error(f"Telegram bot connection test failed: {e}")
            return False

    def send_threat_alert_sync(
        self,
        threat_event: dict,
        camera_id: str,
        snapshot: Optional[bytes] = None,
        agent_id: Optional[str] = None,
        camera_name: str = "",
    ):
        """Synchronous wrapper for send_threat_alert."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.send_threat_alert(threat_event, camera_id, snapshot, agent_id, camera_name)
                )
            else:
                loop.run_until_complete(
                    self.send_threat_alert(threat_event, camera_id, snapshot, agent_id, camera_name)
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.send_threat_alert(threat_event, camera_id, snapshot, agent_id, camera_name)
            )
