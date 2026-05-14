"""
SafeWatch Telegram Bot
Asynchronous alert dispatching via Telegram Bot API.
"""

import asyncio
from pathlib import Path
from typing import Optional, List
import telegram
from telegram.constants import ParseMode
from loguru import logger


class TelegramAlertBot:
    """Handles asynchronous dispatch of alerts and snapshots to Telegram."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = chat_id
        self._bot: Optional[telegram.Bot] = None
        
        if token and chat_id:
            try:
                self._bot = telegram.Bot(token=token)
                logger.info("TelegramAlertBot initialized")
            except Exception as exc:
                logger.error("Failed to initialize Telegram Bot: {}", exc)

    async def send_alert(self, message: str, 
                         image_path: Optional[str] = None) -> bool:
        """Dispatch a message with an optional image."""
        if not self._bot:
            return False

        try:
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as photo:
                    await self._bot.send_photo(
                        chat_id=self._chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode=ParseMode.HTML
                    )
            else:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=message,
                    parse_mode=ParseMode.HTML
                )
            return True
        except Exception as exc:
            logger.error("Telegram dispatch failed: {}", exc)
            return False

    async def send_system_status(self, status: str) -> bool:
        """Send system status updates."""
        msg = f"<b>[SafeWatch System Status]</b>\n{status}"
        return await self.send_alert(msg)
