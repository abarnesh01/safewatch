import asyncio
import httpx
from typing import Dict, Any, List
from loguru import logger

import threading

class WebhookManager:
    """Handles external HTTP webhook dispatching with retry logic."""
    
    def __init__(self, config: dict):
        self._config = config.get("webhooks", {})
        self._endpoints = self._config.get("endpoints", [])
        self._max_retries = self._config.get("max_retries", 3)
        self._queue = asyncio.Queue()
        self._running = False
        
    def start(self):
        self._running = True
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._worker())

        threading.Thread(
            target=run_loop,
            daemon=True
        ).start()
        logger.info("WebhookManager started.")
        
    def stop(self):
        self._running = False
        
    async def dispatch(self, payload: Dict[str, Any]):
        """Queue payload for all configured webhooks."""
        for endpoint in self._endpoints:
            if endpoint.get("enabled", False):
                await self._queue.put({"url": endpoint["url"], "payload": payload, "retries": 0})
                
    async def _worker(self):
        async with httpx.AsyncClient() as client:
            while self._running:
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    await self._send(client, task)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Webhook worker error: {e}")
                    
    async def _send(self, client: httpx.AsyncClient, task: dict):
        url = task["url"]
        payload = task["payload"]
        retries = task["retries"]
        
        try:
            response = await client.post(url, json=payload, timeout=5.0)
            if response.status_code >= 400:
                logger.warning(f"Webhook to {url} failed with status {response.status_code}")
                await self._retry(task)
            else:
                logger.debug(f"Webhook sent successfully to {url}")
        except Exception as e:
            logger.warning(f"Webhook exception for {url}: {e}")
            await self._retry(task)
            
    async def _retry(self, task: dict):
        if task["retries"] < self._max_retries:
            task["retries"] += 1
            await asyncio.sleep(2 ** task["retries"]) # Exponential backoff
            await self._queue.put(task)
        else:
            logger.error(f"Webhook failed permanently after {self._max_retries} retries: {task['url']}")
