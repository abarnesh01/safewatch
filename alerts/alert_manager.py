import time
import threading
from collections import defaultdict
from typing import Optional
from queue import PriorityQueue, Empty
from datetime import datetime
from loguru import logger
from alerts.telegram_bot import SafeWatchTelegramBot
from alerts.snapshot_builder import SnapshotBuilder
from database.incident_logger import IncidentLogger, IncidentEvent
from threats.threat_engine import ThreatReport

class AlertManager:
    """
    Coordinates threat alerts with priority-based dispatching and persistence.
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
        
        # 1. Priority-based dispatch queue
        self._alert_queue: PriorityQueue = PriorityQueue(maxsize=200)
        self._alert_counter = 0

        # ... (rest of init remains same)
        self._camera_names: dict[str, str] = {}
        for cam in config.get("cameras", []):
            self._camera_names[cam["id"]] = cam.get("name", cam["id"])

        self._camera_agents: dict[str, list[str]] = defaultdict(list)
        agents = telegram_config.get("agents", {})
        for agent_id, agent_cfg in agents.items():
            for cam_id in agent_cfg.get("cameras", []):
                self._camera_agents[cam_id].append(agent_id)

        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_queue_loop,
            name="AlertManager-Queue",
            daemon=True,
        )
        self._process_thread.start()

        from collections import deque
        self._recent_events: deque = deque(maxlen=50)
        self._dedup_window = telegram_config.get("deduplication_window", 5.0)

        logger.info("AlertManager initialized with Priority Queue engine")

    def process_threat_report(self, threat_report: ThreatReport, frame=None):
        """Process threat with priority escalation and cooldown bypass."""
        if not threat_report.threats_detected:
            return

        camera_id = threat_report.camera_id
        camera_name = self._camera_names.get(camera_id, camera_id)

        for threat in threat_report.threats_detected:
            # 2. Critical Bypass: CRITICAL threats ignore standard cooldown
            is_critical = threat.severity == "CRITICAL"
            
            if not is_critical and self._is_duplicate(threat, camera_id):
                continue

            cooldown_key = f"{camera_id}:{threat.threat_type}"
            with self._lock:
                last_time = self._cooldowns.get(cooldown_key, 0)
                now = time.time()
                if not is_critical and (now - last_time < self._cooldown_seconds):
                    continue
                self._cooldowns[cooldown_key] = now

            # 3. Dynamic Smart Prioritization
            severity_weights = {"CRITICAL": 0, "HIGH": 10, "MEDIUM": 20, "LOW": 30}
            base_priority = severity_weights.get(threat.severity, 40)
            
            # Refinement based on confidence and behavior persistence
            persistence = getattr(threat, "behavior_score", 0.0)
            priority_refinement = int((1.0 - threat.confidence) * 5) + int((1.0 - persistence) * 5)
            final_priority = max(0, base_priority + priority_refinement)

            threat_dict = {
                "threat_type": threat.threat_type,
                "confidence": threat.confidence,
                "severity": threat.severity,
                "persistence": persistence,
                "urgency_score": 100 - final_priority,
                "persons_involved": threat.persons_involved,
                "description": threat.description,
                "location_bbox": threat.location_bbox,
                "timestamp": threat.timestamp or datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "alert_sent": 0,
            }

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

            # Standardize timestamp
            evt_ts = datetime.now()
            if hasattr(threat, "timestamp") and threat.timestamp:
                if isinstance(threat.timestamp, (int, float)):
                    evt_ts = datetime.fromtimestamp(threat.timestamp)
                elif isinstance(threat.timestamp, datetime):
                    evt_ts = threat.timestamp
                elif isinstance(threat.timestamp, str):
                    try:
                        evt_ts = datetime.strptime(threat.timestamp, "%d/%m/%Y %H:%M:%S")
                    except ValueError:
                        try:
                            evt_ts = datetime.fromisoformat(threat.timestamp)
                        except ValueError:
                            pass

            incident_event = IncidentEvent(
                camera_id=camera_id,
                threat_type=threat.threat_type,
                severity=threat.severity,
                confidence=threat.confidence,
                snapshot_path=snapshot_path,
                description=threat.description,
                timestamp=evt_ts,
                metadata=threat_dict
            )
            incident_id = self._logger.log_incident(incident_event)

            # 4. Auto Forensic Export for CRITICAL
            if is_critical and incident_id is not None:
                threading.Thread(target=self._logger.export_forensic_bundle, args=(incident_id,), daemon=True).start()

            alert_data = {
                "threat_dict": threat_dict,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "snapshot_bytes": snapshot_bytes,
                "incident_id": incident_id,
            }

            try:
                # Store as (priority, timestamp, data) to ensure FIFO for same priority
                self._alert_queue.put_nowait((final_priority, time.time(), alert_data))
            except Exception:
                logger.error(f"Alert queue overflow! Dropping {threat.threat_type}")

            self._record_event(threat, camera_id, incident_id)
            self._alert_counter += 1

    def _is_duplicate(self, threat, camera_id: str) -> bool:
        """Check for spatial and temporal duplication."""
        now = time.time()
        for event in self._recent_events:
            if event["camera_id"] == camera_id and event["type"] == threat.threat_type:
                if now - event["time"] < self._dedup_window:
                    iou = self._calculate_iou(threat.location_bbox, event["bbox"])
                    if iou > 0.5:
                        return True
        return False

    def _calculate_iou(self, box1, box2) -> float:
        if not box1 or not box2: return 0.0
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter_area
        return inter_area / (union_area + 1e-6)

    def _correlate_incident(self, threat, camera_id: str) -> Optional[dict]:
        now = time.time()
        for event in self._recent_events:
            if event["camera_id"] != camera_id and event["type"] == threat.threat_type:
                if now - event["time"] < 10.0:
                    return event
        return None

    def _record_event(self, threat, camera_id: str, incident_id: int):
        self._recent_events.append({
            "camera_id": camera_id,
            "type": threat.threat_type,
            "bbox": threat.location_bbox,
            "time": time.time(),
            "incident_id": incident_id,
        })

    def _process_queue_loop(self):
        """Background processor for Priority Queue."""
        logger.info("Alert priority processor active")
        while self._running:
            try:
                # Get item from PriorityQueue
                priority, ts, alert_data = self._alert_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                camera_id = alert_data["camera_id"]
                agents = self._camera_agents.get(camera_id, [None])

                for agent_id in agents:
                    # Retry logic for network failures
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self._telegram.send_threat_alert_sync(
                                alert_data["threat_dict"],
                                camera_id,
                                snapshot=alert_data.get("snapshot_bytes"),
                                agent_id=agent_id,
                                camera_name=alert_data.get("camera_name", ""),
                            )
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.error(f"Final alert failure for {camera_id}: {e}")
                            time.sleep(2 ** attempt) # Exponential backoff

                logger.info(f"Priority Alert ({priority}) sent for {camera_id}")
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
            finally:
                self._alert_queue.task_done()

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
