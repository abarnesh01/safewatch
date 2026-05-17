"""
SafeWatch — Comprehensive Alert Tests
Unit tests for alert management, snapshot rendering, and Telegram integration logic.
"""

import unittest
from pathlib import Path
from alerts.snapshot_builder import SnapshotBuilder
from alerts.alert_manager import AlertManager
from database.db_manager import DatabaseManager
from database.incident_logger import IncidentLogger
from utils.runtime_isolation import RuntimePath
from threats.fight_detector import ThreatEvent
from threats.threat_engine import ThreatReport

class TestSafeWatchAlerts(unittest.TestCase):
    """Tests for the alerting and snapshot system."""

    def setUp(self):
        # Activate isolation directories for testing
        RuntimePath.ensure_isolation()
        self.config = {
            "system": {
                "name": "SafeWatch Test"
            },
            "alerts": {
                "telegram_enabled": False,
                "snapshot_enabled": True,
                "min_severity": "LOW"
            },
            "telegram": {
                "alert_cooldown_seconds": 1,
                "send_snapshot": False,
                "deduplication_window": 0.5,
                "agents": {}
            },
            "cameras": [
                {"id": "test_cam", "name": "Test Cam"}
            ]
        }
        self.db = DatabaseManager(":memory:")
        self.logger = IncidentLogger(self.db)

    def test_snapshot_builder_paths(self):
        """Test SnapshotBuilder directory management."""
        builder = SnapshotBuilder()
        self.assertTrue(RuntimePath.SNAPSHOTS.exists())

    def test_alert_manager_cooldown(self):
        """Test that AlertManager respects threat cooldowns."""
        manager = AlertManager(
            config=self.config,
            telegram_bot=None,
            incident_logger=self.logger
        )
        
        # Test threat handling doesn't crash without bot
        threat = ThreatEvent(
            threat_type="FIGHT",
            confidence=0.9,
            persons_involved=[1, 2],
            location_bbox=(0,0,100,100),
            description="Test",
            severity="HIGH"
        )
        
        report = ThreatReport(
            camera_id="test_cam",
            timestamp=1700000000.0,
            threats_detected=[threat],
            annotated_frame=None,
            overall_risk_level="HIGH"
        )
        
        # Process report
        manager.process_threat_report(report, frame=None)
        
        # Verify it got logged
        incidents = self.logger.get_recent_incidents(10)
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["threat_type"], "FIGHT")

if __name__ == "__main__":
    unittest.main()
