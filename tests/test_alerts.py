"""
SafeWatch — Alert Module Tests
Tests for SnapshotBuilder, AlertManager, and TelegramBot.
"""

import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np


class TestSnapshotBuilder(unittest.TestCase):
    """Tests for the SnapshotBuilder class."""

    def test_snapshot_builder_creation(self):
        """Test SnapshotBuilder instantiation."""
        from alerts.snapshot_builder import SnapshotBuilder

        builder = SnapshotBuilder()
        self.assertIn("SnapshotBuilder", repr(builder))

    def test_build_snapshot(self):
        """Test building a snapshot from a frame and threat event."""
        from alerts.snapshot_builder import SnapshotBuilder

        builder = SnapshotBuilder()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        threat = {
            "threat_type": "FIGHT",
            "confidence": 0.92,
            "severity": "HIGH",
            "persons_involved": [1, 2],
            "location_bbox": (100, 100, 300, 300),
        }

        result = builder.build(frame, threat, "CAM-01", time.time(), "Main Entrance")
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_save_snapshot(self):
        """Test saving snapshot to disk."""
        import tempfile
        from pathlib import Path
        from alerts.snapshot_builder import SnapshotBuilder

        builder = SnapshotBuilder()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        threat = {
            "threat_type": "FALL",
            "confidence": 0.85,
            "severity": "MEDIUM",
            "persons_involved": [1],
            "location_bbox": (50, 50, 200, 400),
        }

        jpeg_bytes = builder.build(frame, threat, "CAM-02", time.time())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = builder.save_snapshot(jpeg_bytes, "CAM-02", time.time(), tmpdir)
            self.assertTrue(Path(path).exists())
            self.assertTrue(path.endswith(".jpg"))

    def test_empty_frame_snapshot(self):
        """Test snapshot with minimum size frame."""
        from alerts.snapshot_builder import SnapshotBuilder

        builder = SnapshotBuilder()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        threat = {"threat_type": "TEST", "confidence": 0.5, "severity": "LOW",
                  "persons_involved": [], "location_bbox": (10, 10, 50, 50)}

        result = builder.build(frame, threat, "TEST", time.time())
        self.assertIsInstance(result, bytes)


class TestTelegramBot(unittest.TestCase):
    """Tests for the SafeWatchTelegramBot class."""

    def test_bot_creation_without_token(self):
        """Test bot creation without valid token (graceful degradation)."""
        from alerts.telegram_bot import SafeWatchTelegramBot

        config = {
            "telegram": {
                "enabled": True,
                "bot_token": "",
                "agents": {},
                "max_retries": 3,
                "send_snapshot": True,
            }
        }
        bot = SafeWatchTelegramBot(config)
        self.assertFalse(bot._enabled)

    def test_bot_disabled(self):
        """Test bot when explicitly disabled."""
        from alerts.telegram_bot import SafeWatchTelegramBot

        config = {
            "telegram": {
                "enabled": False,
                "bot_token": "fake",
                "agents": {},
            }
        }
        bot = SafeWatchTelegramBot(config)
        # Should not crash even when disabled
        self.assertIn("SafeWatchTelegramBot", repr(bot))

    def test_agent_routing(self):
        """Test camera-to-agent routing logic."""
        from alerts.telegram_bot import SafeWatchTelegramBot

        config = {
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "agents": {
                    "agent_main": {
                        "chat_id": "123",
                        "name": "Main",
                        "cameras": ["CAM-01", "CAM-02"],
                    },
                    "agent_parking": {
                        "chat_id": "456",
                        "name": "Parking",
                        "cameras": ["CAM-03"],
                    },
                },
            }
        }
        bot = SafeWatchTelegramBot(config)
        agents = bot._get_agents_for_camera("CAM-01")
        self.assertIn("agent_main", agents)
        self.assertNotIn("agent_parking", agents)

        agents_cam3 = bot._get_agents_for_camera("CAM-03")
        self.assertIn("agent_parking", agents_cam3)


class TestDatabaseManager(unittest.TestCase):
    """Tests for the DatabaseManager class."""

    def test_database_creation(self):
        """Test database creates tables successfully."""
        import tempfile
        from pathlib import Path
        from database.db_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            db = DatabaseManager(db_path)
            self.assertTrue(Path(db_path).exists())

    def test_log_and_retrieve_incident(self):
        """Test logging and retrieving an incident."""
        import tempfile
        from pathlib import Path
        from database.db_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            db = DatabaseManager(db_path)

            incident_id = db.log_incident({
                "camera_id": "CAM-01",
                "timestamp": "2024-01-01 12:00:00",
                "threat_type": "FIGHT",
                "confidence": 0.9,
                "severity": "HIGH",
                "persons_involved": 2,
                "description": "Test fight",
            })

            self.assertGreater(incident_id, 0)

            incidents = db.get_recent_incidents(n=5)
            self.assertEqual(len(incidents), 1)
            self.assertEqual(incidents[0]["threat_type"], "FIGHT")

    def test_daily_stats(self):
        """Test daily statistics calculation."""
        import tempfile
        from pathlib import Path
        from database.db_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            db = DatabaseManager(db_path)

            stats = db.get_daily_stats("2024-01-01")
            self.assertEqual(stats["total_incidents"], 0)
            self.assertEqual(stats["by_type"], {})


if __name__ == "__main__":
    unittest.main()
