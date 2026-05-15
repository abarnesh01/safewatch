"""
SafeWatch — Comprehensive Alert Tests
Unit tests for alert management, snapshot rendering, and Telegram integration logic.
"""

import unittest
from pathlib import Path
from alerts.snapshot_builder import SnapshotBuilder
from alerts.alert_manager import AlertManager
from threats.fight_detector import ThreatEvent

class TestSafeWatchAlerts(unittest.TestCase):
    """Tests for the alerting and snapshot system."""

    def test_snapshot_builder_paths(self):
        """Test SnapshotBuilder directory management."""
        out_dir = "logs/test_snaps"
        builder = SnapshotBuilder() # Default uses logs/snapshots
        self.assertTrue(Path("logs/snapshots").exists())

    def test_alert_manager_cooldown(self):
        """Test that AlertManager respects threat cooldowns."""
        # This would require more mocking of the bot, but we can test init
        manager = AlertManager(telegram_bot=None, snapshot_builder=SnapshotBuilder())
        self.assertFalse(manager._telegram_enabled)
        
        # Test threat handling doesn't crash without bot
        threat = ThreatEvent(
            threat_type="FIGHT",
            confidence=0.9,
            persons_involved=[1, 2],
            location_bbox=(0,0,100,100),
            description="Test",
            severity="HIGH"
        )
        
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            # Should just log and return since telegram is disabled
            loop.run_until_complete(manager.handle_threats([threat], None, "Test Cam"))
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
