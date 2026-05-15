"""
SafeWatch Alert Tests
Unit tests for notification and snapshot generation.
"""

import unittest
from alerts.snapshot_builder import SnapshotBuilder
from alerts.alert_manager import AlertManager


class TestAlerts(unittest.TestCase):
    """Test suite for the alerts module."""

    def test_snapshot_builder_init(self):
        """Test SnapshotBuilder directory creation."""
        builder = SnapshotBuilder(output_dir="recordings/test_snapshots")
        self.assertTrue(builder._output_dir.exists())

    def test_alert_manager_init(self):
        """Test AlertManager setup."""
        builder = SnapshotBuilder()
        manager = AlertManager(telegram_bot=None, snapshot_builder=builder)
        self.assertIsNone(manager._bot)


if __name__ == "__main__":
    unittest.main()
