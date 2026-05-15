"""
SafeWatch — Main Entry Point
AI-Powered CCTV Threat Detection System
"""

import os
import sys
import time
import signal
import argparse
import asyncio
import threading
from pathlib import Path

import yaml
from loguru import logger
from dotenv import load_dotenv

# ─── Setup Logging ───────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{module}</cyan> — {message}")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_logging(config: dict):
    """Configure loguru logging from config."""
    system = config.get("system", {})
    log_level = system.get("log_level", "INFO")
    log_file = system.get("log_file", "logs/safewatch.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {module}:{function}:{line} — {message}",
    )


def print_banner():
    """Print the SafeWatch startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ███████╗ █████╗ ███████╗███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗ ║
║   ██╔════╝██╔══██╗██╔════╝██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║ ║
║   ███████╗███████║█████╗  █████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║ ║
║   ╚════██║██╔══██║██╔══╝  ██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║ ║
║   ███████║██║  ██║██║     ███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║ ║
║   ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝ ║
║                                                           ║
║   AI-Powered CCTV Threat Detection System  v1.0.0        ║
║   ─────────────────────────────────────────────────────   ║
║   🛡️  Real-time threat detection & alerting              ║
║   📹  Multi-camera monitoring                            ║
║   🤖  YOLOv8 + MediaPipe + LSTM                          ║
║   📱  Telegram instant alerts                             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def test_cameras(config: dict):
    """Test camera connections."""
    from capture.stream_manager import StreamManager

    logger.info("Testing camera connections...")
    sm = StreamManager(config)
    sm.start_all()

    time.sleep(3)

    status = sm.get_status()
    for cam_id, cam_status in status.items():
        connected = cam_status["connected"]
        icon = "✅" if connected else "❌"
        fps = cam_status["fps"]
        logger.info(f"  {icon} {cam_id}: connected={connected}, fps={fps:.1f}")

    sm.stop_all()
    logger.info("Camera test complete")


def test_telegram(config: dict):
    """Test Telegram bot connection."""
    from alerts.telegram_bot import SafeWatchTelegramBot

    logger.info("Testing Telegram bot connection...")
    bot = SafeWatchTelegramBot(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        success = loop.run_until_complete(bot.test_connection())
        if success:
            logger.info("✅ Telegram bot connection successful!")
            loop.run_until_complete(
                bot.send_system_alert("🧪 SafeWatch test message — connection verified!")
            )
        else:
            logger.error("❌ Telegram bot connection failed")
    except Exception as e:
        logger.error(f"❌ Telegram test error: {e}")
    finally:
        loop.close()


def run_dashboard(config: dict):
    """Launch the Streamlit dashboard in a subprocess."""
    import subprocess

    dashboard_config = config.get("dashboard", {})
    host = dashboard_config.get("host", "0.0.0.0")
    port = dashboard_config.get("port", 8501)

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--theme.base", "dark",
    ]

    logger.info(f"Starting dashboard at http://{host}:{port}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def main_loop(config: dict, show_display: bool = True):
    """Main processing loop."""

    # Import components
    from database.db_manager import DatabaseManager
    from database.incident_logger import IncidentLogger
    from capture.stream_manager import StreamManager
    from detection.person_detector import PersonDetector
    from detection.pose_estimator import PoseEstimator
    from detection.optical_flow import OpticalFlowAnalyzer
    from detection.zone_manager import ZoneManager
    from classifier.velocity_tracker import VelocityTracker
    from classifier.action_classifier import ActionClassifier
    from threats.threat_engine import ThreatEngine
    from alerts.telegram_bot import SafeWatchTelegramBot
    from alerts.alert_manager import AlertManager

    # ─── Initialize Components ────────────────────────────────────
    logger.info("Initializing SafeWatch components...")

    # 1. Database
    db_config = config.get("database", {})
    db_manager = DatabaseManager(db_config.get("path", "logs/safewatch.db"))
    incident_logger = IncidentLogger(db_manager)
    logger.info("✅ Database initialized")

    # 2. Stream Manager
    stream_manager = StreamManager(config)
    stream_manager.start_all()
    logger.info("✅ Camera streams started")

    # 3. Person Detector
    person_detector = PersonDetector(config)
    logger.info("✅ Person detector loaded")

    # 4. Pose Estimator
    pose_estimator = PoseEstimator(config)
    logger.info("✅ Pose estimator loaded")

    # 5. Optical Flow
    flow_analyzer = OpticalFlowAnalyzer(config)
    logger.info("✅ Optical flow analyzer ready")

    # 6. Zone Manager
    zone_manager = ZoneManager(config)
    logger.info("✅ Zone manager loaded")

    # 7. Velocity Tracker
    velocity_tracker = VelocityTracker()
    logger.info("✅ Velocity tracker ready")

    # 8. Action Classifier
    action_classifier = ActionClassifier(config)
    logger.info(f"✅ Action classifier ready ({action_classifier!r})")

    # 9. Threat Engine
    threat_engine = ThreatEngine(config, zone_manager)
    logger.info("✅ Threat engine initialized")

    # 10. Telegram Bot
    telegram_bot = SafeWatchTelegramBot(config)
    logger.info(f"✅ Telegram bot ready ({telegram_bot!r})")

    # 11. Alert Manager
    alert_manager = AlertManager(config, telegram_bot, incident_logger)
    logger.info("✅ Alert manager ready")

    # Start dashboard
    dashboard_proc = None
    try:
        dashboard_proc = run_dashboard(config)
        logger.info("✅ Dashboard started")
    except Exception as e:
        logger.warning(f"Dashboard failed to start: {e}")

    logger.info("🚀 SafeWatch is now running!")
    logger.info("Press Ctrl+C to stop.")

    # ─── Live Display Setup ───────────────────────────────────────
    if show_display:
        cv2.namedWindow("SafeWatch Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SafeWatch Monitor", 960, 540)
        logger.info("📺 Live monitor window opened — press 'q' to quit")
    display_fps = 0.0
    fps_timer = time.time()
    fps_frame_count = 0

    # ─── Graceful Shutdown Handler ────────────────────────────────
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ─── Processing Loop ─────────────────────────────────────────
    camera_ids = stream_manager.get_all_camera_ids()
    prev_frames: dict[str, any] = {}
    frame_counters: dict[str, int] = {cam_id: 0 for cam_id in camera_ids}

    while running:
        for cam_id in camera_ids:
            try:
                # 1. Get frame
                frame = stream_manager.get_frame(cam_id)
                if frame is None:
                    # Show "No Signal" placeholder on the monitor
                    if show_display:
                        res = config.get("cameras", [{}])[0].get("resolution", [640, 480])
                        no_signal = np.zeros((res[1], res[0], 3), dtype=np.uint8)
                        # Dark background with status text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        h_ns, w_ns = no_signal.shape[:2]

                        # SafeWatch branding
                        cv2.putText(no_signal, "SAFEWATCH", (w_ns // 2 - 110, h_ns // 2 - 60),
                                    font, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

                        # Camera status
                        stream_obj = stream_manager.get_stream(cam_id)
                        if stream_obj and stream_obj.is_connected():
                            status_text = "Buffering..."
                            status_color = (0, 255, 255)
                        else:
                            status_text = f"Waiting for {cam_id}..."
                            status_color = (0, 100, 255)

                        (tw, _), _ = cv2.getTextSize(status_text, font, 0.7, 2)
                        cv2.putText(no_signal, status_text,
                                    (w_ns // 2 - tw // 2, h_ns // 2),
                                    font, 0.7, status_color, 2, cv2.LINE_AA)

                        # Animated dots
                        n_dots = int(time.time() * 2) % 4
                        dots = "." * n_dots
                        cv2.putText(no_signal, dots,
                                    (w_ns // 2 + tw // 2, h_ns // 2),
                                    font, 0.7, status_color, 2, cv2.LINE_AA)

                        # Timestamp
                        ts = time.strftime("%H:%M:%S")
                        cv2.putText(no_signal, ts, (w_ns - 120, h_ns - 15),
                                    font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

                        cv2.imshow("SafeWatch Monitor", no_signal)
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            running = False
                    continue

                frame_counters[cam_id] += 1

                # Check frame skip
                cam_config = next(
                    (c for c in config.get("cameras", []) if c["id"] == cam_id),
                    {}
                )
                frame_skip = cam_config.get("frame_skip", 5)
                if frame_counters[cam_id] % frame_skip != 0:
                    continue

                # 2. Person detection
                persons = person_detector.detect(frame)

                # 3. Pose estimation
                poses = pose_estimator.estimate(frame, persons)

                # Update velocity tracker
                timestamp = time.time()
                for pose in poses:
                    velocity_tracker.update(pose.person_id, pose, timestamp)

                # Update action classifier buffers
                for pose in poses:
                    action_classifier.update_buffer(pose.person_id, pose)

                # 4. Optical flow
                flow_result = None
                if cam_id in prev_frames and prev_frames[cam_id] is not None:
                    flow_result = flow_analyzer.analyze(prev_frames[cam_id], frame)
                prev_frames[cam_id] = frame.copy()

                # 5. Threat analysis
                frame_data = {
                    "frame": frame,
                    "camera_id": cam_id,
                    "timestamp": timestamp,
                    "persons": persons,
                    "poses": poses,
                    "flow_result": flow_result,
                    "velocity_tracker": velocity_tracker,
                }

                threat_report = threat_engine.analyze(frame_data)

                # 6. Process alerts
                if threat_report.threats_detected:
                    alert_manager.process_threat_report(threat_report, frame)

                # 7. Update database
                stream = stream_manager.get_stream(cam_id)
                if stream:
                    db_manager.update_camera_status(cam_id, {
                        "status": "online" if stream.is_connected() else "offline",
                        "fps": stream.get_fps(),
                        "frames_processed": frame_counters[cam_id],
                        "threats_today": 0,
                    })

                # ─── 8. Live Display ──────────────────────────────
                if show_display:
                    display = frame.copy()

                    # Draw person bounding boxes
                    display = person_detector.draw_detections(display, persons)

                    # Draw pose skeletons
                    display = pose_estimator.draw_skeleton(display, poses)

                    # Draw threat overlays from the engine
                    if threat_report.annotated_frame is not None:
                        display = threat_report.annotated_frame.copy()
                        display = person_detector.draw_detections(display, persons)
                        display = pose_estimator.draw_skeleton(display, poses)

                    # Draw zones if available
                    display = zone_manager.draw_zones(display)

                    # ── HUD overlay ────────────────────────────────
                    h_disp, w_disp = display.shape[:2]

                    # Calculate FPS
                    fps_frame_count += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        display_fps = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_timer = time.time()

                    # Semi-transparent HUD bar at the top
                    hud_h = 38
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (w_disp, hud_h), (20, 20, 20), -1)
                    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                    # HUD text
                    risk = threat_report.overall_risk_level
                    risk_colors = {
                        "SAFE": (0, 200, 0), "LOW": (0, 255, 255),
                        "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255),
                        "CRITICAL": (255, 0, 128),
                    }
                    risk_col = risk_colors.get(risk, (200, 200, 200))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y_text = 26

                    cv2.putText(display, "SAFEWATCH", (10, y_text),
                                font, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, f"| {cam_id}", (140, y_text),
                                font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                    cv2.putText(display, f"FPS: {display_fps:.1f}", (260, y_text),
                                font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(display, f"Persons: {len(persons)}", (380, y_text),
                                font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                    cv2.putText(display, f"Poses: {len(poses)}", (510, y_text),
                                font, 0.5, (200, 150, 255), 1, cv2.LINE_AA)

                    risk_text = f"RISK: {risk}"
                    (tw, _), _ = cv2.getTextSize(risk_text, font, 0.6, 2)
                    cv2.putText(display, risk_text, (w_disp - tw - 15, y_text),
                                font, 0.6, risk_col, 2, cv2.LINE_AA)

                    # Threat count bottom bar
                    n_threats = len(threat_report.threats_detected)
                    if n_threats > 0:
                        bar_y = h_disp - 32
                        overlay2 = display.copy()
                        cv2.rectangle(overlay2, (0, bar_y), (w_disp, h_disp), (0, 0, 80), -1)
                        cv2.addWeighted(overlay2, 0.7, display, 0.3, 0, display)
                        threat_text = f"!! {n_threats} THREAT{'S' if n_threats > 1 else ''} DETECTED"
                        cv2.putText(display, threat_text, (10, h_disp - 10),
                                    font, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                    # Timestamp
                    ts_text = time.strftime("%H:%M:%S")
                    (tw2, _), _ = cv2.getTextSize(ts_text, font, 0.45, 1)
                    cv2.putText(display, ts_text, (w_disp - tw2 - 10, h_disp - 10),
                                font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

                    cv2.imshow("SafeWatch Monitor", display)

            except Exception as e:
                logger.error(f"[{cam_id}] Processing error: {e}")

        # Handle display key events + small sleep
        if show_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("'q' pressed — shutting down...")
                running = False
        else:
            time.sleep(0.01)

    # ─── Cleanup ──────────────────────────────────────────────────
    logger.info("Shutting down SafeWatch...")

    if show_display:
        cv2.destroyAllWindows()

    stream_manager.stop_all()
    alert_manager.stop()
    threat_engine.shutdown()
    pose_estimator.close()

    if dashboard_proc:
        dashboard_proc.terminate()

    logger.info("👋 SafeWatch stopped. Goodbye!")


def main():
    """Parse arguments and run SafeWatch."""
    parser = argparse.ArgumentParser(
        description="SafeWatch — AI-Powered CCTV Threat Detection System",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--test-cameras", action="store_true",
        help="Test camera connections and exit",
    )
    parser.add_argument(
        "--test-telegram", action="store_true",
        help="Test Telegram bot connection and exit",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Start only the Streamlit dashboard",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless without the live monitor window",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    print_banner()

    system = config.get("system", {})
    logger.info(f"SafeWatch v{system.get('version', '1.0.0')}")
    logger.info(f"Debug mode: {system.get('debug', False)}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Config: {args.config}")

    # Handle sub-commands
    if args.test_cameras:
        test_cameras(config)
        return

    if args.test_telegram:
        test_telegram(config)
        return

    if args.dashboard_only:
        proc = run_dashboard(config)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    # Run the main system
    main_loop(config, show_display=not args.no_display)


if __name__ == "__main__":
    main()
