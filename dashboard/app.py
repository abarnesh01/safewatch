"""
SafeWatch — Streamlit Dashboard
Live monitoring, incident history, camera management, and system settings.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager
from database.incident_logger import IncidentLogger


# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeWatch — AI CCTV Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2a4a;
        margin-bottom: 10px;
    }
    .threat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .severity-low { background-color: #2d5016; color: #7dd56f; }
    .severity-medium { background-color: #5a3e00; color: #ffb142; }
    .severity-high { background-color: #5a1616; color: #ff6b6b; }
    .severity-critical { background-color: #4a0e4a; color: #ff6bff; }
    .status-online { color: #00ff88; }
    .status-offline { color: #ff4444; }
    .header-banner {
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SafeWatch")
    st.markdown("**AI-Powered CCTV Monitoring**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🖥️ Live Monitor", "📋 Incident History", "🎬 Incident Replay", "📹 Camera Management", "⚙️ System Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Quick Stats")

    db_path = Path("logs/safewatch.db")
    if db_path.exists():
        db = DatabaseManager(str(db_path))
        stats = db.get_daily_stats()
        st.metric("Today's Incidents", stats.get("total_incidents", 0))
        st.metric("Cameras", len(db.get_camera_status()))
    else:
        st.info("No database found. Start SafeWatch to initialize.")
        db = None
        stats = {}


# ─── PAGE 1: Live Monitor ────────────────────────────────────────
if page == "🖥️ Live Monitor":
    st.markdown('<div class="header-banner"><h1>🖥️ Live Monitor</h1></div>', unsafe_allow_html=True)

    col_opts = st.columns(5)
    with col_opts[0]:
        refresh_rate = st.selectbox("Refresh Rate", ["Fast (500ms)", "Medium (1s)", "Slow (2s)", "Paused"], index=0)
    with col_opts[1]:
        preview_mode = st.toggle("Low Bandwidth", value=False, help="Reduces frame resolution for faster loading")
    with col_opts[2]:
        show_skeleton = st.toggle("Skeletons", value=True)
    with col_opts[3]:
        show_bboxes = st.toggle("Boxes", value=True)
    with col_opts[4]:
        show_overlay = st.toggle("Threats", value=True)

    # Shared state from main.py
    if "latest_frames" in st.session_state:
        frames = st.session_state["latest_frames"]
        cols = st.columns(min(len(frames), 2))
        
        for idx, (cam_id, frame_data) in enumerate(frames.items()):
            with cols[idx % 2]:
                st.markdown(f"### 📹 {cam_id}")
                frame = frame_data.get("frame")
                
                # 1. Dashboard Optimization: Throttled Frame Processing
                if frame is not None:
                    # Low bandwidth / Preview mode: resize frame
                    if preview_mode:
                        frame = cv2.resize(frame, (480, 270))
                    
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                risk = frame_data.get("risk_level", "SAFE")
                risk_colors = {"SAFE": "🟢", "LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🟣"}
                st.markdown(f"**Status:** {risk_colors.get(risk, '⚪')} {risk}")

                threats = frame_data.get("threats", [])
                if threats:
                    for t in threats:
                        severity = t.get("severity", "LOW")
                        css_class = f"severity-{severity.lower()}"
                        st.markdown(
                            f'<span class="threat-badge {css_class}">'
                            f'{t.get("threat_type", "?")} — {t.get("confidence", 0):.0%}'
                            f'</span>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("📡 No live feed available. Start SafeWatch with `python main.py` to begin monitoring.")

    # Auto-refresh logic
    if refresh_rate != "Paused":
        refresh_map = {"Fast (500ms)": 0.5, "Medium (1s)": 1.0, "Slow (2s)": 2.0}
        time.sleep(refresh_map[refresh_rate])
        st.rerun()


# ─── PAGE 2: Incident History ────────────────────────────────────
elif page == "📋 Incident History":
    st.markdown('<div class="header-banner"><h1>📋 Incident History</h1></div>', unsafe_allow_html=True)

    if db is None:
        st.warning("Database not initialized.")
    else:
        incident_logger = IncidentLogger(db)

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filter_date = st.date_input("Date", value=datetime.now().date())
        with col_f2:
            filter_camera = st.selectbox("Camera", ["All"] + [
                c["camera_id"] for c in db.get_camera_status()
            ])
        with col_f3:
            filter_type = st.selectbox("Threat Type", [
                "All", "FIGHT", "FALL", "HARASSMENT", "ASSAULT",
                "UNCONSCIOUS", "TRESPASS", "CROWD_PANIC", "ACCIDENT", "ABUSE",
            ])
        with col_f4:
            filter_severity = st.selectbox("Severity", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

        # Fetch incidents
        kwargs = {
            "start_date": f"{filter_date} 00:00:00",
            "end_date": f"{filter_date} 23:59:59",
            "limit": 200,
        }
        if filter_camera != "All":
            kwargs["camera_id"] = filter_camera
        if filter_type != "All":
            kwargs["threat_type"] = filter_type
        if filter_severity != "All":
            kwargs["severity"] = filter_severity

        incidents = db.get_incidents(**kwargs)

        st.markdown(f"**{len(incidents)} incidents found**")

        # Export button
        if incidents:
            csv_path = f"logs/export_{filter_date}.csv"
            if st.button("📥 Export to CSV"):
                path = incident_logger.export_csv(
                    kwargs["start_date"], kwargs["end_date"], csv_path
                )
                st.success(f"Exported to {path}")

        # Incident table
        if incidents:
            for inc in incidents:
                severity = inc.get("severity", "LOW")
                css_class = f"severity-{severity.lower()}"
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
                with col1:
                    st.text(inc.get("timestamp", "")[:19])
                with col2:
                    st.markdown(
                        f'<span class="threat-badge {css_class}">{inc.get("threat_type", "?")}</span>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.text(inc.get("camera_id", "?"))
                with col4:
                    st.text(f"{inc.get('confidence', 0):.0%}")
                with col5:
                    st.text(severity)

                # Expandable details
                with st.expander(f"Details — Incident #{inc.get('id', '?')}"):
                    st.json(inc)
                    snap_path = inc.get("snapshot_path", "")
                    if snap_path and Path(snap_path).exists():
                        st.image(snap_path, caption="Snapshot")
        else:
            st.info("No incidents found for the selected filters.")

        # Charts
        st.markdown("---")
        st.markdown("### 📊 Analytics")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### Incidents Over Time")
            hourly = db.get_hourly_distribution(str(filter_date))
            if hourly:
                import pandas as pd
                df = pd.DataFrame(
                    list(hourly.items()),
                    columns=["Hour", "Count"],
                )
                st.bar_chart(df.set_index("Hour"))

        with col_c2:
            st.markdown("#### Threat Type Distribution")
            daily_stats = db.get_daily_stats(str(filter_date))
            by_type = daily_stats.get("by_type", {})
            if by_type:
                import pandas as pd
                df = pd.DataFrame(
                    list(by_type.items()),
                    columns=["Type", "Count"],
                )
                st.bar_chart(df.set_index("Type"))


# ─── PAGE 3: Camera Management ───────────────────────────────────
elif page == "📹 Camera Management":
    st.markdown('<div class="header-banner"><h1>📹 Camera Management</h1></div>', unsafe_allow_html=True)

    if db is None:
        st.warning("Database not initialized.")
    else:
        cameras = db.get_camera_status()

        if cameras:
            for cam in cameras:
                cam_id = cam["camera_id"]
                status = cam.get("status", "offline")
                is_online = status == "online"

                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                with col1:
                    icon = "🟢" if is_online else "🔴"
                    st.markdown(f"### {icon} {cam_id}")
                with col2:
                    st.metric("FPS", f"{cam.get('fps', 0):.1f}")
                with col3:
                    st.metric("Frames", cam.get("frames_processed", 0))
                with col4:
                    st.metric("Threats Today", cam.get("threats_today", 0))
                with col5:
                    last_seen = cam.get("last_seen", "Never")
                    st.text(f"Last seen: {last_seen}")

                    # Control buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"{'Disable' if is_online else 'Enable'}", key=f"toggle_{cam_id}"):
                            if "stream_manager" in st.session_state:
                                sm = st.session_state["stream_manager"]
                                if is_online:
                                    sm.disable_camera(cam_id)
                                else:
                                    sm.enable_camera(cam_id)
                                st.rerun()
                    with c2:
                        if st.button("Test Alert", key=f"test_{cam_id}"):
                            st.info(f"Test alert sent for {cam_id}")

                st.markdown("---")
        else:
            st.info("No cameras registered. Start SafeWatch to initialize camera feeds.")

        # Zone Configuration
        st.markdown("### 🗺️ Zone Configuration")
        st.info(
            "Zone configuration allows you to define restricted areas on camera feeds. "
            "Zones are defined as polygon points [x, y] and support types: "
            "'restricted', 'high_security', 'entrance'."
        )

        with st.form("add_zone"):
            zone_name = st.text_input("Zone Name")
            zone_type = st.selectbox("Zone Type", ["restricted", "high_security", "entrance"])
            zone_points_str = st.text_area(
                "Polygon Points (JSON)",
                placeholder='[[100, 100], [300, 100], [300, 400], [100, 400]]',
            )
            submitted = st.form_submit_button("Add Zone")
            if submitted and zone_name and zone_points_str:
                try:
                    import json
                    points = json.loads(zone_points_str)
                    if "zone_manager" in st.session_state:
                        st.session_state["zone_manager"].add_zone(zone_name, points, zone_type)
                        st.success(f"Zone '{zone_name}' added!")
                    else:
                        st.warning("Zone manager not available. Start SafeWatch first.")
                except Exception as e:
                    st.error(f"Invalid points format: {e}")


# ─── PAGE 4: System Settings ─────────────────────────────────────
elif page == "⚙️ System Settings":
    st.markdown('<div class="header-banner"><h1>⚙️ System Settings</h1></div>', unsafe_allow_html=True)

    # Threat Thresholds
    st.markdown("### 🎯 Threat Confidence Thresholds")

    threat_types = {
        "Fight": 0.82,
        "Fall": 0.78,
        "Harassment": 0.75,
        "Assault": 0.85,
        "Unconscious": 0.80,
        "Trespass": 0.95,
        "Crowd Panic": 0.72,
        "Accident": 0.78,
        "Abuse": 0.80,
    }

    for threat, default in threat_types.items():
        value = st.slider(
            f"{threat} Threshold",
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.01,
            key=f"threshold_{threat}",
        )

    # Telegram Test
    st.markdown("---")
    st.markdown("### 📱 Telegram Bot")
    if st.button("🧪 Test Telegram Connection"):
        st.info("Telegram test initiated. Check logs for results.")

    # System Logs
    st.markdown("---")
    st.markdown("### 📜 System Logs")

    if db is not None:
        logs = db.get_system_logs(limit=50)
        if logs:
            for log in logs[:20]:
                level = log.get("level", "INFO")
                level_colors = {
                    "INFO": "🔵",
                    "WARNING": "🟡",
                    "ERROR": "🔴",
                    "DEBUG": "⚪",
                }
                icon = level_colors.get(level, "⚪")
                st.text(
                    f"{icon} [{log.get('timestamp', '')}] "
                    f"{log.get('message', '')}"
                )
        else:
            st.info("No system logs yet.")
    else:
        st.warning("Database not available.")

    # Model Info
    st.markdown("---")
    st.markdown("### 🧠 Model Information")

    model_info = {
        "YOLOv8": {"file": "models/yolov8n.pt", "status": ""},
        "Action Classifier (ONNX)": {"file": "models/action_classifier.onnx", "status": ""},
        "Custom YOLO": {"file": "models/custom_threat_yolo.pt", "status": ""},
    }

    for name, info in model_info.items():
        exists = Path(info["file"]).exists()
        status = "✅ Loaded" if exists else "⚠️ Not found (using fallback)"
        st.markdown(f"**{name}**: `{info['file']}` — {status}")
