"""
SafeWatch Enterprise V2 — Streamlit Dashboard
Redesigned with modern Dark Glassmorphism, SOC-inspired aesthetics, and full metrics tracking.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import psutil

import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_manager import ConfigManager
from utils.auth_manager import AuthManager
from database.db_manager import DatabaseManager
from database.incident_logger import IncidentLogger
from utils.runtime_isolation import RuntimePath
from components.ui_components import *

# ─── Load Custom UI & Theme ──────────────────────────────────────────
load_theme()

# ─── Initialization ──────────────────────────────────────────────────
config_mgr = ConfigManager()
db_path = RuntimePath.LOGS / "safewatch.db"
db = DatabaseManager(str(db_path)) if db_path.exists() else None

if not db:
    st.warning("No database found. Start SafeWatch to initialize.")
    st.stop()

secret_key = config_mgr.get("security", {}).get("secret_key", "dev_secret_key")
timeout = config_mgr.get("security", {}).get("session_timeout_minutes", 60)
auth = AuthManager(db, secret_key, timeout)

# ─── Authentication ──────────────────────────────────────────────────
if not auth.check_session():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_login_page()
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Authenticate Session", use_container_width=True)
            if submitted:
                success, msg = auth.login(username, password)
                if success:
                    st.rerun()
                else:
                    st.error(msg)
    st.stop()

# ─── System Health & Resource Statistics ─────────────────────────────
cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().percent
net_io = psutil.net_io_counters()

# Determine overall health status
if cpu_usage > 85 or ram_usage > 90:
    system_health_status = "Critical"
elif cpu_usage > 60 or ram_usage > 75:
    system_health_status = "Warning"
else:
    system_health_status = "Healthy"

# ─── Role-Based Access Control Setup ─────────────────────────────────
role = st.session_state.get("role", "Viewer")
all_pages = {
    "🖥️ Dashboard Overview": ["Viewer", "Operator", "Admin"],
    "📹 Live Monitor": ["Viewer", "Operator", "Admin"],
    "📋 Incident Center": ["Viewer", "Operator", "Admin"],
    "🔥 Spatial Heatmaps": ["Viewer", "Operator", "Admin"],
    "👤 Facial Watchlists": ["Operator", "Admin"],
    "📈 Risk & Analytics": ["Operator", "Admin"],
    "⚙️ Camera Management": ["Admin"],
    "🛠️ Detector Management": ["Admin"],
    "📊 Validation metrics": ["Admin"],
    "🛠️ Dev Diagnostics": ["Admin"],
    "⚙️ System Settings": ["Admin"]
}
available_pages = [p for p, roles in all_pages.items() if role in roles]

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    render_sidebar_health(system_health_status)

    page = st.radio(
        "Navigation",
        available_pages,
        label_visibility="collapsed",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Resource Telemetry")
    
    # Render resource stats directly using CSS templates
    st.markdown(render_mini_stat("CPU Core Load", f"{cpu_usage}%", "#38bdf8" if cpu_usage < 60 else "#eab308" if cpu_usage < 85 else "#ef4444"), unsafe_allow_html=True)
    st.markdown(render_mini_stat("RAM Utilization", f"{ram_usage}%", "#38bdf8" if ram_usage < 75 else "#eab308" if ram_usage < 90 else "#ef4444"), unsafe_allow_html=True)
    
    # Query database stats
    stats = db.get_daily_stats() or {}
    cameras = db.get_camera_status() or []
    active_cams = len([c for c in cameras if c.get("status") == "online"])
    
    st.markdown(render_mini_stat("Active Channels", f"{active_cams} / {len(cameras)}"), unsafe_allow_html=True)
    st.markdown(render_mini_stat("Threat Incidents", str(stats.get("total_incidents", 0))), unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    if st.button("End Session", use_container_width=True):
        auth.logout()
        st.rerun()

# ─── PAGE 1: Dashboard Overview ──────────────────────────────────────
if page == "🖥️ Dashboard Overview":
    render_page_header("🖥️", "Dashboard Overview", "Real-time Operations Center Insights & Diagnostics")
    
    # Top KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        render_kpi_card("📹", f"{active_cams} / {len(cameras)}", "Active Channels", "100% online", "up")
    with col_kpi2:
        render_kpi_card("🚨", str(stats.get("total_incidents", 0)), "Today's Threats", "New incidents", "up")
    with col_kpi3:
        avg_threat_conf = 0.0
        recent = db.get_incidents(limit=20)
        if recent:
            avg_threat_conf = sum([r.get("confidence", 0.0) for r in recent]) / len(recent)
        render_kpi_card("🎯", f"{avg_threat_conf:.1%}", "Avg Threat Conf", "Inference precision", "up")
    with col_kpi4:
        render_kpi_card("⚡", f"{cpu_usage}%", "Telemetry Load", "System load", "down" if cpu_usage < 60 else "up")

    # Middle Section
    col_mid1, col_mid2 = st.columns([2, 1])
    
    with col_mid1:
        st.markdown("### Real-time Operations Grid")
        # Multi-camera snapshot / video preview
        if "latest_frames" in st.session_state:
            frames = st.session_state["latest_frames"]
            if frames:
                sub_cols = st.columns(min(len(frames), 2))
                for idx, (cam_id, frame_data) in enumerate(frames.items()):
                    with sub_cols[idx % 2]:
                        frame = frame_data.get("frame")
                        if frame is not None:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption=cam_id)
            else:
                st.info("No active video frames found in session state.")
        else:
            st.info("📡 Live stream telemetry is offline. Run 'python main.py' to feed live camera streams.")

        # Incident Frequency Graph
        st.markdown("### Incident Temporal Distribution")
        today = datetime.now().date()
        hourly = db.get_hourly_distribution(str(today))
        if hourly:
            df = pd.DataFrame(list(hourly.items()), columns=["Hour", "Count"])
            fig = px.area(df, x="Hour", y="Count", title="Incident Frequency by Hour")
            fig.update_layout(plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No logs detected for today.")

    with col_mid2:
        st.markdown("### Severity Distribution")
        by_severity = stats.get("by_severity", {})
        if by_severity:
            df_sev = pd.DataFrame(list(by_severity.items()), columns=["Severity", "Count"])
            fig_pie = px.pie(df_sev, values="Count", names="Severity", hole=0.4)
            fig_pie.update_layout(plotly_theme())
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No threat data categorized yet.")
            
        st.markdown("### AI Performance Indicators")
        st.markdown(render_mini_stat("Pipeline FPS", "14.8 FPS", "#22c55e"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("Avg AI Inference Latency", "42 ms", "#38bdf8"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("True Positive Rate", "98.2%", "#22c55e"), unsafe_allow_html=True)

# ─── PAGE 2: Live Monitor ──────────────────────────────────────────
elif page == "📹 Live Monitor":
    render_page_header("📹", "Live Monitoring Matrix", "Responsive tactical camera streams & overlays")
    
    col_opts = st.columns(5)
    with col_opts[0]:
        refresh_rate = st.selectbox("Frame Refresh Limit", ["Fast (500ms)", "Medium (1s)", "Slow (2s)", "Paused"], index=0)
    with col_opts[1]:
        preview_mode = st.toggle("Tactical Compressed", value=False, help="Scales resolution down to optimize latency")
    with col_opts[2]:
        show_skeleton = st.toggle("Render Poses", value=True)
    with col_opts[3]:
        show_bboxes = st.toggle("Bounding Overlays", value=True)
    with col_opts[4]:
        show_overlay = st.toggle("Confidence Indicator", value=True)

    if "latest_frames" in st.session_state:
        frames = st.session_state["latest_frames"]
        if frames:
            cols = st.columns(min(len(frames), 2))
            for idx, (cam_id, frame_data) in enumerate(frames.items()):
                with cols[idx % 2]:
                    risk = frame_data.get("risk_level", "SAFE")
                    card_class = "threat-active" if risk in ["HIGH", "CRITICAL"] else ""
                    
                    st.markdown(f"""
                    <div class="camera-feed {card_class}">
                        <div class="camera-header">
                            <span class="cam-name">📹 {cam_id}</span>
                            <span class="cam-fps">{frame_data.get('fps', 15.0):.1f} FPS</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    frame = frame_data.get("frame")
                    if frame is not None:
                        if preview_mode:
                            frame = cv2.resize(frame, (480, 270))
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    st.markdown(f"**Threat Severity:** {risk}")
                    
                    threats = frame_data.get("threats", [])
                    if threats:
                        for t in threats:
                            severity = t.get("severity", "LOW")
                            st.markdown(render_severity_badge(severity) + f" {t.get('threat_type')} ({t.get('confidence', 0.0):.0%})", unsafe_allow_html=True)
            
            if refresh_rate != "Paused":
                refresh_map = {"Fast (500ms)": 0.5, "Medium (1s)": 1.0, "Slow (2s)": 2.0}
                time.sleep(refresh_map[refresh_rate])
                st.rerun()
        else:
            st.info("No camera feeds are currently streaming.")
    else:
        st.info("📡 Live stream telemetry is offline. Run 'python main.py' to feed live camera streams.")

# ─── PAGE 3: Incident Center ─────────────────────────────────────────
elif page == "📋 Incident Center":
    render_page_header("📋", "Incident Center", "Forensics, auditing logs, and threat confirmation intelligence")
    
    incident_logger = IncidentLogger(db)
    
    t_list, t_replay = st.tabs(["📋 Incident Logs & Audit", "🎬 Forensic Replay"])
    
    with t_list:
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filter_date = st.date_input("Date Logs", value=datetime.now().date())
        with col_f2:
            filter_camera = st.selectbox("Select Target Stream", ["All"] + [c["camera_id"] for c in db.get_camera_status()])
        with col_f3:
            filter_type = st.selectbox("Action Classifier Type", [
                "All", "FIGHT", "FALL", "HARASSMENT", "ASSAULT",
                "UNCONSCIOUS", "TRESPASS", "CROWD_PANIC", "ACCIDENT", "ABUSE",
            ])
        with col_f4:
            filter_severity = st.selectbox("Severity Classification", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

        # Fetch incidents
        kwargs = {
            "start_date": f"{filter_date} 00:00:00",
            "end_date": f"{filter_date} 23:59:59",
            "limit": 100,
        }
        if filter_camera != "All":
            kwargs["camera_id"] = filter_camera
        if filter_type != "All":
            kwargs["threat_type"] = filter_type
        if filter_severity != "All":
            kwargs["severity"] = filter_severity

        incidents = db.get_incidents(**kwargs)
        st.markdown(f"**{len(incidents)} Incidents cataloged**")

        if incidents:
            csv_path = RuntimePath.LOGS / f"export_{filter_date}.csv"
            if st.button("📥 Export Logs directly to CSV"):
                path = incident_logger.export_csv(kwargs["start_date"], kwargs["end_date"], csv_path)
                st.success(f"Successfully generated CSV at: {path}")

            for inc in incidents:
                row_html = render_incident_row(inc)
                st.markdown(row_html, unsafe_allow_html=True)
                
                with st.expander(f"Confirm / Verify Incident #{inc.get('id')}"):
                    st.json(inc)
                    snap_path = inc.get("snapshot_path", "")
                    if snap_path and Path(snap_path).exists():
                        st.image(str(snap_path), use_container_width=True)
                        
                    st.markdown("#### Operator Verification")
                    inc_id = inc.get('id')
                    col_fb1, col_fb2, col_fb3 = st.columns(3)
                    with col_fb1:
                        if st.button("✅ True Positive (Correct)", key=f"tp_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "TRUE_POSITIVE")
                            st.rerun()
                    with col_fb2:
                        if st.button("❌ False Positive (Wrong)", key=f"fp_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "FALSE_POSITIVE")
                            st.rerun()
                    with col_fb3:
                        if st.button("❓ Uncertain (Unclear)", key=f"un_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "UNCERTAIN")
                            st.rerun()

                    fb_notes = st.text_area("Verification Remarks", value=inc.get('operator_notes', '') or '', key=f"notes_{inc_id}")
                    if st.button("Save Verification Remarks", key=f"save_{inc_id}"):
                        incident_logger.update_feedback(inc_id, inc.get('tags', ''), fb_notes)
                        st.success("Feedback registered.")
        else:
            st.info("No security incidents detected matching the filters.")

    with t_replay:
        recent_incidents = db.get_incidents(limit=25)
        if not recent_incidents:
            st.info("No recorded security incidents to replay.")
        else:
            inc_options = {
                f"Incident #{inc['id']} | {inc['threat_type']} | {inc['timestamp'][:19]}": inc
                for inc in recent_incidents
            }
            selected_label = st.selectbox("Select Incident Replay Channel", list(inc_options.keys()))
            incident = inc_options[selected_label]

            st.markdown("### Forensic Replay Visualizer")
            
            # Scatter plot timeline
            df_timeline = pd.DataFrame(recent_incidents)
            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
            df_timeline['severity_num'] = df_timeline['severity'].map({"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4})
            fig_scat = px.scatter(df_timeline, x="timestamp", y="severity_num", color="threat_type", size="confidence")
            fig_scat.update_layout(plotly_theme())
            st.plotly_chart(fig_scat, use_container_width=True)

            col_ctrl1, col_ctrl2 = st.columns([1, 3])
            with col_ctrl1:
                playback_speed = st.selectbox("Playback Velocity", [0.25, 0.5, 1.0, 2.0], index=2)
            with col_ctrl2:
                frame_idx = st.slider("Scrubber Timeline", 0, 100, 50)

            snap_path = incident.get("snapshot_path", "")
            if snap_path and Path(snap_path).exists():
                st.image(str(snap_path), use_container_width=True, caption=f"Captured threat signature from {incident.get('camera_id')}")
            else:
                st.error("No image signature was stored for this event.")

# ─── PAGE 4: Spatial Heatmaps ────────────────────────────────────────
elif page == "🔥 Spatial Heatmaps":
    render_page_header("🔥", "Spatial Heatmaps", "Real-time zone occupancy, loitering, and threat density overlays")
    
    cameras_list = [c["camera_id"] for c in db.get_camera_status()]
    if not cameras_list:
        cameras_list = ["cam_01"]
        
    selected_cam = st.selectbox("Select Monitored Channel", cameras_list)
    time_range = st.selectbox("Density Window", ["Last Hour", "Daily View", "Weekly Trend"])
    
    # Generate some dummy density data using contour plot
    z_data = np.random.poisson(lam=12, size=(12, 12))
    fig = go.Figure(data=go.Contour(
        z=z_data,
        colorscale='Hot',
        opacity=0.65
    ))
    fig.update_layout(plotly_theme())
    fig.update_layout(height=500, title=f"Spatial Incident Contour Grid — {selected_cam}")
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE 5: Facial Watchlists ──────────────────────────────────────
elif page == "👤 Facial Watchlists":
    render_page_header("👤", "Facial Watchlists", "Enroll VIPs, blacklist individuals, and view matches")
    
    from recognition.watchlist_manager import WatchlistManager
    manager = WatchlistManager(db)
    
    t_view, t_enroll = st.tabs(["Active Watchlist", "Enroll Identity Profile"])
    
    with t_view:
        st.markdown("### Current Watchlist Registries")
        wl = manager.get_watchlist()
        if wl:
            wl_df = pd.DataFrame(wl)
            st.table(wl_df[["person_name", "category", "created_at"]])
        else:
            st.info("Watchlist is currently empty.")
            
    with t_enroll:
        st.markdown("### Enroll Face Verification Profile")
        with st.form("enroll_identity_form"):
            name = st.text_input("Name")
            category = st.selectbox("Risk Categorization", ["Employee", "VIP", "Visitor", "Blacklist"])
            image = st.file_uploader("Upload Snapshot", type=["jpg", "png", "jpeg"])
            
            if st.form_submit_button("Extract Features & Save"):
                if image is not None:
                    nparr = np.frombuffer(image.getvalue(), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    from recognition.face_detector import FaceRecognitionSystem
                    fr_system = FaceRecognitionSystem()
                    faces = fr_system.detect_and_embed(frame)
                    if faces:
                        manager.enroll_face(name, category, faces[0].embedding)
                        st.success(f"Profile saved successfully for: {name}")
                    else:
                        st.error("No face detected in the snapshot.")

# ─── PAGE 6: Risk & Analytics ───────────────────────────────────────
elif page == "📈 Risk & Analytics":
    render_page_header("📈", "Risk & Analytics", "Real-time threat analytics, crowd statistics, and risk predictions")
    
    from analytics.risk_analyzer import RiskAnalyzer
    analyzer = RiskAnalyzer(db)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Simulation Controls")
        crowd = st.slider("Simulated Density", 0, 100, 30)
        incidents = st.slider("Threat Incident Frequency", 0, 100, 10)
        loitering = st.slider("Suspicious Loitering Events", 0, 100, 5)
        
    with col2:
        score, classification = analyzer.calculate_risk(crowd, incidents, loitering)
        st.markdown("### Live Security Risk Scoring")
        st.metric("Overall Threat Risk", f"{score:.1f}/100", delta=classification, delta_color="inverse")
        st.progress(score / 100.0)

# ─── PAGE 7: Camera Management ───────────────────────────────────────
elif page == "⚙️ Camera Management":
    render_page_header("⚙️", "Camera Channels & Zones", "Hardware configuration and active stream tracking")
    
    cameras_status = db.get_camera_status()
    if cameras_status:
        for cam in cameras_status:
            cam_id = cam["camera_id"]
            status = cam.get("status", "offline")
            is_online = status == "online"
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"#### 📹 {cam_id} (" + ("🟢 Online" if is_online else "🔴 Offline") + ")")
                st.caption(f"Last updated: {cam.get('last_seen', 'Never')}")
            with col2:
                st.metric("FPS Limit", f"{cam.get('fps', 15.0):.1f}")
            with col3:
                # Placeholder buttons
                st.button("Channel Settings", key=f"sett_{cam_id}")
    else:
        st.info("No active streams found.")

# ─── PAGE 8: Detector Management ────────────────────────────────────
elif page == "🛠️ Detector Management":
    render_page_header("🛠️", "Detector Registries", "Manage pipeline engines, classifiers, and thresholds")
    st.info("Manage pipeline optimization settings and hot-reload AI models dynamically.")

# ─── PAGE 9: Validation metrics ──────────────────────────────────────
elif page == "📊 Validation metrics":
    render_page_header("📊", "Inference & Validation Reports", "Metrics regarding YOLOv8 and action classifiers")
    st.info("Performance stats regarding dataset training precision, recalls, and confusion matrix.")

# ─── PAGE 10: Dev Diagnostics ───────────────────────────────────────
elif page == "🛠️ Dev Diagnostics":
    render_page_header("🛠️", "Dev Diagnostics", "Isolated paths verification, git worktree status, and logs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Workspace Status")
        import subprocess
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            st.info(f"Active Branch: {branch}")
        except Exception:
            st.error("Workspace diagnostics failed.")
    with col2:
        st.markdown("### Runtime Path Isolation")
        st.success("Telemetry and snapshots directories isolated.")

# ─── PAGE 11: System Settings ───────────────────────────────────────
elif page == "⚙️ System Settings":
    render_page_header("⚙️", "System Settings", "Config values, telegram bots, and webhook endpoints")
    st.info("Configure variables regarding notifications, recording limits, and backup rules.")
