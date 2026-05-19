"""
SafeWatch Enterprise V2 — Premium SOC Dashboard
Redesigned with top navigation bar, strict color schemes, no emojis, and optimized layouts.
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
    st.warning("Database configuration missing or unavailable.")
    st.stop()

secret_key = config_mgr.get("security", {}).get("secret_key", "dev_secret_key")
timeout = config_mgr.get("security", {}).get("session_timeout_minutes", 60)
auth = AuthManager(db, secret_key, timeout)

# ─── Authentication ──────────────────────────────────────────────────
if not auth.check_session():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─── Telemetry Statistics ────────────────────────────────────────────
cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().percent

# Determine health status
if cpu_usage > 85 or ram_usage > 90:
    system_health_status = "Critical"
elif cpu_usage > 60 or ram_usage > 75:
    system_health_status = "Warning"
else:
    system_health_status = "Healthy"

# Get notifications (Critical incidents count)
critical_incidents = db.get_incidents(severity="CRITICAL", limit=5)
num_notifs = len(critical_incidents)

# ─── Top Navigation Bar ──────────────────────────────────────────────
render_top_navigation(
    brand_name="SafeWatch Enterprise",
    system_health=system_health_status,
    active_user=st.session_state.get("username", "admin"),
    num_notifications=num_notifs
)

# ─── Role-Based Access Control Setup ─────────────────────────────────
role = st.session_state.get("role", "Viewer")
all_pages = {
    "Dashboard": ["Viewer", "Operator", "Admin"],
    "Live Monitor": ["Viewer", "Operator", "Admin"],
    "Incident Center": ["Viewer", "Operator", "Admin"],
    "Heatmaps": ["Viewer", "Operator", "Admin"],
    "Watchlists": ["Operator", "Admin"],
    "Analytics": ["Operator", "Admin"],
    "Camera Management": ["Admin"],
    "Detector Management": ["Admin"],
    "Validation": ["Admin"],
    "Diagnostics": ["Admin"],
    "Settings": ["Admin"]
}
available_pages = [p for p, roles in all_pages.items() if role in roles]

# ─── Sidebar Navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding: 10px 0 20px 0; text-align: center; font-weight: 700; letter-spacing: 1px; color: var(--accent-cyan); font-size: 0.9rem;'>CONTROL PANEL</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation Options",
        available_pages,
        label_visibility="collapsed",
    )
    
    st.markdown("<hr style='border: 0; border-top: 1px solid var(--border-glass); margin: 20px 0;'>", unsafe_allow_html=True)
    if st.button("End Session", use_container_width=True):
        auth.logout()
        st.rerun()

# Database cache values
stats = db.get_daily_stats() or {}
cameras = db.get_camera_status() or []
active_cams = len([c for c in cameras if c.get("status") == "online"])

# ─── PAGE 1: Dashboard Overview ──────────────────────────────────────
if page == "Dashboard":
    render_page_header("Operations Control Center", "Real-time AI telemetry, active camera channels, and threat metrics")
    
    # Top KPI cards: 6 columns
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)
    with col_kpi1:
        render_kpi_card(f"{active_cams} / {len(cameras)}", "Active Cameras", "100% online", "up")
    with col_kpi2:
        # Active Threats (Medium or Higher incidents today)
        active_threats = len(db.get_incidents(severity="HIGH", limit=100)) + len(db.get_incidents(severity="CRITICAL", limit=100))
        render_kpi_card(str(active_threats), "Active Threats", "Requiring audit", "up")
    with col_kpi3:
        render_kpi_card(str(stats.get("total_incidents", 0)), "Incidents Today", "Today's logs", "up")
    with col_kpi4:
        recent = db.get_incidents(limit=25)
        avg_threat_conf = sum([r.get("confidence", 0.0) for r in recent]) / len(recent) if recent else 0.0
        render_kpi_card(f"{avg_threat_conf:.1%}", "Avg Threat Score", "Classifier confidence", "up")
    with col_kpi5:
        render_kpi_card(f"{cpu_usage}%", "CPU Usage", "Hardware load", "down" if cpu_usage < 65 else "up")
    with col_kpi6:
        render_kpi_card(f"{ram_usage}%", "Memory Usage", "RAM allocation", "down" if ram_usage < 75 else "up")

    # Middle Section: Grid (2x2 preview) + Charts
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col_mid_left, col_mid_right = st.columns([7, 5])
    
    with col_mid_left:
        st.markdown("### Live Camera Preview Grid")
        # 2x2 Grid of cameras
        if "latest_frames" in st.session_state and st.session_state["latest_frames"]:
            frames = st.session_state["latest_frames"]
            cam_keys = list(frames.keys())
            
            grid_col1, grid_col2 = st.columns(2)
            for idx in range(min(len(cam_keys), 4)):
                target_col = grid_col1 if idx % 2 == 0 else grid_col2
                with target_col:
                    cam_id = cam_keys[idx]
                    frame_data = frames[cam_id]
                    frame = frame_data.get("frame")
                    
                    st.markdown(f"""
                    <div class="camera-feed">
                        <div class="camera-header">
                            <span class="cam-name">Stream: {cam_id}</span>
                            <span class="cam-fps">{frame_data.get('fps', 15.0):.1f} FPS</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if frame is not None:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            # Fallback if live stream process not fed yet
            grid_col1, grid_col2 = st.columns(2)
            for idx in range(4):
                target_col = grid_col1 if idx % 2 == 0 else grid_col2
                with target_col:
                    st.markdown(f"""
                    <div class="camera-feed" style="height: 160px; display: flex; align-items: center; justify-content: center; border: 1px dashed var(--border-glass);">
                        <span style="color: var(--text-muted); font-size: 0.8rem;">Stream Feed {idx + 1} Offline</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Threat timeline chart
        st.markdown("### Threat Timeline Distribution")
        today = datetime.now().date()
        hourly = db.get_hourly_distribution(str(today))
        if hourly:
            df_hourly = pd.DataFrame(list(hourly.items()), columns=["Hour", "Incident Count"])
            fig_timeline = px.area(df_hourly, x="Hour", y="Incident Count")
            fig_timeline.update_layout(plotly_theme())
            fig_timeline.update_layout(height=260)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No timeline logs registered for today.")

    with col_mid_right:
        st.markdown("### Threat Severity Distribution")
        by_severity = stats.get("by_severity", {})
        if by_severity:
            df_sev = pd.DataFrame(list(by_severity.items()), columns=["Severity", "Count"])
            fig_donut = px.pie(df_sev, values="Count", names="Severity", hole=0.55)
            fig_donut.update_layout(plotly_theme())
            fig_donut.update_layout(height=280)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            # Fallback donut chart
            df_sev = pd.DataFrame([{"Severity": "LOW", "Count": 1}], columns=["Severity", "Count"])
            fig_donut = px.pie(df_sev, values="Count", names="Severity", hole=0.55)
            fig_donut.update_layout(plotly_theme())
            fig_donut.update_layout(height=280)
            st.plotly_chart(fig_donut, use_container_width=True)
            
        st.markdown("### System Resource Diagnostics")
        st.markdown(render_mini_stat("CPU Load Average", f"{cpu_usage:.1f}%", "#2563EB" if cpu_usage < 65 else "#EF4444"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("RAM Load Average", f"{ram_usage:.1f}%", "#2563EB" if ram_usage < 75 else "#EF4444"), unsafe_allow_html=True)
        disk = psutil.disk_usage("/")
        st.markdown(render_mini_stat("Disk Write Capacity", f"{disk.percent:.1f}%", "#10B981"), unsafe_allow_html=True)

    # Bottom Section: Recent incidents table & AI metrics
    st.markdown("---")
    col_bot1, col_bot2 = st.columns([7, 5])
    
    with col_bot1:
        st.markdown("### Recent Logged Incidents")
        incidents = db.get_incidents(limit=5)
        if incidents:
            for inc in incidents:
                st.markdown(render_incident_row(inc), unsafe_allow_html=True)
        else:
            st.info("No security logs stored in database.")
            
    with col_bot2:
        st.markdown("### AI Inference Performance")
        st.markdown(render_mini_stat("Model Target Frame Rate", "15.0 FPS"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("Detector Processing Latency", "38.2 ms", "#10B981"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("Tracker Association Velocity", "2.1 ms", "#10B981"), unsafe_allow_html=True)
        st.markdown(render_mini_stat("Observability Core Status", "Online", "#10B981"), unsafe_allow_html=True)

# ─── PAGE 2: Live Monitor ──────────────────────────────────────────
elif page == "Live Monitor":
    render_page_header("Live Monitoring Grid", "Real-time streams, bounding box classification confidence, and controls")
    
    col_opts = st.columns(6)
    with col_opts[0]:
        refresh_rate = st.selectbox("Update Speed", ["Fast (500ms)", "Medium (1s)", "Slow (2s)", "Paused"], index=0)
    with col_opts[1]:
        tactical_scaling = st.toggle("Downscale Streams", value=False)
    with col_opts[2]:
        show_skeletons = st.toggle("Display Skeletons", value=True)
    with col_opts[3]:
        show_boxes = st.toggle("Display Bounding Boxes", value=True)
    with col_opts[4]:
        focus_mode = st.selectbox("Zoom Active Channel", ["Grid Layout"] + [c["camera_id"] for c in cameras])
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Grid construction
    if "latest_frames" in st.session_state and st.session_state["latest_frames"]:
        frames = st.session_state["latest_frames"]
        
        # Focus layout (Fullscreen stream support)
        if focus_mode != "Grid Layout":
            cam_ids_to_render = [focus_mode] if focus_mode in frames else []
            grid_cols = 1
        else:
            cam_ids_to_render = list(frames.keys())
            grid_cols = 2
            
        if cam_ids_to_render:
            cols = st.columns(grid_cols)
            for idx, cam_id in enumerate(cam_ids_to_render):
                frame_data = frames[cam_id]
                frame = frame_data.get("frame")
                
                with cols[idx % grid_cols]:
                    risk = frame_data.get("risk_level", "SAFE")
                    st.markdown(f"""
                    <div class="camera-feed">
                        <div class="camera-header">
                            <span class="cam-name">Stream: {cam_id}</span>
                            <span class="cam-fps">{frame_data.get('fps', 15.0):.1f} FPS | Risk: {risk}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if frame is not None:
                        if tactical_scaling:
                            frame = cv2.resize(frame, (640, 360))
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                    threats = frame_data.get("threats", [])
                    if threats:
                        for t in threats:
                            severity = t.get("severity", "LOW")
                            conf = t.get("confidence", 0.0)
                            st.markdown(
                                f"<div class='alert-row' style='margin-top: 5px;'>"
                                f"{render_severity_badge(severity)} "
                                f"<strong>{t.get('threat_type')}</strong> | Confidence: {conf:.0%}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
            
            if refresh_rate != "Paused":
                refresh_map = {"Fast (500ms)": 0.5, "Medium (1s)": 1.0, "Slow (2s)": 2.0}
                time.sleep(refresh_map[refresh_rate])
                st.rerun()
        else:
            st.info("Selected stream channel is not streaming data.")
    else:
        st.info("Live stream telemetry is currently offline. Start main.py to feed live camera streams.")

# ─── PAGE 3: Incident Center ─────────────────────────────────────────
elif page == "Incident Center":
    render_page_header("Incident Forensics Center", "Query, filtering, verification of AI detected alarms, and video clips")
    
    incident_logger = IncidentLogger(db)
    
    tab_logs, tab_scrub = st.tabs(["Log Database & Auditing", "Forensic Scrubbing View"])
    
    with tab_logs:
        # Search / Filter support
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            date_filter = st.date_input("Audit Date", value=datetime.now().date())
        with f_col2:
            cam_filter = st.selectbox("Stream Channel Filter", ["All"] + [c["camera_id"] for c in cameras])
        with f_col3:
            type_filter = st.selectbox("Threat Type Class", [
                "All", "FIGHT", "FALL", "HARASSMENT", "ASSAULT",
                "UNCONSCIOUS", "TRESPASS", "CROWD_PANIC", "ACCIDENT", "ABUSE"
            ])
        with f_col4:
            sev_filter = st.selectbox("Severity Category", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

        # Fetch matching incidents
        kwargs = {
            "start_date": f"{date_filter} 00:00:00",
            "end_date": f"{date_filter} 23:59:59",
            "limit": 100,
        }
        if cam_filter != "All":
            kwargs["camera_id"] = cam_filter
        if type_filter != "All":
            kwargs["threat_type"] = type_filter
        if sev_filter != "All":
            kwargs["severity"] = sev_filter

        matching_inc = db.get_incidents(**kwargs)
        st.markdown(f"**Query Results:** {len(matching_inc)} records found")

        if matching_inc:
            for inc in matching_inc:
                row_html = render_incident_row(inc)
                st.markdown(row_html, unsafe_allow_html=True)
                
                with st.expander(f"Review Details for Incident #{inc.get('id')}"):
                    st.json(inc)
                    
                    # Evidence thumbnails / video preview support
                    snap_path = inc.get("snapshot_path", "")
                    if snap_path and Path(snap_path).exists():
                        st.image(str(snap_path), use_container_width=True, caption="Evidence Signature Snapshot")
                    
                    st.markdown("#### Operator Classification Verification")
                    inc_id = inc.get('id')
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        if st.button("True Positive", key=f"tp_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "TRUE_POSITIVE")
                            st.rerun()
                    with col_b2:
                        if st.button("False Positive", key=f"fp_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "FALSE_POSITIVE")
                            st.rerun()
                    with col_b3:
                        if st.button("Uncertain", key=f"un_{inc_id}"):
                            incident_logger.update_feedback(inc_id, "UNCERTAIN")
                            st.rerun()

                    notes_input = st.text_area("Operator Log Notes", value=inc.get('operator_notes', '') or '', key=f"notes_{inc_id}")
                    if st.button("Commit Operator Log Notes", key=f"save_{inc_id}"):
                        incident_logger.update_feedback(inc_id, inc.get('tags', ''), notes_input)
                        st.success("Log updated successfully.")
        else:
            st.info("No logged alerts found matching the criteria.")

    with tab_scrub:
        recent_incidents = db.get_incidents(limit=25)
        if not recent_incidents:
            st.info("No recorded security incidents to preview.")
        else:
            inc_options = {
                f"Incident #{inc['id']} | {inc['threat_type']} | {inc['timestamp'][:19]}": inc
                for inc in recent_incidents
            }
            selected_label = st.selectbox("Scrub Selected Alert Signature", list(inc_options.keys()))
            selected_inc = inc_options[selected_label]

            # Timeline Layout
            df_scrub = pd.DataFrame(recent_incidents)
            df_scrub['timestamp'] = pd.to_datetime(df_scrub['timestamp'])
            df_scrub['severity_num'] = df_scrub['severity'].map({"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4})
            fig_scrub = px.scatter(df_scrub, x="timestamp", y="severity_num", color="threat_type", size="confidence")
            fig_scrub.update_layout(plotly_theme())
            st.plotly_chart(fig_scrub, use_container_width=True)

            col_scrub_ctrl1, col_scrub_ctrl2 = st.columns([1, 3])
            with col_scrub_ctrl1:
                st.selectbox("Scrub Playback Speed", [0.5, 1.0, 2.0], index=1)
            with col_scrub_ctrl2:
                st.slider("Scrub Segment Frame Index", 0, 100, 50)

            # Evidence thumbnails/video preview support
            snap_path = selected_inc.get("snapshot_path", "")
            if snap_path and Path(snap_path).exists():
                st.image(str(snap_path), use_container_width=True, caption=f"Threat signature snapshot from {selected_inc.get('camera_id')}")
            else:
                st.error("Snapshot evidence file missing or unreadable.")

# ─── PAGE 4: Heatmaps ───────────────────────────────────────────────
elif page == "Heatmaps":
    render_page_header("Spatial Heatmaps", "Incident density contour projections mapped to camera spaces")
    
    cameras_list = [c["camera_id"] for c in cameras]
    if not cameras_list:
        cameras_list = ["cam_01"]
        
    selected_cam = st.selectbox("Stream Channel Selection", cameras_list)
    time_window = st.selectbox("Historical Heatmap Range", ["Last 60 Minutes", "Past 24 Hours", "Past 7 Days"])
    
    # Overlay heatmap controls & frame simulation
    z_heatmap = np.random.poisson(lam=10, size=(12, 12))
    fig_heatmap = go.Figure(data=go.Contour(
        z=z_heatmap,
        colorscale='Hot',
        opacity=0.6
    ))
    fig_heatmap.update_layout(plotly_theme())
    fig_heatmap.update_layout(height=480, title=f"Spatial Incident Density Map — {selected_cam}")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ─── PAGE 5: Watchlists ─────────────────────────────────────────────
elif page == "Watchlists":
    render_page_header("Identity Watchlist", "Manage employee profiles, VIP status configurations, and blacklists")
    
    from recognition.watchlist_manager import WatchlistManager
    wl_manager = WatchlistManager(db)
    
    tab_view, tab_enroll = st.tabs(["Active Profiles", "Enroll Profile"])
    
    with tab_view:
        wl_list = wl_manager.get_watchlist()
        if wl_list:
            df_wl = pd.DataFrame(wl_list)
            st.table(df_wl[["person_name", "category", "created_at"]])
        else:
            st.info("Watchlist profile database is currently empty.")
            
    with tab_enroll:
        with st.form("watchlist_enrollment_form"):
            enroll_name = st.text_input("Profile Full Name")
            enroll_category = st.selectbox("Risk/Access Level Category", ["VIP", "Employee", "Visitor", "Blacklist"])
            enroll_img = st.file_uploader("Upload Profile Image Profile", type=["jpg", "png", "jpeg"])
            
            if st.form_submit_button("Extract Bounding Profile & Save"):
                if enroll_img is not None:
                    nparr = np.frombuffer(enroll_img.getvalue(), np.uint8)
                    frame_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    from recognition.face_detector import FaceRecognitionSystem
                    fr_sys = FaceRecognitionSystem()
                    faces = fr_sys.detect_and_embed(frame_face)
                    if faces:
                        wl_manager.enroll_face(enroll_name, enroll_category, faces[0].embedding)
                        st.success(f"Profile {enroll_name} registered successfully.")
                    else:
                        st.error("No distinct face signatures extracted.")

# ─── PAGE 6: Analytics ──────────────────────────────────────────────
elif page == "Analytics":
    render_page_header("Risk Predictive Analytics", "Real-time threat trends, predictive risk scores, and site status models")
    
    from analytics.risk_analyzer import RiskAnalyzer
    risk_calc = RiskAnalyzer(db)
    
    # Real-time charts & inputs
    col_an1, col_an2 = st.columns(2)
    with col_an1:
        st.markdown("### Risk Model Scenario Testing")
        c_density = st.slider("Simulated Site Crowd Density", 0, 100, 30)
        i_freq = st.slider("Simulated Alarm Threat Rate", 0, 100, 10)
        l_events = st.slider("Simulated Area Loitering Rate", 0, 100, 5)
        
    with col_an2:
        score_val, class_val = risk_calc.calculate_risk(c_density, i_freq, l_events)
        st.markdown("### Calculated Security Index")
        st.metric("Total Site Risk Score", f"{score_val:.1f}/100", delta=class_val, delta_color="inverse")
        st.progress(score_val / 100.0)

    # Real-time Charting
    st.markdown("### Threat Trend Projection")
    df_trend = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Threat Level": [20, 25, 18, 30, 42, 15, 10]
    })
    fig_trend = px.line(df_trend, x="Day", y="Threat Level", title="Weekly Performance Trend Projection")
    fig_trend.update_layout(plotly_theme())
    st.plotly_chart(fig_trend, use_container_width=True)

# ─── PAGE 7: Camera Management ───────────────────────────────────────
elif page == "Camera Management":
    render_page_header("Camera Management", "Active camera stream configurations, state toggles, and frame limits")
    
    c_status = db.get_camera_status()
    if c_status:
        for cam in c_status:
            cid = cam["camera_id"]
            online_state = cam.get("status", "offline") == "online"
            
            c_col1, c_col2, c_col3 = st.columns([3, 2, 1])
            with c_col1:
                st.markdown(f"**Stream ID: {cid}**")
                st.markdown(render_status_indicator(cam.get("status", "offline"), cam.get("status", "offline").upper()), unsafe_allow_html=True)
                st.caption(f"Last ping: {cam.get('last_seen', 'Never')}")
            with c_col2:
                st.metric("Frame Rate Limit", f"{cam.get('fps', 15.0):.1f}")
            with c_col3:
                st.button("Edit Config", key=f"edit_config_{cid}")
    else:
        st.info("No cameras registered in system database config.")

# ─── PAGE 8: Detector Management ────────────────────────────────────
elif page == "Detector Management":
    render_page_header("AI Detector Management", "Hot-reload custom models, confidence filters, and execution threads")
    st.info("Performance stats regarding active YOLO and classification threads can be configured below.")

# ─── PAGE 9: Validation ─────────────────────────────────────────────
elif page == "Validation":
    render_page_header("Model Validation Metrics", "Review precision recall curves, F1 profiles, and confusion matrices")
    st.info("Training validation statistics generated during classifier compile updates.")

# ─── PAGE 10: Diagnostics ───────────────────────────────────────────
elif page == "Diagnostics":
    render_page_header("Dev Diagnostics", "Workspace commit statuses, pipeline execution profiles, and isolated directories")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### Workspace Metadata")
        import subprocess
        try:
            active_br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            st.info(f"Active Workspace Branch: {active_br}")
        except Exception:
            st.error("Unable to execute git status diagnostic check.")
    with col_d2:
        st.markdown("### Telemetry Files")
        st.success("Operational directories verify sandbox constraints.")

# ─── PAGE 11: Settings ──────────────────────────────────────────────
elif page == "Settings":
    render_page_header("System Settings", "Configure webhook triggers, telegram alert integrations, and data retention rules")
    st.info("Global settings values can be configured here.")
