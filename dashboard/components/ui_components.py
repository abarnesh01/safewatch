"""
SafeWatch Enterprise V2 — Reusable UI Components
Provides consistent, enterprise-grade visual components for the dashboard.
No emojis used as per requirements.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime


def load_theme():
    """Load the enterprise CSS theme."""
    css_path = Path(__file__).parent.parent / "theme.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_top_navigation(brand_name: str, system_health: str, active_user: str, num_notifications: int = 0):
    """Render the standard top navigation bar."""
    health_class = "success" if system_health.lower() == "healthy" else "warning" if system_health.lower() == "warning" else "danger"
    notif_badge = f'<span class="badge badge-critical" style="padding: 1px 6px; margin-left: 5px;">{num_notifications}</span>' if num_notifications > 0 else ""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    st.markdown(f"""
    <div class="top-nav">
        <div class="top-nav-brand">
            <span>{brand_name}</span> <span class="accent">V2</span>
        </div>
        <div class="top-nav-metrics">
            <span>Time: {current_time}</span>
            <span class="top-nav-status">
                <span class="status-indicator {health_class}"></span>
                System: {system_health}
            </span>
            <span>User: {active_user}</span>
            <span>Alerts: {num_notifications}{notif_badge}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = ""):
    """Render a professional page header with gradient styling without emojis."""
    sub_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(value, label: str, delta: str = "", delta_dir: str = "up"):
    """Render a single KPI card with value, label, and optional delta without emojis."""
    delta_html = ""
    if delta:
        css_dir = "up" if delta_dir == "up" else "down"
        arrow = "Up" if delta_dir == "up" else "Down"
        delta_html = f'<div class="kpi-delta {css_dir}" style="font-size: 0.72rem; color: var(--accent-cyan); font-weight: 500;">{arrow}: {delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_severity_badge(severity: str):
    """Return HTML for a severity badge without emojis."""
    sev_lower = severity.lower()
    badge_class = f"badge-{sev_lower}" if sev_lower in ("critical", "high", "medium", "low") else "badge-info"
    return f'<span class="badge {badge_class}">{severity}</span>'


def render_status_indicator(status: str, label: str):
    """Render an online/offline/warning status dot with label without emojis."""
    status_class = "success" if status == "online" else "warning" if status == "warning" else "danger"
    return f'<span class="status-indicator {status_class}"></span> {label}'


def render_glass_card(content_html: str, extra_class: str = ""):
    """Wrap content in a glassmorphism card."""
    st.markdown(f"""
    <div class="glass-card {extra_class}">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


def render_mini_stat(label: str, value: str, color: str = ""):
    """Render a compact label-value stat row without emojis."""
    style = f'color: {color};' if color else ''
    return f"""
    <div class="mini-stat">
        <span class="label">{label}</span>
        <span class="value" style="{style}">{value}</span>
    </div>
    """


def render_login_page():
    """Render the login header without emojis."""
    st.markdown("""
    <div class="login-header">
        <h2>SAFEWATCH ENTERPRISE</h2>
        <p>Tactical Command Control Interface</p>
    </div>
    """, unsafe_allow_html=True)


def render_incident_row(incident: dict):
    """Render a single incident as a styled row without emojis."""
    ts = incident.get("timestamp", "")[:19]
    threat = incident.get("threat_type", "Unknown")
    severity = incident.get("severity", "LOW")
    cam = incident.get("camera_id", "Unknown")
    conf = incident.get("confidence", 0)
    badge = render_severity_badge(severity)

    return f"""
    <div class="mini-stat" style="padding: 10px 0; border-bottom: 1px solid var(--border-subtle);">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-secondary); width: 140px;">{ts}</span>
        <span style="font-weight: 600; width: 120px; color: var(--text-primary);">{threat}</span>
        <span style="width: 80px;">{badge}</span>
        <span style="font-size: 0.8rem; color: var(--text-secondary); width: 100px;">Stream: {cam}</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--accent-cyan); text-align: right; width: 60px;">{conf:.0%}</span>
    </div>
    """


def plotly_theme():
    """Return a consistent Plotly layout theme dict."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.5)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=35, r=15, t=35, b=35),
        xaxis=dict(
            gridcolor="rgba(100,116,139,0.08)",
            zerolinecolor="rgba(100,116,139,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,0.08)",
            zerolinecolor="rgba(100,116,139,0.08)",
        ),
        colorway=["#2563EB", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#F97316"],
    )
