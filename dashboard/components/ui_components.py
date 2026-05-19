"""
SafeWatch Enterprise V2 — Reusable UI Components
Provides consistent, enterprise-grade visual components for the dashboard.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime


def load_theme():
    """Load the enterprise CSS theme."""
    css_path = Path(__file__).parent.parent / "theme.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_page_header(icon: str, title: str, subtitle: str = ""):
    """Render a professional page header with gradient title."""
    sub_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header animate-in">
        <div>
            <h1>{icon} {title}</h1>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(icon: str, value, label: str, delta: str = "", delta_dir: str = "up"):
    """Render a single KPI card with icon, value, label, and optional delta."""
    delta_html = ""
    if delta:
        css_dir = "up" if delta_dir == "up" else "down"
        arrow = "↑" if delta_dir == "up" else "↓"
        delta_html = f'<div class="kpi-delta {css_dir}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="kpi-card animate-in">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_severity_badge(severity: str):
    """Return HTML for a severity badge."""
    sev_lower = severity.lower()
    badge_class = f"badge-{sev_lower}" if sev_lower in ("critical", "high", "medium", "low") else "badge-info"
    return f'<span class="badge {badge_class}">{severity}</span>'


def render_status_indicator(status: str, label: str):
    """Render an online/offline/warning status dot with label."""
    status_class = "online" if status == "online" else "warning" if status == "warning" else "offline"
    return f'<span class="status-dot {status_class}"></span>{label}'


def render_glass_card(content_html: str, extra_class: str = ""):
    """Wrap content in a glassmorphism card."""
    st.markdown(f"""
    <div class="glass-card {extra_class} animate-in">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


def render_mini_stat(label: str, value: str, color: str = ""):
    """Render a compact label-value stat row."""
    style = f'color: {color};' if color else ''
    return f"""
    <div class="mini-stat">
        <span class="label">{label}</span>
        <span class="value" style="{style}">{value}</span>
    </div>
    """


def render_sidebar_brand():
    """Render the branded sidebar header."""
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🛡️</div>
        <div class="title">SAFEWATCH</div>
        <div class="version">Enterprise V2</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_health(status: str = "Healthy"):
    """Render the system health indicator in sidebar."""
    if status == "Healthy":
        css_class = ""
        icon = "🟢"
    elif status == "Warning":
        css_class = "warning"
        icon = "🟡"
    else:
        css_class = "critical"
        icon = "🔴"

    st.markdown(f"""
    <div class="sidebar-health {css_class}">
        <span>{icon}</span>
        <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-primary);">
            System: {status}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_login_page():
    """Render the enterprise login page."""
    st.markdown("""
    <div class="login-container animate-in">
        <div class="login-logo">
            <div class="icon">🛡️</div>
            <h1>SAFEWATCH</h1>
            <p>Enterprise Security Operations Center</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_incident_row(incident: dict):
    """Render a single incident as a styled row."""
    ts = incident.get("timestamp", "")[:19]
    threat = incident.get("threat_type", "Unknown")
    severity = incident.get("severity", "LOW")
    cam = incident.get("camera_id", "—")
    conf = incident.get("confidence", 0)
    badge = render_severity_badge(severity)

    return f"""
    <div class="incident-row">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-secondary); min-width: 140px;">{ts}</span>
        <span style="font-weight: 600; min-width: 120px; color: var(--text-primary);">{threat}</span>
        {badge}
        <span style="font-size: 0.8rem; color: var(--text-secondary); min-width: 70px;">📹 {cam}</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--accent-cyan);">{conf:.0%}</span>
    </div>
    """


def plotly_theme():
    """Return a consistent Plotly layout theme dict."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,40,0.5)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            gridcolor="rgba(100,116,139,0.1)",
            zerolinecolor="rgba(100,116,139,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,0.1)",
            zerolinecolor="rgba(100,116,139,0.1)",
        ),
        colorway=["#38bdf8", "#6366f1", "#22c55e", "#eab308", "#ef4444", "#a855f7", "#ec4899", "#f97316"],
    )
