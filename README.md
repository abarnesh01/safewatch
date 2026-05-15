# SafeWatch — AI-Powered CCTV Threat Detection System

SafeWatch is an enterprise-grade surveillance intelligence platform designed to detect complex human behaviors and threats in real-time. Built for high-performance CPU inference, it integrates YOLOv8, MediaPipe, and custom behavioral analytics.

## Core Features

- **Real-Time Threat Detection**: Fight, Assault, Harassment, Fall, Unconscious, Trespassing, Crowd Panic, and more.
- **Multi-Camera Support**: RTSP and USB stream management with auto-recovery.
- **Smart Alerting**: Instant Telegram notifications with annotated snapshots.
- **Behavioral Intelligence**: Skeleton geometry analysis and kinetic velocity tracking.
- **SOC Dashboard**: Dark-themed monitoring and incident analytics via Streamlit.
- **Local CPU Inference**: Fully optimized for edge deployment without cloud dependencies.

## Installation

```bash
git clone https://github.com/abarnesh01/safewatch.git
cd safewatch
pip install -r requirements.txt
python setup.py install
```

## Quick Start

1.  Configure your cameras in `config.yaml`.
2.  Set up your Telegram Bot in `.env`.
3.  Launch the surveillance engine:
    ```bash
    python main.py
    ```
4.  Launch the dashboard:
    ```bash
    streamlit run dashboard/app.py
    ```

## Project Structure

- `capture/`: Stream management and frame sampling.
- `detection/`: YOLOv8, MediaPipe, and Optical Flow engines.
- `threats/`: Behavioral threat detection logic.
- `classifier/`: Pose feature extraction and action recognition.
- `alerts/`: Telegram notification and snapshot building.
- `database/`: Incident logging and camera health tracking.
- `dashboard/`: Streamlit SOC interface.

## License
Enterprise Proprietary - SafeWatch AI
