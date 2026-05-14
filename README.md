```
 ███████╗ █████╗ ███████╗███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
 ██╔════╝██╔══██╗██╔════╝██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
 ███████╗███████║█████╗  █████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
 ╚════██║██╔══██║██╔══╝  ██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
 ███████║██║  ██║██║     ███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
 ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
```

# SafeWatch — AI-Powered CCTV Threat Detection System

> 🛡️ Real-time surveillance threat detection using YOLOv8, MediaPipe, and LSTM action classification with instant Telegram alerts.

---

## 🎯 Overview

SafeWatch is a production-ready, AI-powered CCTV monitoring system that detects threats in real-time from camera feeds. It runs entirely on local hardware — no cloud services, no paid APIs. The system uses computer vision and deep learning to detect dangerous situations and instantly alerts security personnel via Telegram.

---

## 🚨 Threat Types Detected

| # | Threat | Description | Severity |
|---|--------|-------------|----------|
| 1 | **Fight Detection** | Physical fights between 2+ people | HIGH |
| 2 | **Fall Detection** | Sudden falls, slip-and-fall | MEDIUM-HIGH |
| 3 | **Harassment** | Sustained close proximity with aggression | MEDIUM-HIGH |
| 4 | **Physical Assault** | One-sided physical attack (victim/assailant roles) | CRITICAL |
| 5 | **Unconscious Person** | Extended motionless horizontal position | HIGH-CRITICAL |
| 6 | **Trespass** | Entry into restricted polygon zones | MEDIUM-HIGH |
| 7 | **Crowd Panic** | Mass movement in scattered directions | HIGH-CRITICAL |
| 8 | **Accident** | Multi-person falls, impact events | HIGH-CRITICAL |
| 9 | **Abuse** | Sustained repeated attacks over time | HIGH-CRITICAL |
| 10 | **Pass Away** | Extended unconsciousness (emergency) | CRITICAL |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 22.04 |
| Python | 3.10+ | 3.11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | Not required (CPU-only) | NVIDIA GPU (optional) |
| Storage | 2 GB + recordings space | 10 GB+ |
| Cameras | USB webcam or RTSP IP camera | 2+ IP cameras |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/safewatch.git
cd safewatch
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Telegram bot credentials
```

### 5. Configure Cameras
Edit `config.yaml` — update camera sources:
```yaml
cameras:
  - id: "CAM-01"
    source: 0                    # USB webcam
    enabled: true
  - id: "CAM-02"
    source: "rtsp://ip:port/stream"  # IP camera
    enabled: true
```

---

## 📱 Telegram Bot Setup

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```
4. Start a chat with your bot
5. Get your chat ID:
   - Send a message to your bot
   - Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Copy the `chat.id` value to `.env`:
   ```
   TELEGRAM_CHAT_ID_MAIN=your_chat_id
   ```

---

## 🚀 Running SafeWatch

### Start the Full System
```bash
python main.py
```

### Test Camera Connections
```bash
python main.py --test-cameras
```

### Test Telegram Bot
```bash
python main.py --test-telegram
```

### Dashboard Only
```bash
python main.py --dashboard-only
```

### Custom Config
```bash
python main.py --config my_config.yaml
```

---

## 📊 Dashboard Access

Open your browser to:
```
http://localhost:8501
```

The dashboard provides:
- 🖥️ **Live Monitor** — Real-time camera feeds with threat overlays
- 📋 **Incident History** — Filterable incident log with charts
- 📹 **Camera Management** — Camera status, controls, zone configuration
- ⚙️ **System Settings** — Threat thresholds, Telegram test, system logs

---

## 🧠 Training Guide (Google Colab)

Training happens on Google Colab — NOT on your local machine.

### Step 1: Prepare Datasets
Use the scripts in `training/` on Google Colab:
```python
from training.dataset_prep import DatasetPrep
prep = DatasetPrep()
prep.download_all()          # Get dataset links
prep.process_all_datasets()  # Extract frames & poses
prep.split_dataset()         # 80/20 train/val split
```

### Step 2: Train the LSTM Classifier
```python
from training.train_classifier import ActionClassifierTrainer
trainer = ActionClassifierTrainer()
trainer.train(epochs=50)
trainer.evaluate()
trainer.plot_curves()
```

### Step 3: Export to ONNX
```python
from training.export_onnx import ONNXExporter
exporter = ONNXExporter()
exporter.export()
exporter.validate()
exporter.benchmark()
```

### Step 4: Deploy
Download `action_classifier.onnx` and place it in `models/`.

### Datasets Used
| Dataset | Classes | Size |
|---------|---------|------|
| [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Fight/NonFight | 2000 clips |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | 13 anomaly types | Varies |
| [Le2i Fall](http://le2i.cnrs.fr/Fall-detection-Dataset) | Fall detection | Varies |
| [Hockey Fight](https://academictorrents.com/details/38d9ed996a5a75a039b84571f6e2b02e) | Fight/NonFight | 1000 clips |

---

## ⚙️ Configuration Guide

### Key Config Options

| Setting | Path | Default | Description |
|---------|------|---------|-------------|
| Frame Skip | `cameras[].frame_skip` | 5 | Process every Nth frame |
| YOLO Confidence | `detection.yolo_confidence` | 0.5 | Min detection confidence |
| Pose Complexity | `detection.pose_model_complexity` | 0 | 0=fast, 1=balanced, 2=heavy |
| Fight Threshold | `threats.fight.confidence_threshold` | 0.82 | Min confidence to trigger |
| Alert Cooldown | `telegram.alert_cooldown_seconds` | 30 | Seconds between repeat alerts |
| Recording | `recording.enabled` | true | Save video on threat |

---

## ⚡ Performance Tips for Low-End PCs

1. **Increase frame_skip** to 8-10 (process fewer frames)
2. **Use pose_model_complexity: 0** (fastest MediaPipe model)
3. **Reduce resolution** to 320x240
4. **Disable optical flow** if not needed (`enable_optical_flow: false`)
5. **Lower max_persons_tracked** to 5
6. **Disable unused threat detectors** in config
7. **Use YOLOv8n** (nano) — it's the fastest variant

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        main.py                            │
│                    (Orchestrator)                          │
├──────────┬──────────┬──────────┬──────────┬───────────────┤
│          │          │          │          │               │
│  Camera  │ Detection│ Classifier│ Threats │    Alerts     │
│  Capture │  Layer   │  Layer   │  Engine │    System     │
│          │          │          │          │               │
│ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────────┐ │
│ │Stream│ │ │YOLO  │ │ │Skel. │ │ │Fight │ │ │Telegram  │ │
│ │Mgr   │→│ │v8    │→│ │Anlyz │→│ │Fall  │→│ │Bot       │ │
│ │      │ │ │      │ │ │      │ │ │Harass│ │ │          │ │
│ │Frame │ │ │Media │ │ │Veloc.│ │ │Asslt │ │ │Snapshot  │ │
│ │Sample│ │ │Pipe  │ │ │Track │ │ │Panic │ │ │Builder   │ │
│ │      │ │ │      │ │ │      │ │ │Abuse │ │ │          │ │
│ │Camera│ │ │Opt.  │ │ │Action│ │ │Trsps │ │ │Alert     │ │
│ │Stream│ │ │Flow  │ │ │Class.│ │ │Accdn │ │ │Manager   │ │
│ └──────┘ │ │Zone  │ │ └──────┘ │ └──────┘ │ └──────────┘ │
│          │ │Mgr   │ │          │          │               │
│          │ └──────┘ │          │          │               │
├──────────┴──────────┴──────────┴──────────┴───────────────┤
│                     Database (SQLite)                      │
│                   Dashboard (Streamlit)                     │
└──────────────────────────────────────────────────────────┘
```

### Data Flow
```
Camera → Frame → Person Detection → Pose Estimation → Feature Extraction
                                                            ↓
Optical Flow ──────────────────────────────────→ Threat Engine
                                                            ↓
                                                    Threat Report
                                                            ↓
                                              Alert Manager → Telegram
                                                            ↓
                                                    Database Log
                                                            ↓
                                                    Dashboard UI
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
safewatch/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── capture/             # Camera stream management
├── detection/           # YOLO, MediaPipe, Optical Flow
├── classifier/          # Skeleton analysis, velocity, action classification
├── threats/             # 9 threat detectors + engine
├── alerts/              # Telegram bot, snapshots, alert management
├── database/            # SQLite logging
├── dashboard/           # Streamlit web UI
├── training/            # Colab training scripts
├── models/              # ML model files
├── logs/                # Log files
├── recordings/          # Threat snapshots & videos
└── tests/               # Unit tests
```

---

## 📄 License

This project is for educational and security research purposes.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

---

**Built with ❤️ for public safety**
