# 🛡️ SafeWatch Enterprise V2
**Edge-Optimized Multi-Modal AI Surveillance using Threat Fusion and Adaptive Intelligence**

SafeWatch is an autonomous, planetary-scale surveillance ecosystem that blends cutting-edge YOLO object detection, Mediapipe pose estimation, optical flow analysis, and InsightFace facial recognition into a unified, CPU-optimized edge node pipeline.

## 🌟 Key Features
- **Threat Fusion Engine**: Dynamically weights loitering, crowd density, behavior anomalies, and watchlists into a 0-100 threat score.
- **Adaptive Intelligence**: Self-calibrating alarm thresholds that autonomously harden if a camera is too noisy.
- **Active Learning MLOps**: Automatically isolates false positives and stages them in a `retraining_dataset` for CI/CD model refinement.
- **Edge-to-Cloud Federation**: Resilient offline queuing of incident metadata and async transmissions to a Central SOC Hub without transmitting raw video.
- **Enterprise Privacy Layer**: Dynamic Gaussian face redaction complying with GDPR/CCPA, supporting `PUBLIC`, `INTERNAL`, and `SECURITY_ONLY` modes.
- **Automated Forensics**: 15-second rolling buffer that automatically dumps high-quality `.mp4` evidence on critical threats.

## 🏗️ Architecture
SafeWatch operates via a multi-threaded asynchronous pipeline:
1. **Ingestion Loop**: Thread-safe RTSP frame capture and jitter buffering.
2. **Detection Loop**: Quantized YOLOv8 bounding boxes + InsightFace embeddings.
3. **Analysis Loop**: Spatial heatmaps, velocity tracking, and optical flow.
4. **Threat Engine**: Fuses multiple AI signals into actionable alerts.
5. **Observability**: Prometheus telemetry sidecars exposing real-time API metrics.

## 🚀 Quick Start
```bash
docker-compose up --build -d
```
Navigate to `http://localhost:8501` to access the SOC Intelligence Dashboard.
