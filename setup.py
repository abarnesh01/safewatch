from setuptools import setup, find_packages

setup(
    name="safewatch",
    version="1.0.0",
    description="AI-Powered CCTV Threat Detection System",
    author="abarnesh01",
    author_email="abarnesh772@gmail.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "ultralytics",
        "mediapipe",
        "onnxruntime",
        "numpy",
        "pyyaml",
        "streamlit",
        "loguru",
        "python-telegram-bot"
    ],
    entry_points={
        "console_scripts": [
            "safewatch=main:main",
        ],
    },
    python_requires=">=3.10",
)
