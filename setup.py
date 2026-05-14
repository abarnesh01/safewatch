"""
SafeWatch Setup Script
"""

from setuptools import setup, find_packages

setup(
    name="safewatch",
    version="1.0.0",
    description="AI-Powered CCTV Threat Detection System",
    author="SafeWatch Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
        "python-telegram-bot>=20.0",
        "streamlit>=1.28.0",
        "imutils>=0.5.4",
        "Pillow>=10.0.0",
        "schedule>=1.2.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "safewatch=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
