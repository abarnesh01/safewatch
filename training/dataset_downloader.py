"""
SafeWatch — DatasetDownloader
Automates the retrieval of public surveillance datasets for training.
Supports RWF-2000, UCF-Crime, Hockey Fight, and more.
"""

import os
import zipfile
import requests
from pathlib import Path
from loguru import logger
from typing import Optional

class DatasetDownloader:
    """
    Downloads and extracts common surveillance datasets.
    Note: Some datasets require manual registration or Kaggle API.
    """

    DATASETS = {
        "RWF-2000": {
            "url": "https://github.com/mo-mo-666/RWF-2000/archive/refs/heads/master.zip",
            "filename": "rwf2000.zip",
            "extract_to": "rwf2000"
        },
        "HockeyFight": {
            "url": "http://visilab.etsii.uclm.es/personas/oscar/HockeyFights/hockey_fight.zip",
            "filename": "hockey_fight.zip",
            "extract_to": "hockey_fight"
        },
        "UCF-Crime-Small": {
            "url": "https://raw.githubusercontent.com/airctic/rashomon/master/data/ucf_crime_subset.zip",
            "filename": "ucf_crime_subset.zip",
            "extract_to": "ucf_crime"
        }
    }

    def __init__(self, download_dir: str = "data/raw"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetDownloader initialized. Target: {self.download_dir}")

    def download_file(self, url: str, filename: str) -> Optional[Path]:
        """Download a file with progress tracking."""
        target_path = self.download_dir / filename
        if target_path.exists():
            logger.info(f"File {filename} already exists, skipping download.")
            return target_path

        logger.info(f"Downloading {url} to {target_path}...")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Successfully downloaded {filename}")
            return target_path
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def extract_zip(self, zip_path: Path, extract_to: str):
        """Extract a zip file to a specific directory."""
        target_dir = self.download_dir / extract_to
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {target_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Extraction complete for {extract_to}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")

    def download_all(self):
        """Download and extract all registered datasets."""
        for name, info in self.DATASETS.items():
            logger.info(f"Processing dataset: {name}")
            path = self.download_file(info["url"], info["filename"])
            if path:
                self.extract_zip(path, info["extract_to"])
        
        logger.info("All datasets processed successfully.")

    def print_manual_instructions(self):
        """Print instructions for datasets that cannot be easily automated."""
        print("\n" + "="*50)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*50)
        print("1. UCF-Crime (Full): Download from https://www.crcv.ucf.edu/data/UCF_Crime_Dataset.php")
        print("2. Le2i Fall: Download from http://le2i.cnrs.fr/Fall-detection-Dataset")
        print("3. Place downloaded files in: " + str(self.download_dir))
        print("="*50 + "\n")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all()
    downloader.print_manual_instructions()
