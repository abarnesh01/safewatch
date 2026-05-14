"""
SafeWatch — Dataset Downloader
Downloads open-source action recognition and anomaly detection datasets.
Designed for Google Colab use.
"""

import os
import zipfile
import shutil
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None


class DatasetDownloader:
    """Downloads open-source datasets for SafeWatch model training."""

    DATASETS = {
        "rwf2000": {
            "name": "RWF-2000 Fight Detection",
            "description": "2000 video clips of fight/non-fight scenarios",
            "url": "https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection",
            "type": "github",
        },
        "hockey_fight": {
            "name": "Hockey Fight Dataset",
            "description": "1000 hockey fight/non-fight video clips",
            "url": "https://academictorrents.com/details/38d9ed996a5a75a039b84571f6e2b02e",
            "type": "torrent",
        },
        "ucf_crime": {
            "name": "UCF-Crime Dataset",
            "description": "13 anomaly classes from real-world surveillance",
            "url": "https://www.crcv.ucf.edu/projects/real-world/",
            "type": "web",
        },
        "le2i_fall": {
            "name": "Le2i Fall Detection",
            "description": "Fall detection video dataset",
            "url": "http://le2i.cnrs.fr/Fall-detection-Dataset",
            "type": "web",
        },
    }

    def __init__(self, output_dir: str = "data/raw"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"DatasetDownloader(output_dir='{self._output_dir}')"

    def list_datasets(self) -> dict:
        """List all available datasets and their download info."""
        return {k: {"name": v["name"], "description": v["description"], "url": v["url"]}
                for k, v in self.DATASETS.items()}

    def download_rwf2000(self, output_path: str = None):
        """Download RWF-2000 dataset (requires manual setup — provides instructions)."""
        dest = Path(output_path) if output_path else self._output_dir / "rwf2000"
        dest.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("RWF-2000 Dataset Download Instructions")
        print("=" * 60)
        print(f"1. Visit: {self.DATASETS['rwf2000']['url']}")
        print("2. Follow the instructions to request access")
        print("3. Download the dataset ZIP files")
        print(f"4. Extract to: {dest}")
        print("5. Expected structure:")
        print("   rwf2000/")
        print("   ├── train/")
        print("   │   ├── Fight/  (1000 videos)")
        print("   │   └── NonFight/  (1000 videos)")
        print("   └── val/")
        print("       ├── Fight/  (200 videos)")
        print("       └── NonFight/  (200 videos)")
        print("=" * 60)

        readme = dest / "DOWNLOAD_INSTRUCTIONS.txt"
        readme.write_text(
            f"RWF-2000 Dataset\n"
            f"URL: {self.DATASETS['rwf2000']['url']}\n"
            f"Follow the GitHub instructions to request access.\n"
        )
        return str(dest)

    def download_hockey_fight(self, output_path: str = None):
        """Download Hockey Fight dataset instructions."""
        dest = Path(output_path) if output_path else self._output_dir / "hockey_fight"
        dest.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Hockey Fight Dataset Download Instructions")
        print("=" * 60)
        print(f"1. Visit: {self.DATASETS['hockey_fight']['url']}")
        print("2. Download the dataset")
        print(f"3. Extract to: {dest}")
        print("4. Expected structure:")
        print("   hockey_fight/")
        print("   ├── fight/  (500 videos)")
        print("   └── nonfight/  (500 videos)")
        print("=" * 60)

        return str(dest)

    def download_ucf_crime(self, output_path: str = None):
        """Download UCF-Crime dataset subset instructions."""
        dest = Path(output_path) if output_path else self._output_dir / "ucf_crime"
        dest.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("UCF-Crime Dataset Download Instructions")
        print("=" * 60)
        print(f"1. Visit: {self.DATASETS['ucf_crime']['url']}")
        print("2. Download the relevant anomaly category videos")
        print("3. Categories relevant to SafeWatch:")
        print("   - Assault")
        print("   - Fighting")
        print("   - Abuse")
        print("   - Arrest (for reference)")
        print("   - Normal_Videos")
        print(f"4. Extract to: {dest}")
        print("=" * 60)

        return str(dest)

    def download_all(self):
        """Download all datasets (provides instructions for each)."""
        self.download_rwf2000()
        self.download_hockey_fight()
        self.download_ucf_crime()
        print("\n✅ Download instructions provided for all datasets.")
        print("After downloading, run DatasetPrep to process the data.")
