"""
SafeWatch Dataset Downloader
Automates the retrieval of public surveillance datasets for training.
"""

import os
from pathlib import Path
from loguru import logger

def download_datasets():
    """Download RWF-2000, UCF-Crime, and other relevant datasets."""
    data_dir = Path("training/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "RWF-2000": "https://github.com/m666/RWF-2000",
        "UCF-Crime": "https://www.crcv.ucf.edu/data/UCF_Crime_Dataset.php"
    }
    
    for name, url in datasets.items():
        logger.info("Dataset {} available at: {}", name, url)
        # In a real environment, this would call wget/gdown or kaggle API
    
    logger.info("Dataset downloader finished (manual download required for large sets)")

if __name__ == "__main__":
    download_datasets()
