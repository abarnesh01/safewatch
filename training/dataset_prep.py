"""
SafeWatch — DatasetPrep
Prepares training data: extracts frames, extracts poses, splits dataset.
Designed for Google Colab.
"""

import os
import csv
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class DatasetPrep:
    """
    Prepares training data from raw video datasets.
    Extracts frames at 1fps, runs pose estimation, and organizes into
    train/val splits.
    """

    CLASSES = ["normal", "fight", "fall", "assault", "harassment", "abuse", "panic", "unconscious", "other"]

    CLASS_MAPPING = {
        "Fight": "fight",
        "fight": "fight",
        "NonFight": "normal",
        "nonfight": "normal",
        "Normal": "normal",
        "normal": "normal",
        "Assault": "assault",
        "Abuse": "abuse",
        "Fall": "fall",
        "fall": "fall",
    }

    def __init__(self, raw_dir: str = "data/raw", output_dir: str = "data", fps: int = 1):
        self._raw_dir = Path(raw_dir)
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._train_dir = self._output_dir / "train"
        self._val_dir = self._output_dir / "val"
        self._pose_estimator = None

        for cls in self.CLASSES:
            (self._train_dir / cls).mkdir(parents=True, exist_ok=True)
            (self._val_dir / cls).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"DatasetPrep(raw='{self._raw_dir}', output='{self._output_dir}')"

    def _init_pose(self):
        """Initialize MediaPipe pose estimator."""
        if MP_AVAILABLE and self._pose_estimator is None:
            self._pose_estimator = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )

    def extract_frames(self, video_path: str, output_dir: str, max_frames: int = 60) -> list[str]:
        """
        Extract frames from a video at configured FPS.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            max_frames: Maximum frames to extract per video

        Returns:
            List of saved frame paths
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ⚠️ Cannot open: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30

        frame_interval = int(video_fps / self._fps)
        if frame_interval <= 0:
            frame_interval = 1

        saved_paths = []
        frame_count = 0
        save_count = 0
        stem = Path(video_path).stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0 and save_count < max_frames:
                frame_resized = cv2.resize(frame, (224, 224))
                fname = f"{stem}_f{save_count:04d}.jpg"
                fpath = out / fname
                cv2.imwrite(str(fpath), frame_resized)
                saved_paths.append(str(fpath))
                save_count += 1

            frame_count += 1

        cap.release()
        return saved_paths

    def extract_poses(self, frame_path: str) -> Optional[np.ndarray]:
        """
        Extract MediaPipe pose landmarks from a frame image.

        Args:
            frame_path: Path to image file

        Returns:
            numpy array of shape (99,) — 33 landmarks x 3 coords, or None
        """
        self._init_pose()
        if self._pose_estimator is None:
            return None

        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose_estimator.process(rgb)

        if result.pose_landmarks is None:
            return None

        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])

        return np.array(landmarks, dtype=np.float32)

    def download_all(self):
        """Download all datasets (provides instructions)."""
        from training.dataset_downloader import DatasetDownloader
        downloader = DatasetDownloader(str(self._raw_dir))
        downloader.download_all()

    def process_raw_dataset(self, dataset_dir: str, class_name: str):
        """
        Process a raw dataset directory: extract frames and poses.

        Args:
            dataset_dir: Path to raw video directory
            class_name: SafeWatch class name to assign
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"  ⚠️ Dataset not found: {dataset_dir}")
            return

        mapped_class = self.CLASS_MAPPING.get(class_name, class_name)
        if mapped_class not in self.CLASSES:
            print(f"  ⚠️ Unknown class: {mapped_class}")
            return

        output = self._output_dir / "all_frames" / mapped_class
        output.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        videos = [f for f in dataset_path.rglob("*") if f.suffix.lower() in video_extensions]

        print(f"  Processing {len(videos)} videos for class '{mapped_class}'...")

        for i, video in enumerate(videos):
            frames = self.extract_frames(str(video), str(output / video.stem))
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(videos)} videos")

        print(f"  ✅ Finished processing '{mapped_class}'")

    def process_all_datasets(self):
        """Process all downloaded raw datasets."""
        print("Processing raw datasets...")

        # RWF-2000
        rwf_path = self._raw_dir / "rwf2000"
        if rwf_path.exists():
            for split in ["train", "val"]:
                for cls_dir in (rwf_path / split).iterdir():
                    if cls_dir.is_dir():
                        self.process_raw_dataset(str(cls_dir), cls_dir.name)

        # Hockey Fight
        hockey_path = self._raw_dir / "hockey_fight"
        if hockey_path.exists():
            for cls_dir in hockey_path.iterdir():
                if cls_dir.is_dir():
                    self.process_raw_dataset(str(cls_dir), cls_dir.name)

        # UCF Crime
        ucf_path = self._raw_dir / "ucf_crime"
        if ucf_path.exists():
            for cls_dir in ucf_path.iterdir():
                if cls_dir.is_dir():
                    self.process_raw_dataset(str(cls_dir), cls_dir.name)

        print("✅ All datasets processed")

    def split_dataset(self, train_ratio: float = 0.8):
        """
        Split processed frames into train/val sets.

        Args:
            train_ratio: Fraction of data for training
        """
        all_frames_dir = self._output_dir / "all_frames"
        if not all_frames_dir.exists():
            print("⚠️ No processed frames found. Run process_all_datasets() first.")
            return

        for cls_name in self.CLASSES:
            cls_dir = all_frames_dir / cls_name
            if not cls_dir.exists():
                continue

            all_files = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.png"))
            random.shuffle(all_files)

            split_idx = int(len(all_files) * train_ratio)
            train_files = all_files[:split_idx]
            val_files = all_files[split_idx:]

            for f in train_files:
                dest = self._train_dir / cls_name / f.name
                shutil.copy2(f, dest)

            for f in val_files:
                dest = self._val_dir / cls_name / f.name
                shutil.copy2(f, dest)

            print(f"  {cls_name}: {len(train_files)} train, {len(val_files)} val")

        print("✅ Dataset split complete")

    def extract_pose_sequences(self, frames_dir: str, output_path: str, seq_length: int = 30):
        """
        Extract pose sequences from frame directories and save as numpy arrays.

        Args:
            frames_dir: Directory containing class subdirectories of frames
            output_path: Path to save the numpy dataset
            seq_length: Number of frames per sequence
        """
        self._init_pose()

        X_sequences = []
        y_labels = []
        class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        frames_path = Path(frames_dir)
        for cls_name in self.CLASSES:
            cls_dir = frames_path / cls_name
            if not cls_dir.exists():
                continue

            # Group frames by video (by prefix)
            frame_groups: dict[str, list] = {}
            for f in sorted(cls_dir.rglob("*.jpg")):
                prefix = "_".join(f.stem.split("_")[:-1])
                if prefix not in frame_groups:
                    frame_groups[prefix] = []
                frame_groups[prefix].append(f)

            for prefix, frames in frame_groups.items():
                if len(frames) < seq_length:
                    continue

                # Extract poses for each frame
                pose_sequence = []
                for frame_path in frames[:seq_length]:
                    pose = self.extract_poses(str(frame_path))
                    if pose is not None:
                        pose_sequence.append(pose)
                    else:
                        pose_sequence.append(np.zeros(99, dtype=np.float32))

                if len(pose_sequence) == seq_length:
                    X_sequences.append(np.stack(pose_sequence))
                    y_labels.append(class_to_idx[cls_name])

        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int64)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out), X=X, y=y, classes=self.CLASSES)
        print(f"✅ Saved pose sequences: X={X.shape}, y={y.shape} to {out}")

    def augment_data(self, frame_path: str) -> list[np.ndarray]:
        """
        Apply data augmentation to a frame.

        Args:
            frame_path: Path to image

        Returns:
            List of augmented frame arrays
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return []

        augmented = []

        # Horizontal flip
        augmented.append(cv2.flip(frame, 1))

        # Brightness adjustment
        bright = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        augmented.append(bright)

        dark = cv2.convertScaleAbs(frame, alpha=0.8, beta=-20)
        augmented.append(dark)

        # Slight rotation
        h, w = frame.shape[:2]
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (w, h))
            augmented.append(rotated)

        return augmented

    def generate_manifest(self, output_path: str = "data/manifest.csv"):
        """
        Generate a manifest CSV mapping frame paths to labels.

        Args:
            output_path: Output CSV path
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for split in ["train", "val"]:
            split_dir = self._output_dir / split
            for cls_name in self.CLASSES:
                cls_dir = split_dir / cls_name
                if not cls_dir.exists():
                    continue
                for f in cls_dir.rglob("*"):
                    if f.suffix.lower() in {".jpg", ".png", ".npy"}:
                        rows.append({
                            "path": str(f),
                            "label": cls_name,
                            "split": split,
                        })

        with open(out, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["path", "label", "split"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ Manifest saved with {len(rows)} entries to {out}")
