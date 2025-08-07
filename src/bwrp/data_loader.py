"""
Data loading and preprocessing module for pneumonia detection.
Handles Kaggle dataset integration and image preprocessing using PyTorch.
"""

import os
import kagglehub
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PneumoniaDataLoader:
    """Handles loading and preprocessing of the pneumonia chest X-ray dataset."""

    def __init__(self, dataset_name: str = "paultimothymooney/chest-xray-pneumonia"):
        """
        Initialize the data loader.

        Args:
            dataset_name: Kaggle dataset identifier
        """
        self.dataset_name = dataset_name
        self.dataset_path: Optional[Path] = None
        self.image_size = (224, 224)  # Standard input size for most models
        self.batch_size = 32

    def download_dataset(self) -> Path:
        """
        Download the dataset from Kaggle using kagglehub.

        Returns:
            Path to the downloaded dataset
        """
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            self.dataset_path = Path(dataset_path)
            logger.info(f"Dataset downloaded to: {self.dataset_path}")
            return self.dataset_path
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def explore_dataset_structure(self) -> Dict[str, any]:
        """
        Explore and analyze the dataset structure.

        Returns:
            Dictionary containing dataset information
        """
        if not self.dataset_path:
            self.download_dataset()

        structure = {}

        # Walk through the dataset directory
        for root, dirs, files in os.walk(self.dataset_path):
            rel_path = os.path.relpath(root, self.dataset_path)
            if rel_path == ".":
                rel_path = "root"

            structure[rel_path] = {
                "directories": dirs,
                "file_count": len(files),
                "file_types": list(
                    set([os.path.splitext(f)[1] for f in files if f.startswith(".")])
                ),
            }

        logger.info(f"Dataset structure: {structure}")
        return structure

    def create_data_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training, validation, and testing.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if not self.dataset_path:
            self.download_dataset()

        try:
            # Look for common dataset structure patterns
            chest_xray_path = self.dataset_path / "chest_xray"
            if chest_xray_path.exists():
                base_path = chest_xray_path
            else:
                base_path = self.dataset_path

            # Define transforms for training (with augmentation)
            train_transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Define transforms for validation/test (no augmentation)
            val_test_transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Create datasets using ImageFolder
            train_dataset = datasets.ImageFolder(
                root=base_path / "train", transform=train_transform
            )

            # For validation, we'll use a subset of training data
            # Split training dataset for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            # Apply different transforms to validation subset
            val_dataset = datasets.ImageFolder(
                root=base_path / "train", transform=val_test_transform
            )
            val_subset.dataset = val_dataset

            # Create test dataset
            test_dataset = datasets.ImageFolder(
                root=base_path / "test", transform=val_test_transform
            )

            # Create DataLoaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            logger.info("PyTorch DataLoaders created successfully")
            logger.info(
                f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_dataset)}"
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise

    def get_class_names(self) -> list:
        """
        Get the class names from the dataset.

        Returns:
            List of class names
        """
        try:
            if not self.dataset_path:
                self.download_dataset()

            # Look for train directory to get class names
            chest_xray_path = self.dataset_path / "chest_xray"
            if chest_xray_path.exists():
                train_path = chest_xray_path / "train"
            else:
                train_path = self.dataset_path / "train"

            if train_path.exists():
                class_names = [d.name for d in train_path.iterdir() if d.is_dir()]
                logger.info(f"Found classes: {class_names}")
                return sorted(class_names)
            else:
                logger.warning("Train directory not found, using default classes")
                return ["NORMAL", "PNEUMONIA"]

        except Exception as e:
            logger.error(f"Failed to get class names: {e}")
            return ["NORMAL", "PNEUMONIA"]


def get_dataset_info() -> Dict[str, any]:
    """
    Get comprehensive information about the pneumonia dataset.

    Returns:
        Dictionary with dataset metadata
    """
    loader = PneumoniaDataLoader()

    info = {
        "dataset_name": loader.dataset_name,
        "description": "Chest X-Ray Images (Pneumonia) dataset",
        "source": "Kaggle - Paul Mooney",
        "classes": loader.get_class_names(),
        "image_size": loader.image_size,
        "batch_size": loader.batch_size,
        "purpose": "Binary classification: Normal vs Pneumonia chest X-rays",
    }

    try:
        info["structure"] = loader.explore_dataset_structure()
    except Exception as e:
        logger.warning(f"Could not explore dataset structure: {e}")
        info["structure"] = "Not available"

    return info


if __name__ == "__main__":
    # Example usage
    loader = PneumoniaDataLoader()

    # Download and explore dataset
    dataset_info = get_dataset_info()
    print("Dataset Info:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    # Create data loaders
    try:
        train_loader, val_loader, test_loader = loader.create_data_loaders()
        print(f"\nDataLoaders created successfully!")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"Image data type: {images.dtype}, Label data type: {labels.dtype}")
            break

    except Exception as e:
        print(f"Error creating datasets: {e}")
