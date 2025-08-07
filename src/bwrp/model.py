"""
AI model for pneumonia detection using deep learning.
Implements ResNet18 architecture for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import math
import sys
import io
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import json
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PneumoniaDetectionModel:
    """ResNet18-based model for pneumonia detection from chest X-rays."""

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize the pneumonia detection model.

        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model: Optional[nn.Module] = None
        self.class_names = ["NORMAL", "PNEUMONIA"]
        self.model_name = "pneumonia_detector_resnet18_v1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define transforms for preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def build_model(self) -> nn.Module:
        """
        Build the ResNet18 model architecture.

        Returns:
            PyTorch model
        """
        model = self._build_resnet18()
        self.model = model.to(self.device)
        logger.info("ResNet18 model built successfully")
        return self.model

    def _build_resnet18(self) -> nn.Module:
        """Build ResNet18 architecture for binary classification."""
        # Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)

        # Modify the first layer to accept 3-channel input (if needed)
        # ResNet18 already accepts 3-channel RGB input by default

        # Replace the final fully connected layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, 1), nn.Sigmoid()
        )

        return model

    def calculate_class_weights(self, train_loader: DataLoader) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of class weights
        """
        # Count samples per class
        class_counts = [0, 0]  # [NORMAL, PNEUMONIA]

        for _, labels in train_loader:
            labels_np = labels.numpy()
            class_counts[0] += np.sum(labels_np == 0)
            class_counts[1] += np.sum(labels_np == 1)

        total_samples = sum(class_counts)

        # Calculate weights (inverse frequency)
        class_weights = {
            0: total_samples / (2.0 * class_counts[0]),
            1: total_samples / (2.0 * class_counts[1]),
        }

        logger.info(
            f"Class distribution: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}"
        )
        logger.info(f"Class weights: {class_weights}")

        return class_weights

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        model_dir: str = "models",
        use_class_weights: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model on the provided datasets with PyTorch.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            model_dir: Directory to save model checkpoints
            use_class_weights: Whether to use class weights for imbalanced data

        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)

        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            class_weights_dict = self.calculate_class_weights(train_loader)
            class_weights = torch.tensor(
                [class_weights_dict[0], class_weights_dict[1]], dtype=torch.float32
            ).to(self.device)

        # Define loss function and optimizer
        if class_weights is not None:
            criterion = nn.BCELoss(weight=class_weights[1])  # Weight for positive class
        else:
            criterion = nn.BCELoss()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=0.0001
        )

        # Training history
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0

        logger.info(f"Starting ResNet18 training for {epochs} epochs...")
        logger.info(f"Using class weights: {use_class_weights}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = (
                        inputs.to(self.device),
                        labels.to(self.device).float(),
                    )
                    outputs = self.model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            # Update history
            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss / len(val_loader))
            history["val_acc"].append(val_acc)

            # Update learning rate
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(),
                    str(model_path / f"{self.model_name}_best.pth"),
                )

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Loss: {history['val_loss'][-1]:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

        # Save final model
        torch.save(
            self.model.state_dict(), str(model_path / f"{self.model_name}_final.pth")
        )

        # Save training history
        history_path = model_path / f"{self.model_name}_history.json"
        with open(str(history_path), "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return history

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test dataset.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Evaluating model on test dataset...")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.BCELoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        evaluation_results = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        logger.info(f"Evaluation results: {evaluation_results}")
        return evaluation_results

    def predict_image(self, image: np.ndarray, use_tta: bool = False) -> Dict[str, Any]:
        """
        Predict pneumonia probability for a single image.

        Args:
            image: Preprocessed image array (numpy) or PIL Image
            use_tta: Whether to use Test Time Augmentation for better accuracy

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.eval()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # If image is already normalized (0-1), denormalize it
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Apply transforms
        if use_tta:
            # Test Time Augmentation - average predictions from multiple versions
            predictions = []

            # Original image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.sigmoid(self.model(img_tensor)).cpu().numpy()[0][0]
                predictions.append(pred)

            # Horizontal flip
            flipped = transforms.functional.hflip(image)
            img_tensor = self.transform(flipped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.sigmoid(self.model(img_tensor)).cpu().numpy()[0][0]
                predictions.append(pred)

            # Average all predictions
            probability = float(np.mean(predictions))

        else:
            # Single prediction
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(img_tensor)
                probability = float(prediction.cpu().numpy()[0][0])

        # Determine class
        predicted_class = "PNEUMONIA" if probability > 0.5 else "NORMAL"
        confidence = probability if probability > 0.5 else 1 - probability

        return {
            "predicted_class": predicted_class,
            "probability": probability,
            "confidence": confidence,
            "pneumonia_probability": probability,
            "normal_probability": 1 - probability,
            "used_tta": use_tta,
        }

    def load_model(self, model_path: str):
        """
        Load a trained PyTorch model.

        Args:
            model_path: Path to the saved model file (.pth or .pt)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Build model architecture first
        self.build_model()

        # Load the state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle case where model is wrapped in DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            # Remove 'module.' prefix
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded from {model_path}")

    def save_model(self, model_path: str):
        """
        Save the trained PyTorch model.

        Args:
            model_path: Path to save the model file (.pth or .pt)
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        # Save only the state dict (recommended approach)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def get_model_summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
            String representation of model summary
        """
        if self.model is None:
            return "Model not built yet"

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout

        return buffer.getvalue()


class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts learning rate scheduler for PyTorch."""

    def __init__(self, optimizer, first_restart_step=5, t_mul=2.0, eta_min=0.0001):
        self.optimizer = optimizer
        self.first_restart_step = first_restart_step
        self.t_mul = t_mul
        self.eta_min = eta_min
        self.current_restart_step = first_restart_step
        self.last_restart = 0
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        """Update learning rate for current epoch."""
        # Simple cosine annealing
        fraction = (epoch % self.first_restart_step) / self.first_restart_step
        lr = (
            self.eta_min
            + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * fraction)) / 2
        )

        # Set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


def create_and_train_model(
    train_dir: str, val_dir: str, test_dir: str, epochs: int = 20, batch_size: int = 32
) -> Tuple[PneumoniaDetectionModel, Dict[str, Any]]:
    """
    Complete pipeline to create and train a pneumonia detection model using PyTorch.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    # Import here to avoid circular imports
    from .data_loader import PneumoniaDataLoader

    # Create data loader instance
    loader = PneumoniaDataLoader()
    loader.dataset_path = Path(train_dir).parent  # Assuming train_dir is inside dataset
    loader.batch_size = batch_size

    # Create data loaders
    train_loader, val_loader, test_loader = loader.create_data_loaders()

    # Create and build model
    model = PneumoniaDetectionModel()
    model.build_model()

    # Print model summary
    logger.info("Model Architecture:")
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}"
    )

    # Train model
    history = model.train_model(train_loader, val_loader, epochs=epochs)

    # Evaluate model
    evaluation_results = model.evaluate_model(test_loader)

    return model, evaluation_results


def create_data_loaders(
    train_dir: str,
    val_dir: str = None,
    test_dir: str = None,
    batch_size: int = 32,
    img_size: tuple = (224, 224),
    num_workers: int = 4,
) -> tuple:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        batch_size: Batch size for training
        img_size: Target image size (height, width)
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Optimized Pneumonia Detection Model")
    print("=" * 50)

    # Create model instance
    model = PneumoniaDetectionModel()

    # Build model
    model.build_model()

    # Print model summary
    print(model.get_model_summary())
