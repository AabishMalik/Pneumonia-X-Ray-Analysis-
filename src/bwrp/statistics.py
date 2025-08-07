"""
Model statistics and performance analysis module for BWRP pneumonia detection.
Provides comprehensive metrics, visualizations, and model insights.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatistics:
    """Comprehensive model statistics and analysis."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize statistics analyzer.

        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model_info_path = self.model_dir / "model_info.json"
        self.history_path = self.model_dir / "pneumonia_detector_v1_history.json"
        self.model_path = self.model_dir / "pneumonia_detector_v1_best.h5"

    def load_training_history(self) -> Dict:
        """Load training history from JSON file."""
        try:
            if self.history_path.exists():
                with open(self.history_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning("Training history file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return {}

    def load_model_info(self) -> Dict:
        """Load model information from JSON file."""
        try:
            if self.model_info_path.exists():
                with open(self.model_info_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning("Model info file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading model info: {e}")
            return {}

    def calculate_model_parameters(self, model=None) -> Dict:
        """Calculate model architecture parameters for PyTorch model."""
        try:
            if model is None:
                if not self.model_path.exists():
                    return {"error": "Model file not found"}

                # For PyTorch, we need to load the model differently
                # This would need the model class definition
                logger.warning("Model parameter calculation requires model instance")
                return {"error": "Model instance required for PyTorch models"}

            # Count parameters for PyTorch model
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            non_trainable_params = total_params - trainable_params

            # Get model size (if saved)
            model_size_mb = 0
            if self.model_path.exists():
                model_size_mb = self.model_path.stat().st_size / (1024 * 1024)

            # Count layers
            layer_count = len(list(model.modules()))

            # Get layer types
            layer_types = {}
            for module in model.modules():
                module_type = type(module).__name__
                if module_type != "ResNet":  # Skip the main model container
                    layer_types[module_type] = layer_types.get(module_type, 0) + 1

            return {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(non_trainable_params),
                "model_size_mb": round(model_size_mb, 2),
                "layer_count": layer_count,
                "layer_types": layer_types,
            }

        except Exception as e:
            logger.error(f"Error calculating model parameters: {e}")
            return {"error": str(e)}

    def calculate_training_metrics(self) -> Dict:
        """Calculate comprehensive training metrics."""
        history = self.load_training_history()

        if not history:
            return {"error": "No training history available"}

        metrics = {}

        # Basic metrics
        epochs = len(history.get("loss", []))
        metrics["epochs_trained"] = epochs

        if epochs > 0:
            # Final metrics
            metrics["final_train_loss"] = (
                round(history["loss"][-1], 4) if "loss" in history else None
            )
            metrics["final_val_loss"] = (
                round(history["val_loss"][-1], 4) if "val_loss" in history else None
            )

            # Accuracy metrics
            if "accuracy" in history:
                metrics["final_train_accuracy"] = round(history["accuracy"][-1], 4)
                metrics["best_train_accuracy"] = round(max(history["accuracy"]), 4)

            if "val_accuracy" in history:
                metrics["final_val_accuracy"] = round(history["val_accuracy"][-1], 4)
                metrics["best_val_accuracy"] = round(max(history["val_accuracy"]), 4)

            # Precision metrics
            if "precision" in history:
                metrics["final_precision"] = round(history["precision"][-1], 4)
                metrics["best_precision"] = round(max(history["precision"]), 4)

            if "val_precision" in history:
                metrics["final_val_precision"] = round(history["val_precision"][-1], 4)
                metrics["best_val_precision"] = round(max(history["val_precision"]), 4)

            # Recall metrics
            if "recall" in history:
                metrics["final_recall"] = round(history["recall"][-1], 4)
                metrics["best_recall"] = round(max(history["recall"]), 4)

            if "val_recall" in history:
                metrics["final_val_recall"] = round(history["val_recall"][-1], 4)
                metrics["best_val_recall"] = round(max(history["val_recall"]), 4)

            # Calculate improvement metrics
            if "val_loss" in history and len(history["val_loss"]) > 1:
                loss_improvement = history["val_loss"][0] - history["val_loss"][-1]
                metrics["loss_improvement"] = round(loss_improvement, 4)

            if "val_accuracy" in history and len(history["val_accuracy"]) > 1:
                accuracy_improvement = (
                    history["val_accuracy"][-1] - history["val_accuracy"][0]
                )
                metrics["accuracy_improvement"] = round(accuracy_improvement, 4)

        return metrics

    def generate_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        model_info = self.load_model_info()
        training_metrics = self.calculate_training_metrics()
        model_params = self.calculate_model_parameters()

        # Dataset statistics from model info
        dataset_stats = {}
        if "dataset_info" in model_info:
            dataset_info = model_info["dataset_info"]
            dataset_stats = {
                "dataset_name": dataset_info.get("dataset_name", "Unknown"),
                "classes": dataset_info.get("classes", []),
                "image_size": dataset_info.get("image_size", [224, 224]),
                "batch_size": dataset_info.get("batch_size", 32),
            }

            # Calculate dataset statistics from structure if available
            if "structure" in dataset_info:
                structure = dataset_info["structure"]
                total_images = 0
                class_distribution = {}

                for path, info in structure.items():
                    if "NORMAL" in path and info.get("file_count", 0) > 0:
                        class_distribution["NORMAL"] = (
                            class_distribution.get("NORMAL", 0) + info["file_count"]
                        )
                        total_images += info["file_count"]
                    elif "PNEUMONIA" in path and info.get("file_count", 0) > 0:
                        class_distribution["PNEUMONIA"] = (
                            class_distribution.get("PNEUMONIA", 0) + info["file_count"]
                        )
                        total_images += info["file_count"]

                dataset_stats["total_images"] = total_images
                dataset_stats["class_distribution"] = class_distribution

                if total_images > 0:
                    dataset_stats["class_balance"] = {
                        "NORMAL": round(
                            class_distribution.get("NORMAL", 0) / total_images * 100, 1
                        ),
                        "PNEUMONIA": round(
                            class_distribution.get("PNEUMONIA", 0) / total_images * 100,
                            1,
                        ),
                    }

        # Training configuration
        training_config = {
            "architecture": model_info.get("architecture", "Unknown"),
            "epochs_trained": model_info.get("epochs_trained", 0),
            "optimizer": "Adam",  # From model definition
            "loss_function": "Binary Crossentropy",
            "metrics": ["Accuracy", "Precision", "Recall"],
        }

        # Performance rating
        performance_rating = self._calculate_performance_rating(training_metrics)

        return {
            "model_info": {
                "name": "BWRP Pneumonia Detector v1",
                "version": "1.0.0",
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": "Deep Convolutional Neural Network",
            },
            "dataset_statistics": dataset_stats,
            "training_configuration": training_config,
            "model_parameters": model_params,
            "training_metrics": training_metrics,
            "performance_rating": performance_rating,
            "recommendations": self._generate_recommendations(
                training_metrics, model_params
            ),
        }

    def _calculate_performance_rating(self, metrics: Dict) -> Dict:
        """Calculate overall performance rating."""
        if "error" in metrics:
            return {"overall": "Unknown", "details": "Insufficient data"}

        # Get key metrics
        val_accuracy = metrics.get("final_val_accuracy", 0)
        val_precision = metrics.get("final_val_precision", 0)
        val_recall = metrics.get("final_val_recall", 0)

        # Calculate F1 score
        if val_precision and val_recall:
            f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        else:
            f1_score = 0

        # Determine rating
        avg_score = (val_accuracy + f1_score) / 2 if f1_score > 0 else val_accuracy

        if avg_score >= 0.9:
            rating = "Excellent"
            color = "green"
        elif avg_score >= 0.8:
            rating = "Good"
            color = "blue"
        elif avg_score >= 0.7:
            rating = "Fair"
            color = "orange"
        elif avg_score >= 0.6:
            rating = "Poor"
            color = "red"
        else:
            rating = "Needs Improvement"
            color = "red"

        return {
            "overall": rating,
            "score": round(avg_score, 3),
            "color": color,
            "f1_score": round(f1_score, 3),
            "details": {
                "accuracy": round(val_accuracy, 3),
                "precision": round(val_precision, 3),
                "recall": round(val_recall, 3),
            },
        }

    def _generate_recommendations(
        self, training_metrics: Dict, model_params: Dict
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if "error" in training_metrics:
            recommendations.append("Insufficient training data available for analysis")
            return recommendations

        # Training-based recommendations
        val_accuracy = training_metrics.get("final_val_accuracy", 0)

        if val_accuracy < 0.7:
            recommendations.append(
                "Consider training for more epochs to improve accuracy"
            )
            recommendations.append(
                "Try data augmentation to increase dataset diversity"
            )
            recommendations.append(
                "Experiment with transfer learning using pre-trained models"
            )

        if val_accuracy < 0.8:
            recommendations.append(
                "Consider fine-tuning hyperparameters (learning rate, batch size)"
            )
            recommendations.append(
                "Add more regularization (dropout, batch normalization)"
            )

        # Check for overfitting
        train_acc = training_metrics.get("final_train_accuracy", 0)
        if train_acc - val_accuracy > 0.1:
            recommendations.append(
                "Model may be overfitting - consider adding regularization"
            )
            recommendations.append("Reduce model complexity or add more training data")

        # Model size recommendations
        if isinstance(model_params, dict) and "total_parameters" in model_params:
            total_params = model_params["total_parameters"]
            if total_params > 10_000_000:
                recommendations.append(
                    "Model is quite large - consider pruning for deployment"
                )
            elif total_params < 100_000:
                recommendations.append(
                    "Model might be too simple - consider adding complexity"
                )

        if not recommendations:
            recommendations.append(
                "Model performance looks good! Consider testing on more diverse datasets"
            )
            recommendations.append(
                "Monitor performance on real-world data for continuous improvement"
            )

        return recommendations

    def export_statistics_json(self) -> str:
        """Export all statistics as JSON for web frontend."""
        stats = self.generate_performance_summary()

        # Add timestamp
        stats["generated_at"] = datetime.now().isoformat()
        stats["version"] = "1.0.0"

        # Save to file
        output_path = self.model_dir / "model_statistics.json"
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics exported to {output_path}")
        return str(output_path)


def generate_model_statistics(model_dir: str = "models") -> Dict:
    """
    Generate comprehensive model statistics.

    Args:
        model_dir: Directory containing model files

    Returns:
        Dictionary containing all model statistics
    """
    stats = ModelStatistics(model_dir)
    return stats.generate_performance_summary()


def export_statistics_for_web(model_dir: str = "models") -> str:
    """
    Export model statistics in web-friendly format.

    Args:
        model_dir: Directory containing model files

    Returns:
        Path to exported JSON file
    """
    stats = ModelStatistics(model_dir)
    return stats.export_statistics_json()


if __name__ == "__main__":
    # Example usage
    print("🔍 BWRP Model Statistics Generator")
    print("=" * 50)

    try:
        stats = generate_model_statistics()

        print("\n📊 Model Performance Summary:")
        print(f"  Architecture: {stats['training_configuration']['architecture']}")
        print(f"  Epochs Trained: {stats['training_configuration']['epochs_trained']}")

        if (
            "training_metrics" in stats
            and "final_val_accuracy" in stats["training_metrics"]
        ):
            accuracy = stats["training_metrics"]["final_val_accuracy"]
            print(f"  Validation Accuracy: {accuracy:.1%}")

        if "performance_rating" in stats:
            rating = stats["performance_rating"]
            print(f"  Performance Rating: {rating['overall']} ({rating['score']:.1%})")

        if (
            "model_parameters" in stats
            and "total_parameters" in stats["model_parameters"]
        ):
            params = stats["model_parameters"]["total_parameters"]
            print(f"  Total Parameters: {params:,}")

        print("\n💡 Recommendations:")
        for rec in stats.get("recommendations", []):
            print(f"  • {rec}")

        # Export for web
        export_path = export_statistics_for_web()
        print(f"\n✅ Statistics exported to: {export_path}")

    except Exception as e:
        print(f"❌ Error generating statistics: {e}")
        import traceback

        traceback.print_exc()
