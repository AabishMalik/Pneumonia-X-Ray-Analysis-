"""
Model visualization utilities for generating plots, matrices, and statistical images.
Generates visual representations of model performance and architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from PIL import Image
import io
import base64
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Set matplotlib backend for headless environments
import matplotlib

matplotlib.use("Agg")


class ModelVisualizer:
    """Generate visual representations of model performance and architecture."""

    def __init__(self, model, model_info: Optional[Dict] = None):
        """Initialize with model and optional model info."""
        self.model = model
        self.model_info = model_info or {}

        # Set style for better looking plots
        plt.style.use("default")
        sns.set_palette("husl")

    def generate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
    ) -> str:
        """Generate confusion matrix visualization."""
        if class_names is None:
            class_names = ["Normal", "Pneumonia"]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure
        plt.figure(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )

        plt.title(
            "Confusion Matrix\nModel Performance on Test Data",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")

        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(
            0.5,
            0.02,
            f"Overall Accuracy: {accuracy:.3f}",
            ha="center",
            fontsize=10,
            style="italic",
        )

        plt.tight_layout()

        # Convert to base64
        return self._fig_to_base64()

    def generate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> str:
        """Generate ROC curve visualization."""
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        # Create figure
        plt.figure(figsize=(8, 6))

        # Plot ROC curve
        plt.plot(
            fpr, tpr, color="darkorange", lw=3, label=f"ROC Curve (AUC = {roc_auc:.3f})"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        plt.title(
            "Receiver Operating Characteristic (ROC) Curve\nPneumonia Detection Performance",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        return self._fig_to_base64()

    def generate_performance_metrics_plot(self, metrics: Dict[str, float]) -> str:
        """Generate performance metrics radar chart."""
        # Prepare data
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        # Number of variables
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Add values
        values += values[:1]  # Complete the circle

        # Plot
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=3,
            label="Model Performance",
            color="#FF6B6B",
        )
        ax.fill(angles, values, alpha=0.25, color="#FF6B6B")

        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
        ax.grid(True)

        plt.title(
            "Model Performance Metrics\nRadar Chart Analysis",
            fontsize=16,
            fontweight="bold",
            pad=30,
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        return self._fig_to_base64()

    def generate_class_distribution_plot(self, class_counts: Dict[str, int]) -> str:
        """Generate class distribution visualization."""
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ["#FF9999", "#66B2FF"]
        explode = (0.05, 0.05)  # Slightly separate slices

        # Create figure
        plt.figure(figsize=(10, 6))

        # Create subplot for pie chart
        plt.subplot(1, 2, 1)
        wedges, texts, autotexts = plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )

        plt.title(
            "Dataset Class Distribution\n(Pie Chart)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Create subplot for bar chart
        plt.subplot(1, 2, 2)
        bars = plt.bar(
            labels, sizes, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )

        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(sizes) * 0.01,
                f"{size:,}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.title(
            "Dataset Class Distribution\n(Bar Chart)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Class", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Images", fontsize=12, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        return self._fig_to_base64()

    def generate_training_history_plot(self, history: Dict[str, List[float]]) -> str:
        """Generate training history visualization."""
        epochs = range(1, len(history.get("accuracy", [1])) + 1)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot accuracy
        if "accuracy" in history and "val_accuracy" in history:
            ax1.plot(
                epochs,
                history["accuracy"],
                "bo-",
                linewidth=2,
                markersize=6,
                label="Training Accuracy",
            )
            ax1.plot(
                epochs,
                history["val_accuracy"],
                "ro-",
                linewidth=2,
                markersize=6,
                label="Validation Accuracy",
            )
            ax1.set_title("Model Accuracy Over Epochs", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

        # Plot loss
        if "loss" in history and "val_loss" in history:
            ax2.plot(
                epochs,
                history["loss"],
                "bo-",
                linewidth=2,
                markersize=6,
                label="Training Loss",
            )
            ax2.plot(
                epochs,
                history["val_loss"],
                "ro-",
                linewidth=2,
                markersize=6,
                label="Validation Loss",
            )
            ax2.set_title("Model Loss Over Epochs", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Loss", fontsize=12, fontweight="bold")
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

        plt.suptitle("Training History Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return self._fig_to_base64()

    def generate_model_architecture_plot(self, layer_info: Dict[str, Any]) -> str:
        """Generate model architecture visualization."""
        layer_types = layer_info.get("layer_types", {})
        total_params = layer_info.get("total_parameters", 0)
        trainable_params = layer_info.get("trainable_parameters", 0)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Layer types distribution
        if layer_types:
            labels = list(layer_types.keys())
            sizes = list(layer_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

            ax1.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 10},
            )
            ax1.set_title("Layer Types Distribution", fontsize=12, fontweight="bold")

        # 2. Parameter distribution
        param_labels = ["Trainable", "Non-Trainable"]
        param_sizes = [trainable_params, total_params - trainable_params]
        param_colors = ["#66B2FF", "#FFB366"]

        ax2.pie(
            param_sizes,
            labels=param_labels,
            colors=param_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax2.set_title("Parameter Distribution", fontsize=12, fontweight="bold")

        # 3. Layer count bar chart
        if layer_types:
            bars = ax3.bar(labels, sizes, color=colors, alpha=0.8)
            ax3.set_title("Layers by Type", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Layer Type", fontsize=10)
            ax3.set_ylabel("Count", fontsize=10)
            ax3.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{size}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 4. Model size information
        ax4.axis("off")
        info_text = f"""
        Model Architecture Summary
        
        Total Parameters: {total_params:,}
        Trainable Parameters: {trainable_params:,}
        Non-Trainable: {total_params - trainable_params:,}
        
        Total Layers: {sum(layer_types.values()) if layer_types else 'N/A'}
        Model Size: {layer_info.get('model_size_mb', 'N/A')} MB
        """

        ax4.text(
            0.1,
            0.5,
            info_text,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

        plt.suptitle("Model Architecture Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return self._fig_to_base64()

    def generate_prediction_confidence_plot(self, predictions: List[Dict]) -> str:
        """Generate prediction confidence distribution plot."""
        confidences = [pred.get("confidence", 0) for pred in predictions]
        predictions_labels = [
            pred.get("predicted_class", "Unknown") for pred in predictions
        ]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Subplot 1: Confidence distribution histogram
        plt.subplot(2, 2, 1)
        plt.hist(confidences, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title("Prediction Confidence Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Confidence Score", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.grid(axis="y", alpha=0.3)

        # Subplot 2: Box plot of confidence by class
        plt.subplot(2, 2, 2)
        df = pd.DataFrame({"Confidence": confidences, "Class": predictions_labels})
        if len(df) > 0:
            sns.boxplot(data=df, x="Class", y="Confidence", ax=plt.gca())
            plt.title("Confidence by Predicted Class", fontsize=12, fontweight="bold")
            plt.xticks(rotation=45)

        # Subplot 3: Confidence vs prediction scatter
        plt.subplot(2, 2, 3)
        colors = [
            "red" if label == "Pneumonia" else "blue" for label in predictions_labels
        ]
        plt.scatter(range(len(confidences)), confidences, c=colors, alpha=0.6)
        plt.title("Prediction Confidence Timeline", fontsize=12, fontweight="bold")
        plt.xlabel("Prediction Index", fontsize=10)
        plt.ylabel("Confidence", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Subplot 4: Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis("off")
        if confidences:
            stats_text = f"""
            Confidence Statistics
            
            Mean: {np.mean(confidences):.3f}
            Median: {np.median(confidences):.3f}
            Std Dev: {np.std(confidences):.3f}
            Min: {np.min(confidences):.3f}
            Max: {np.max(confidences):.3f}
            
            High Confidence (>0.8): {sum(1 for c in confidences if c > 0.8)}
            Low Confidence (<0.6): {sum(1 for c in confidences if c < 0.6)}
            """
            plt.text(
                0.1,
                0.5,
                stats_text,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        plt.suptitle("Prediction Confidence Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return self._fig_to_base64()

    def _fig_to_base64(self) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buffer.seek(0)

        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        plt.close()  # Close the figure to free memory
        buffer.close()

        return f"data:image/png;base64,{image_base64}"


def create_sample_visualizations(model_info: Dict) -> Dict[str, str]:
    """Create sample visualizations with mock data for demonstration."""
    visualizer = ModelVisualizer(None, model_info)

    # Generate sample data
    np.random.seed(42)

    # Sample predictions for confusion matrix
    y_true = np.random.choice([0, 1], size=100, p=[0.3, 0.7])
    y_pred = np.random.choice([0, 1], size=100, p=[0.25, 0.75])
    y_proba = np.random.beta(2, 2, size=100)

    # Sample metrics
    metrics = {
        "Accuracy": 0.825,
        "Precision": 0.811,
        "Recall": 0.889,
        "F1-Score": 0.848,
        "Specificity": 0.742,
    }

    # Sample class distribution
    class_counts = {"Normal": 1583, "Pneumonia": 4273}

    # Sample training history
    history = {
        "accuracy": [0.6, 0.7, 0.75, 0.8, 0.825],
        "val_accuracy": [0.55, 0.65, 0.7, 0.77, 0.777],
        "loss": [0.8, 0.6, 0.5, 0.4, 0.35],
        "val_loss": [0.85, 0.7, 0.6, 0.5, 0.48],
    }

    # Sample predictions
    predictions = [
        {
            "confidence": np.random.uniform(0.5, 1.0),
            "predicted_class": "Normal" if np.random.random() > 0.7 else "Pneumonia",
        }
        for _ in range(50)
    ]

    visualizations = {}

    try:
        visualizations["confusion_matrix"] = visualizer.generate_confusion_matrix(
            y_true, y_pred
        )
        visualizations["roc_curve"] = visualizer.generate_roc_curve(y_true, y_proba)
        visualizations["performance_metrics"] = (
            visualizer.generate_performance_metrics_plot(metrics)
        )
        visualizations["class_distribution"] = (
            visualizer.generate_class_distribution_plot(class_counts)
        )
        visualizations["training_history"] = visualizer.generate_training_history_plot(
            history
        )
        visualizations["model_architecture"] = (
            visualizer.generate_model_architecture_plot(model_info)
        )
        visualizations["prediction_confidence"] = (
            visualizer.generate_prediction_confidence_plot(predictions)
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    return visualizations
