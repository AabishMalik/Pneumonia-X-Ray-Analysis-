"""
Command-line interface for BWRP pneumonia detection system.
Provides easy access to training, prediction, and dataset management.
"""

import click
import sys
from pathlib import Path
import json
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from bwrp.data_loader import PneumoniaDataLoader, get_dataset_info
from bwrp.model import PneumoniaDetectionModel
from bwrp.api import run_server


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """BWRP Pneumonia Detection AI System"""
    pass


@cli.command()
@click.option("--download-only", is_flag=True, help="Only download and explore dataset")
def dataset(download_only):
    """Download and explore the pneumonia dataset."""
    click.echo("🔍 BWRP Dataset Manager")
    click.echo("=" * 50)

    try:
        # Get dataset info
        with click.progressbar(length=100, label="Loading dataset info") as bar:
            dataset_info = get_dataset_info()
            bar.update(50)

            if not download_only:
                data_loader = PneumoniaDataLoader()
                data_loader.download_dataset()
            bar.update(50)

        # Display info
        click.echo("\n📊 Dataset Information:")
        for key, value in dataset_info.items():
            if key == "structure" and isinstance(value, dict):
                click.echo(f"  {key}: ")
                for subkey, subvalue in value.items():
                    click.echo(f"    {subkey}: {subvalue}")
            else:
                click.echo(f"  {key}: {value}")

        click.echo("\n✅ Dataset ready!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option("--epochs", default=5, help="Number of training epochs")
@click.option("--model-dir", default="models", help="Model save directory")
def train(epochs, model_dir):
    """Train the pneumonia detection model."""
    click.echo("🚀 BWRP Model Training")
    click.echo("=" * 50)

    try:
        from bwrp.train import main as train_main
        import sys

        # Set up arguments for training script
        sys.argv = [
            "train",
            "--epochs",
            str(epochs),
            "--model-dir",
            model_dir,
        ]

        train_main()
        click.echo("✅ Training completed successfully!")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--model-path",
    default="models/pneumonia_detector_v1_best.h5",
    help="Path to trained model",
)
def predict(image_path, model_path):
    """Predict pneumonia from chest X-ray image."""
    click.echo("🔬 BWRP Pneumonia Prediction")
    click.echo("=" * 50)

    try:
        # Load model
        click.echo("Loading model...")
        model = PneumoniaDetectionModel()

        if Path(model_path).exists():
            model.load_model(model_path)
        else:
            click.echo(
                "⚠️  Pre-trained model not found. Using untrained model for demo."
            )
            model.build_model()

        # Load and preprocess image
        click.echo(f"Processing image: {image_path}")
        from PIL import Image
        import numpy as np

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0

        # Make prediction
        result = model.predict_image(image_array)

        # Display results
        click.echo("\n📋 Prediction Results:")
        click.echo(f"  Predicted Class: {result['predicted_class']}")
        click.echo(f"  Confidence: {result['confidence']:.2%}")
        click.echo(f"  Pneumonia Probability: {result['pneumonia_probability']:.2%}")
        click.echo(f"  Normal Probability: {result['normal_probability']:.2%}")

        # Color coding for results
        if result["predicted_class"] == "PNEUMONIA":
            click.echo("⚠️  PNEUMONIA detected - Medical consultation recommended")
        else:
            click.echo("✅ NORMAL chest X-ray")

    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}", err=True)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host, port, reload):
    """Start the FastAPI web server."""
    click.echo("🌐 BWRP Web Server")
    click.echo("=" * 50)
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo("API Documentation: http://{host}:{port}/docs")
    click.echo("Press Ctrl+C to stop")

    try:
        run_server(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")


@cli.command()
def info():
    """Show system information."""
    click.echo("ℹ️  BWRP System Information")
    click.echo("=" * 50)

    # Project info
    click.echo("📦 Project:")
    click.echo("  Name: BWRP Pneumonia Detection AI")
    click.echo("  Version: 1.0.0")
    click.echo("  Description: AI-powered pneumonia detection from chest X-rays")
    click.echo("  SDG Goal: 3 - Good Health and Well-being")

    # Technical stack
    click.echo("\n🛠️  Tech Stack:")
    click.echo("  Backend: Python + FastAPI")
    click.echo("  Frontend: React + TypeScript")
    click.echo("  AI/ML: PyTorch + TorchVision")
    click.echo("  Dataset: KaggleHub")
    click.echo("  Project Management: Rye")

    # File structure
    click.echo("\n📁 Project Structure:")
    project_root = Path(__file__).parent.parent.parent

    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return

        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items):
            if item.name.startswith(".") and item.name not in [
                ".gitignore",
                ".python-version",
            ]:
                continue

            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            click.echo(f"  {prefix}{current_prefix}{item.name}")

            if item.is_dir() and not item.name.startswith("."):
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix, max_depth, current_depth + 1)

    show_tree(project_root)


@cli.command()
def demo():
    """Run a complete demo of the system."""
    click.echo("🎯 BWRP Complete Demo")
    click.echo("=" * 50)

    click.echo("This demo will:")
    click.echo("1. Download dataset information")
    click.echo("2. Show model architecture")
    click.echo("3. Start web server")
    click.echo()

    if click.confirm("Continue with demo?"):
        # Dataset info
        click.echo("\n1️⃣  Getting dataset info...")
        try:
            dataset_info = get_dataset_info()
            click.echo(f"   Dataset: {dataset_info['dataset_name']}")
            click.echo(f"   Classes: {dataset_info['classes']}")
        except Exception as e:
            click.echo(f"   ⚠️  {e}")

        # Model info
        click.echo("\n2️⃣  Loading model architecture...")
        try:
            model = PneumoniaDetectionModel()
            model.build_model()
            click.echo("   ✅ Model architecture loaded")
        except Exception as e:
            click.echo(f"   ⚠️  {e}")

        # Server
        click.echo("\n3️⃣  Starting web server...")
        click.echo("   Press Ctrl+C to stop the demo")
        try:
            run_server(host="127.0.0.1", port=8000, reload=False)
        except KeyboardInterrupt:
            click.echo("\n✅ Demo completed!")


if __name__ == "__main__":
    cli()
