"""
Main training script for pneumonia detection model.
Complete pipeline from data loading to model training and evaluation.
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bwrp.data_loader import PneumoniaDataLoader, get_dataset_info
from bwrp.model import PneumoniaDetectionModel, create_and_train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset and show info",
    )
    parser.add_argument(
        "--model-dir", default="models", help="Directory to save models"
    )

    args = parser.parse_args()

    try:
        # Initialize data loader
        logger.info("Initializing pneumonia detection training pipeline...")
        data_loader = PneumoniaDataLoader()

        # Get dataset information
        logger.info("Getting dataset information...")
        dataset_info = get_dataset_info()

        print("\n" + "=" * 60)
        print("PNEUMONIA DETECTION AI - DATASET INFO")
        print("=" * 60)
        for key, value in dataset_info.items():
            if key == "structure":
                print(f"{key.upper()}:")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"  {value}")
            else:
                print(f"{key.upper()}: {value}")
        print("=" * 60)

        if args.download_only:
            logger.info("Download only mode - exiting after dataset info")
            return

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = data_loader.create_data_loaders()

        # Get dataset paths for the training function
        dataset_path = data_loader.dataset_path
        chest_xray_path = dataset_path / "chest_xray"
        if chest_xray_path.exists():
            base_path = chest_xray_path
        else:
            base_path = dataset_path

        train_dir = str(base_path / "train")
        val_dir = str(base_path / "val")
        test_dir = str(base_path / "test")

        # Create model directory
        model_dir = Path(args.model_dir)
        model_dir.mkdir(exist_ok=True)

        # Train model
        model, evaluation_results = create_and_train_model(
            train_dir,
            val_dir,
            test_dir,
            epochs=args.epochs,
        )

        # Print results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED - EVALUATION RESULTS")
        print("=" * 60)
        if isinstance(evaluation_results, dict) and evaluation_results:
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric.upper()}: {value:.4f}")
                else:
                    print(f"{metric.upper()}: {value}")
        else:
            print("Evaluation completed - results may be in model.metrics format")
        print("=" * 60)

        # Save model info
        model_info = {
            "epochs_trained": args.epochs,
            "evaluation_results": evaluation_results,
            "dataset_info": dataset_info,
        }

        info_path = model_dir / "model_info.json"
        import json

        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model training completed successfully!")
        logger.info(f"Model saved in: {model_dir}")
        logger.info(f"Best model: {model_dir / f'{model.model_name}_best.h5'}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
