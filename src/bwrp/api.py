"""
FastAPI backend for pneumonia detection web service.
Provides REST API endpoints for AI model inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional

from .model import PneumoniaDetectionModel
from .data_loader import get_dataset_info
from .statistics import generate_model_statistics
from .model_visualizer import create_sample_visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BWRP Pneumonia Detection API",
    description="AI-powered pneumonia detection from chest X-ray images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance: Optional[PneumoniaDetectionModel] = None
MODEL_PATH = "models/pneumonia_detector_resnet18_v1_best.pth"


async def load_model():
    """Load the trained PyTorch model on startup."""
    global model_instance
    try:
        model_instance = PneumoniaDetectionModel()

        # Try to load the PyTorch model
        pytorch_model_path = "models/pneumonia_detector_resnet18_v1_best.pth"

        if Path(pytorch_model_path).exists():
            model_instance.load_model(pytorch_model_path)
            logger.info(
                f"PyTorch ResNet18 model loaded successfully from {pytorch_model_path}"
            )

            # Diagnostic: Check model parameters and architecture
            if model_instance.model:
                logger.info("=== PYTORCH MODEL DIAGNOSTIC INFO ===")
                total_params = sum(p.numel() for p in model_instance.model.parameters())
                trainable_params = sum(
                    p.numel()
                    for p in model_instance.model.parameters()
                    if p.requires_grad
                )
                logger.info(f"Total parameters: {total_params:,}")
                logger.info(f"Trainable parameters: {trainable_params:,}")

                # Test prediction with dummy data to check output range
                import torch
                import numpy as np

                model_instance.model.eval()
                dummy_input = torch.randn(1, 3, 224, 224).to(model_instance.device)
                with torch.no_grad():
                    test_pred = model_instance.model(dummy_input)
                    logger.info(f"Test prediction shape: {test_pred.shape}")
                    logger.info(
                        f"Test prediction value: {test_pred.cpu().numpy()[0][0]:.6f}"
                    )
                    logger.info(
                        f"Test prediction range: [{test_pred.min():.6f}, {test_pred.max():.6f}]"
                    )
                logger.info("=== END PYTORCH MODEL DIAGNOSTIC ===")

        elif Path(MODEL_PATH).exists():
            model_instance.load_model(MODEL_PATH)
            logger.info("PyTorch model loaded successfully")
        else:
            logger.warning(f"No trained PyTorch model found at {pytorch_model_path}")
            logger.info("Building new ResNet18 model architecture (not trained)")
            model_instance.build_model()
            logger.warning("⚠️ Model is not trained! All predictions will be random.")
            logger.info(
                "To train the model, run: rye run python -m bwrp.cli train --epochs 10"
            )

    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        model_instance = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting BWRP Pneumonia Detection API...")
    await load_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "BWRP Pneumonia Detection API",
        "version": "1.0.0",
        "description": "AI-powered pneumonia detection from chest X-ray images",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "dataset_info": "/dataset/info",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model_instance and model_instance.model else "not_loaded"
    return {"status": "healthy", "model_status": model_status, "api_version": "1.0.0"}


@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predict pneumonia from uploaded chest X-ray image.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results with confidence scores
    """
    if not model_instance or not model_instance.model:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(model_instance.input_shape[:2])

        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0

        # Diagnostic: Log image preprocessing details
        logger.info(f"=== PREDICTION DIAGNOSTIC for {file.filename} ===")
        logger.info(f"Original image mode: {image.mode}")
        logger.info(f"Resized image size: {image.size}")
        logger.info(f"Image array shape: {image_array.shape}")
        logger.info(f"Image array dtype: {image_array.dtype}")
        logger.info(
            f"Image array range: [{image_array.min():.6f}, {image_array.max():.6f}]"
        )
        logger.info(f"Image array mean: {image_array.mean():.6f}")

        # Make prediction
        prediction_result = model_instance.predict_image(image_array)

        # Diagnostic: Log raw prediction details (PyTorch)
        import torch

        model_instance.model.eval()
        image_tensor = (
            model_instance.transform(image).unsqueeze(0).to(model_instance.device)
        )
        with torch.no_grad():
            raw_pred = model_instance.model(image_tensor)
            logger.info(f"Raw PyTorch model output: {raw_pred.cpu().numpy()[0][0]:.6f}")
        logger.info(f"Prediction result: {prediction_result}")
        logger.info("=== END PREDICTION DIAGNOSTIC ===")

        # Check if model is trained (PyTorch models)
        pytorch_model_exists = Path(
            "models/pneumonia_detector_resnet18_v1_best.pth"
        ).exists()
        is_trained = pytorch_model_exists

        # Add metadata
        result = {
            "success": True,
            "filename": file.filename,
            "file_size": len(image_data),
            "image_shape": image_array.shape,
            "is_trained_model": is_trained,
            **prediction_result,
        }

        if not is_trained:
            result["warning"] = "⚠️ Model is not trained! Predictions are random."
            result["training_instructions"] = (
                "Run 'rye run python -m bwrp.cli train --epochs 10' to train the model"
            )

        logger.info(
            f"Prediction completed for {file.filename}: {prediction_result['predicted_class']}"
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if not model_instance:
        return {"status": "Model not loaded"}

    # Check if model seems trained by looking for saved PyTorch models
    pytorch_model_exists = Path(
        "models/pneumonia_detector_resnet18_v1_best.pth"
    ).exists()
    is_trained = pytorch_model_exists

    model_info = {
        "model_name": model_instance.model_name,
        "input_shape": model_instance.input_shape,
        "class_names": model_instance.class_names,
        "model_loaded": model_instance.model is not None,
        "is_trained": is_trained,
        "training_status": (
            "Trained model loaded"
            if is_trained
            else "⚠️ Untrained model - predictions will be random"
        ),
        "model_summary": (
            model_instance.get_model_summary()
            if model_instance.model
            else "Not available"
        ),
    }

    if not is_trained:
        model_info["training_instructions"] = (
            "Run 'rye run python -m bwrp.cli train --epochs 10' to train the model"
        )

    return model_info


@app.get("/dataset/info")
async def get_dataset_information():
    """Get information about the pneumonia dataset."""
    try:
        dataset_info = get_dataset_info()
        return {"success": True, "dataset_info": dataset_info}
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        return {
            "success": False,
            "error": str(e),
            "dataset_info": {
                "dataset_name": "paultimothymooney/chest-xray-pneumonia",
                "description": "Chest X-Ray Images (Pneumonia) dataset",
                "classes": ["NORMAL", "PNEUMONIA"],
                "note": "Detailed info unavailable - run dataset download first",
            },
        }


@app.get("/model/statistics")
async def get_model_statistics():
    """Get comprehensive model statistics and performance metrics."""
    try:
        stats = generate_model_statistics()
        return {"success": True, "statistics": stats}
    except Exception as e:
        logger.error(f"Failed to get model statistics: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Model statistics unavailable - ensure model is trained",
        }


@app.post("/model/train")
async def trigger_training():
    """
    Trigger model training (for demonstration purposes).
    In production, this would be a separate service.
    """
    return {
        "message": "Training endpoint not implemented in this demo",
        "note": "Model training should be done offline for production systems",
        "suggestion": "Use the training script in the src/bwrp/ directory",
    }


@app.get("/predict/demo")
async def prediction_demo():
    """
    Demo endpoint showing expected prediction response format.
    """
    return {
        "demo_response": {
            "success": True,
            "filename": "example_xray.jpg",
            "file_size": 156784,
            "image_shape": [224, 224, 3],
            "predicted_class": "PNEUMONIA",
            "probability": 0.8765,
            "confidence": 0.8765,
            "pneumonia_probability": 0.8765,
            "normal_probability": 0.1235,
        },
        "note": "This is a demo response showing the expected format",
    }


@app.get("/model/visualizations")
async def get_model_visualizations():
    """Get model visualization images including confusion matrix, ROC curve, etc."""
    try:
        # Get model info for visualizations
        model_info = {}
        if model_instance and model_instance.model:
            # PyTorch model parameter counting
            total_params = sum(p.numel() for p in model_instance.model.parameters())
            trainable_params = sum(
                p.numel() for p in model_instance.model.parameters() if p.requires_grad
            )

            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "layer_count": len(list(model_instance.model.modules())),
                "layer_types": {},
                "model_size_mb": round(
                    total_params * 4 / (1024 * 1024), 2
                ),  # 4 bytes per float32
            }

            # Count layer types for PyTorch
            for module in model_instance.model.modules():
                module_type = type(module).__name__
                if module_type != "ResNet":  # Skip the main model container
                    model_info["layer_types"][module_type] = (
                        model_info["layer_types"].get(module_type, 0) + 1
                    )

        # Generate visualizations
        visualizations = create_sample_visualizations(model_info)

        return {
            "success": True,
            "visualizations": visualizations,
            "generated_at": "2025-08-06T20:25:00Z",
            "model_info": model_info,
        }
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to generate model visualizations",
        }


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Run the FastAPI server.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run("bwrp.api:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run_server()
