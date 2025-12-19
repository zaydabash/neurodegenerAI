"""
FastAPI server for NeuroDegenerAI.
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.lib.config import ensure_directories, get_settings
from shared.lib.logging import get_logger, log_api_request, setup_logging
from shared.lib.metrics import get_metrics_collector, record_performance

from ..models.interpretability import ModelInterpretability
from ..models.predict import ModelEnsemble, ModelPredictor
from ..models.train_eeg import EEGPredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EEGPredictionRequest,
    ExplanationRequest,
    ExplanationResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetricsResponse,
    MRIPredictionRequest,
    PredictionResponse,
    TabularPredictionRequest,
)

# Setup logging
setup_logging(service_name="neurodegenerai_api")
logger = get_logger(__name__)

# Global variables for models
predictor: ModelPredictor | None = None
ensemble_predictor: ModelEnsemble | None = None
interpretability: ModelInterpretability | None = None
eeg_predictor: EEGPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""

    global predictor, ensemble_predictor, interpretability, eeg_predictor

    logger.info("Starting NeuroDegenerAI API server")

    # Ensure directories exist
    ensure_directories()

    try:
        # Initialize model predictor
        predictor = ModelPredictor()
        logger.info("Model predictor initialized")

        # Initialize ensemble predictor
        ensemble_predictor = ModelEnsemble()
        logger.info("Ensemble predictor initialized")

        # Initialize interpretability
        interpretability = ModelInterpretability(predictor.tabular_model, "tabular")
        logger.info("Interpretability module initialized")

        # Initialize EEG predictor
        eeg_model_path = "neurodegenerai/models/eeg_model.pth"
        eeg_predictor = EEGPredictor(model_path=eeg_model_path)
        logger.info("EEG predictor initialized")

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # Continue without models for health checks

    yield

    logger.info("Shutting down NeuroDegenerAI API server")


# Create FastAPI app
app = FastAPI(
    title="NeuroDegenerAI API",
    description="ML API for early neurodegenerative pattern detection",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""

    start_time = datetime.now()

    try:
        model_loaded = predictor is not None and predictor.tabular_model is not None

        response = HealthResponse(status="healthy", model_loaded=model_loaded)

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/health", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/tabular", response_model=PredictionResponse)
async def predict_tabular(request: TabularPredictionRequest):
    """Predict using tabular model."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert request to DataFrame
        data_dict = request.dict()
        df = pd.DataFrame([data_dict])

        # Make prediction
        with record_performance("tabular_prediction"):
            prediction_result = predictor.predict_tabular(df)

        # Create response
        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            model_type=prediction_result["model_type"],
            explanation=prediction_result.get("explanation"),
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request(
            "/predict/tabular", "POST", 200, duration, request_id=request_id
        )

        return response

    except Exception as e:
        logger.error(f"Tabular prediction failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request(
            "/predict/tabular", "POST", 500, duration, request_id=request_id
        )

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/mri", response_model=PredictionResponse)
async def predict_mri(request: MRIPredictionRequest):
    """Predict using MRI CNN model."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert volume data to numpy array
        volume = np.array(request.volume_data, dtype=np.float32)

        # Make prediction
        with record_performance("mri_prediction"):
            prediction_result = predictor.predict_mri(volume)

        # Create response
        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            model_type=prediction_result["model_type"],
            heatmap_paths=prediction_result.get("heatmap_paths"),
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/mri", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"MRI prediction failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/mri", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/eeg", response_model=PredictionResponse)
async def predict_eeg(request: EEGPredictionRequest):
    """Predict using EEG time-series model."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if eeg_predictor is None:
            raise HTTPException(status_code=503, detail="EEG model not loaded")

        # Convert data to numpy array
        data = np.array(request.data, dtype=np.float32)

        # Make prediction
        with record_performance("eeg_prediction"):
            prediction_result = eeg_predictor.predict(data)

        # Create response
        response = PredictionResponse(
            prediction=prediction_result["prediction_idx"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            model_type=prediction_result["model_type"],
            explanation={
                "state": prediction_result["prediction"],
                "metadata": request.metadata,
            },
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/eeg", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"EEG prediction failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/eeg", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(
    tabular_request: TabularPredictionRequest, mri_request: MRIPredictionRequest
):
    """Predict using ensemble of tabular and MRI models."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if ensemble_predictor is None:
            raise HTTPException(status_code=503, detail="Ensemble model not loaded")

        # Convert requests to appropriate formats
        tabular_data = pd.DataFrame([tabular_request.dict()])
        mri_volume = np.array(mri_request.volume_data, dtype=np.float32)

        # Make ensemble prediction
        with record_performance("ensemble_prediction"):
            prediction_result = ensemble_predictor.predict_ensemble(
                tabular_data, mri_volume
            )

        # Create response
        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name="ensemble",
            model_type=prediction_result["model_type"],
            explanation={
                "tabular_prediction": prediction_result["tabular_prediction"],
                "cnn_prediction": prediction_result["cnn_prediction"],
            },
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request(
            "/predict/ensemble", "POST", 200, duration, request_id=request_id
        )

        return response

    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request(
            "/predict/ensemble", "POST", 500, duration, request_id=request_id
        )

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        # Make batch predictions
        with record_performance("batch_prediction", {"batch_size": len(df)}):
            if request.model_type == "tabular":
                predictions = predictor.batch_predict_tabular(df)
            else:
                raise HTTPException(
                    status_code=400, detail="Only tabular batch predictions supported"
                )

        # Count successful/failed predictions
        successful = len([p for p in predictions if "error" not in p])
        failed = len(predictions) - successful

        # Create response objects
        response_predictions = []
        for pred in predictions:
            if "error" in pred:
                response_predictions.append(
                    PredictionResponse(
                        prediction=-1,
                        probability=0.0,
                        confidence=0.0,
                        model_name="error",
                        model_type="error",
                        error=pred["error"],
                    )
                )
            else:
                response_predictions.append(
                    PredictionResponse(
                        prediction=pred["prediction"],
                        probability=pred["probability"],
                        confidence=pred["confidence"],
                        model_name=pred["model_name"],
                        model_type=pred["model_type"],
                    )
                )

        response = BatchPredictionResponse(
            predictions=response_predictions,
            total_samples=len(df),
            successful_predictions=successful,
            failed_predictions=failed,
            model_name=predictor.model_name,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/batch", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/predict/batch", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get model performance metrics."""

    start_time = datetime.now()

    try:
        metrics_collector = get_metrics_collector()
        latest_metrics = metrics_collector.get_latest_metrics()

        if latest_metrics is None:
            raise HTTPException(status_code=404, detail="No metrics available")

        response = ModelMetricsResponse(
            accuracy=latest_metrics.accuracy,
            precision=latest_metrics.precision,
            recall=latest_metrics.recall,
            f1_score=latest_metrics.f1,
            roc_auc=latest_metrics.roc_auc,
            pr_auc=latest_metrics.pr_auc,
            confusion_matrix=latest_metrics.confusion_matrix,
            model_name=latest_metrics.model_name,
            model_type="tabular",
            data_hash=latest_metrics.data_hash,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/metrics", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/metrics", "GET", 500, duration)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""

    start_time = datetime.now()

    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_info = predictor.get_model_info()

        response = ModelInfoResponse(
            model_name=model_info["model_name"],
            model_type=model_info.get("model_type", "unknown"),
            model_class=model_info.get("model_class", "unknown"),
            version="0.1.0",
            num_features=model_info.get("num_features"),
            parameters=model_info,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/info", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/info", "GET", 500, duration)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Get feature importance scores."""

    start_time = datetime.now()

    try:
        if predictor is None or predictor.tabular_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Get feature importance from model
        if hasattr(predictor.tabular_model, "feature_importances_"):
            feature_importance = dict(
                zip(
                    predictor.preprocessor.feature_names,
                    predictor.tabular_model.feature_importances_,
                    strict=False,
                )
            )
        else:
            raise HTTPException(
                status_code=404, detail="Feature importance not available"
            )

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        top_features = [
            {"feature": feature, "importance": importance, "rank": i + 1}
            for i, (feature, importance) in enumerate(sorted_features)
        ]

        response = FeatureImportanceResponse(
            feature_importance=feature_importance,
            top_features=top_features,
            model_name=predictor.model_name,
        )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/features/importance", "GET", 200, duration)

        return response

    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/model/features/importance", "GET", 500, duration)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Explain model prediction."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        if interpretability is None:
            raise HTTPException(
                status_code=503, detail="Interpretability module not loaded"
            )

        # Convert data to appropriate format
        if request.model_type == "tabular":
            data_array = np.array(list(request.data.values())).reshape(1, -1)

            # Get explanation
            explanation = interpretability.explain_tabular_prediction(data_array)

            response = ExplanationResponse(
                explanation_method=request.explanation_method,
                explanation_data=explanation,
                feature_importance=explanation.get("feature_importance"),
                model_name=predictor.model_name if predictor else "unknown",
            )
        else:
            raise HTTPException(
                status_code=400, detail="Only tabular explanations supported"
            )

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/explain", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/explain", "POST", 500, duration, request_id=request_id)

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upload/mri")
async def upload_mri_file(file: UploadFile = File(...)):
    """Upload MRI file for prediction."""

    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        # Save uploaded file temporarily
        upload_dir = "./neurodegenerai/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = f"{upload_dir}/{request_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process file based on extension
        if file.filename.endswith(".nii") or file.filename.endswith(".nii.gz"):
            # Load NIfTI file
            import nibabel as nib

            img = nib.load(file_path)
            volume = img.get_fdata()

            # Make prediction
            if predictor is not None:
                prediction_result = predictor.predict_mri(volume)

                response = {
                    "file_id": request_id,
                    "filename": file.filename,
                    "volume_shape": volume.shape,
                    "prediction": prediction_result,
                }
            else:
                response = {
                    "file_id": request_id,
                    "filename": file.filename,
                    "volume_shape": volume.shape,
                    "error": "Model not loaded",
                }
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Clean up uploaded file
        os.remove(file_path)

        # Log API request
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/upload/mri", "POST", 200, duration, request_id=request_id)

        return response

    except Exception as e:
        logger.error(f"MRI upload failed: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("/upload/mri", "POST", 500, duration, request_id=request_id)

        # Clean up file if it exists
        try:
            if "file_path" in locals():
                os.remove(file_path)
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""

    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=settings.env == "dev")
