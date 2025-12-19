"""
Neurodegenerative analysis endpoints.
"""

import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from core_api.src.api.v1.schemas import (
    EEGPredictionRequest,
    MRIPredictionRequest,
    PredictionResponse,
    TabularPredictionRequest,
)
from shared.lib.logging import get_logger, log_api_request
from shared.lib.metrics import record_performance

# We will need to figure out how to handle the global predictors in a consolidated way.
# For now, we'll assume they are available via some dependency or global state.
# (This will be hardened in Phase 2/4).

logger = get_logger(__name__)
router = APIRouter()

# Mock predictors (to be replaced with real initialization in main.py)
predictor = None
eeg_predictor = None


@router.post("/tabular", response_model=PredictionResponse)
async def predict_tabular(request: TabularPredictionRequest):
    """Predict using tabular model."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.predict import ModelPredictor

        global predictor
        if predictor is None:
            predictor = ModelPredictor()  # Lazy init for now

        data_dict = request.dict()
        df = pd.DataFrame([data_dict])

        with record_performance("tabular_prediction"):
            prediction_result = predictor.predict_tabular(df)

        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            label="Dementia/MCI" if prediction_result["prediction"] == 1 else "Normal",
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            method_type=prediction_result["model_type"],
            explanation=prediction_result.get("explanation"),
        )

        log_api_request(
            "/v1/neuro/tabular",
            "POST",
            200,
            (datetime.now() - start_time).total_seconds(),
            request_id=request_id,
        )
        return response
    except Exception as e:
        logger.error(f"Neuro tabular prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mri", response_model=PredictionResponse)
async def predict_mri(request: MRIPredictionRequest):
    """Predict using MRI CNN model."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.predict import ModelPredictor

        global predictor
        if predictor is None:
            predictor = ModelPredictor()

        volume = np.array(request.volume_data, dtype=np.float32)

        with record_performance("mri_prediction"):
            prediction_result = predictor.predict_mri(volume)

        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            label="Dementia/MCI" if prediction_result["prediction"] == 1 else "Normal",
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            method_type=prediction_result["model_type"],
            heatmap_paths=prediction_result.get("heatmap_paths"),
        )

        log_api_request(
            "/v1/neuro/mri",
            "POST",
            200,
            (datetime.now() - start_time).total_seconds(),
            request_id=request_id,
        )
        return response
    except Exception as e:
        logger.error(f"Neuro MRI prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eeg", response_model=PredictionResponse)
async def predict_eeg(request: EEGPredictionRequest):
    """Predict using EEG time-series model."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.train_eeg import EEGPredictor

        global eeg_predictor
        if eeg_predictor is None:
            eeg_predictor = EEGPredictor(
                model_path="neurodegenerai/models/eeg_model.pth"
            )

        data = np.array(request.data, dtype=np.float32)

        with record_performance("eeg_prediction"):
            prediction_result = eeg_predictor.predict(data)

        response = PredictionResponse(
            prediction=prediction_result["prediction_idx"],
            label=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            model_name=prediction_result["model_name"],
            method_type=prediction_result["model_type"],
            explanation={
                "state": prediction_result["prediction"],
                "metadata": request.metadata,
            },
        )

        log_api_request(
            "/v1/neuro/eeg",
            "POST",
            200,
            (datetime.now() - start_time).total_seconds(),
            request_id=request_id,
        )
        return response
    except Exception as e:
        logger.error(f"Neuro EEG prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
