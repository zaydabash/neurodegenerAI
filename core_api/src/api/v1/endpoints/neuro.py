"""
Neurodegenerative analysis endpoints.

Prediction routing:

* In demo mode (``NEURO_DEMO_MODE=true``, the default) requests are served by
  :class:`NeuroInferenceService`, which trains and caches real lightweight
  models (a LightGBM/XGBoost-or-sklearn ensemble, a 3D CNN, and a 1D EEG CNN)
  on synthetic data on first use - so every endpoint works out of the box.
* With demo mode disabled, requests are served by the production
  :class:`ModelPredictor` (trained artifacts on disk); if no artifact is
  present it transparently falls back to the demo service.

All free-text request metadata is PII-scrubbed before it is logged or
persisted.
"""

import uuid
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException

from core_api.src.api.v1.schemas import (
    EEGPredictionRequest,
    MRIPredictionRequest,
    PredictionResponse,
    TabularPredictionRequest,
)
from shared.lib.config import is_demo_mode
from shared.lib.io_utils import PIIScrubber
from shared.lib.logging import get_logger, log_api_request
from shared.lib.metrics import PerformanceTimer
from shared.lib.repository import save_neuro_prediction

logger = get_logger(__name__)
router = APIRouter()
_scrubber = PIIScrubber()


def _scrub(value):
    """Recursively PII-scrub strings inside dicts/lists/strings."""
    if isinstance(value, str):
        return _scrubber.scrub(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _production_predictor():
    """Return a cached production ModelPredictor, or None if unavailable."""
    from neurodegenerai.src.models.predict import ModelPredictor

    return ModelPredictor()


@router.post("/tabular", response_model=PredictionResponse)
async def predict_tabular(request: TabularPredictionRequest):
    """Predict using the tabular biomarker ensemble."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.inference import get_inference_service

        data_dict = request.dict()
        with PerformanceTimer("tabular_prediction"):
            if is_demo_mode():
                result = get_inference_service().predict_tabular(data_dict)
            else:
                try:
                    import pandas as pd

                    result = _production_predictor().predict_tabular(
                        pd.DataFrame([data_dict])
                    )
                except FileNotFoundError:
                    result = get_inference_service().predict_tabular(data_dict)

        response = PredictionResponse(
            prediction=result["prediction"],
            label="Dementia/MCI" if result["prediction"] == 1 else "Normal",
            probability=result["probability"],
            confidence=result["confidence"],
            model_name=result["model_name"],
            method_type=result["model_type"],
            explanation=_scrub(result.get("explanation")),
        )

        save_neuro_prediction(
            model_type="tabular",
            prediction=response.prediction,
            probability=response.probability,
            confidence=response.confidence,
            results_metadata={"model_name": response.model_name},
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/mri", response_model=PredictionResponse)
async def predict_mri(request: MRIPredictionRequest):
    """Predict using the 3D CNN MRI model."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.inference import get_inference_service

        volume = np.array(request.volume_data, dtype=np.float32)
        with PerformanceTimer("mri_prediction"):
            if is_demo_mode():
                result = get_inference_service().predict_mri(volume)
            else:
                try:
                    result = _production_predictor().predict_mri(volume)
                except FileNotFoundError:
                    result = get_inference_service().predict_mri(volume)

        response = PredictionResponse(
            prediction=result["prediction"],
            label="Dementia/MCI" if result["prediction"] == 1 else "Normal",
            probability=result["probability"],
            confidence=result["confidence"],
            model_name=result["model_name"],
            method_type=result["model_type"],
            heatmap_paths=result.get("heatmap_paths"),
        )

        save_neuro_prediction(
            model_type="mri",
            prediction=response.prediction,
            probability=response.probability,
            confidence=response.confidence,
            results_metadata={"model_name": response.model_name},
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/eeg", response_model=PredictionResponse)
async def predict_eeg(request: EEGPredictionRequest):
    """Predict using the EEG time-series model."""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        from neurodegenerai.src.models.eeg_real import get_real_eeg_predictor
        from neurodegenerai.src.models.inference import get_inference_service

        data = np.array(request.data, dtype=np.float32)
        with PerformanceTimer("eeg_prediction"):
            # Prefer the real Alzheimer's-vs-control band-power CNN trained on
            # OpenNeuro ds004504; fall back to the synthetic state decoder.
            real = get_real_eeg_predictor()
            if real.available:
                result = real.predict(data, request.sfreq)
            else:
                result = get_inference_service().predict_eeg(data)

        explanation = dict(result.get("explanation") or {})
        explanation.update(
            {"state": result["prediction"], "metadata": request.metadata}
        )
        response = PredictionResponse(
            prediction=result["prediction_idx"],
            label=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            model_name=result["model_name"],
            method_type=result["model_type"],
            explanation=_scrub(explanation),
        )

        save_neuro_prediction(
            model_type="eeg",
            prediction=response.prediction,
            probability=response.probability,
            confidence=response.confidence,
            results_metadata={
                "model_name": response.model_name,
                "state": result["prediction"],
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
        raise HTTPException(status_code=500, detail=str(e)) from e
