"""
model_loader.py
---------------
Importable module for loading the trained heart-disease model and running
predictions.

Typical usage
-------------
    from model_loader import predict

    result = predict({"age": 40, "sex": 0, "cp": 1, ...})
    # {"prediction": 0, "confidence": 0.8252}

If you need explicit control over the model instance:

    from model_loader import load_model, predict

    model = load_model()
    result = predict({"age": 40, ...}, model=model)
"""

import logging
import joblib
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-relative paths (resolved from this file's location)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent

MODEL_NAME = "heart_disease_pred_model"
MODEL_STAGE = "None"  # no promotion stage used yet
DEFAULT_MLRUNS_PATH = _PROJECT_ROOT / "mlruns"
DEFAULT_LOCAL_MODEL_PATH = _PROJECT_ROOT / "models" / f"{MODEL_NAME}.pkl"

# Module-level lazy cache — populated on first call to get_model()
_model_singleton = None


# ---------------------------------------------------------------------------
# Loading strategies
# ---------------------------------------------------------------------------

def load_model_from_mlflow(
    mlruns_path: Path = DEFAULT_MLRUNS_PATH,
    model_name: str = MODEL_NAME,
    model_stage: str = MODEL_STAGE,
):
    """Load the latest version of *model_name* from the local MLflow registry.

    Returns the sklearn Pipeline on success, or None on any failure.
    """
    try:
        tracking_uri_local = f"file:///{mlruns_path.resolve().as_posix()}"
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI",tracking_uri_local)
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Trying MLflow registry at '%s'...", tracking_uri)

        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow registry (uri='%s').", model_uri)
        return model

    except Exception as exc:
        logger.warning("MLflow load failed: %s", exc)
        return None


def load_model_from_local(local_model_path: Path = DEFAULT_LOCAL_MODEL_PATH):
    """Load the model from a local .pkl file.

    Raises FileNotFoundError if the file does not exist.
    """
    if not local_model_path.exists():
        raise FileNotFoundError(
            f"Local model file not found: '{local_model_path.resolve()}'"
        )
    logger.info("Loading model from local file '%s'...", local_model_path.resolve())
    model = joblib.load(local_model_path)
    logger.info("Model loaded from local file.")
    return model


def load_model(
    mlruns_path: Path = DEFAULT_MLRUNS_PATH,
    local_model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
    model_name: str = MODEL_NAME,
    model_stage: str = MODEL_STAGE,
):
    """Load the model, preferring the MLflow registry with a local fallback.

    Parameters
    ----------
    mlruns_path:       Directory that contains the MLflow tracking store.
    local_model_path:  Path to the fallback .pkl file.
    model_name:        Registered model name in MLflow.
    model_stage:       MLflow model stage (e.g. "None", "Staging", "Production").

    Returns
    -------
    Trained sklearn Pipeline.
    """
    model = load_model_from_mlflow(mlruns_path, model_name, model_stage)
    if model is not None:
        return model

    logger.info("Falling back to local model file...")
    return load_model_from_local(local_model_path)


# ---------------------------------------------------------------------------
# Lazy singleton accessor
# ---------------------------------------------------------------------------

def get_model():
    """Return the cached model, loading it on the first call.

    Uses module-level defaults for paths and model name. Call load_model()
    directly if you need custom paths.
    """
    global _model_singleton
    if _model_singleton is None:
        logger.info("Model not yet loaded — initialising singleton...")
        _model_singleton = load_model()
    return _model_singleton


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(data: dict, model=None) -> dict:
    """Run a heart-disease prediction for a single patient record.

    Parameters
    ----------
    data:  Dictionary of feature values matching the training schema::

               {
                   "age": int, "sex": int, "cp": int, "trestbps": int,
                   "chol": int, "fbs": int, "restecg": int, "thalach": int,
                   "exang": int, "oldpeak": float, "slope": int,
                   "ca": int, "thal": int
               }

    model: Optional pre-loaded sklearn Pipeline. When omitted the module-level
           lazy singleton is used (loaded on first call).

    Returns
    -------
    dict with keys:

    * ``prediction``  – 0 (no disease) or 1 (disease present)
    * ``confidence``  – model's probability for the predicted class (rounded to
                        4 decimal places), or None if predict_proba is unavailable
    """
    if model is None:
        model = get_model()

    try:
        df = pd.DataFrame([data])
        pred = int(model.predict(df)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            confidence = float(round(probs[pred], 4))

        result = {"prediction": pred, "confidence": confidence}
        logger.info("Prediction result: %s", result)
        return result

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Quick smoke-test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    healthy_sample = {
        "age": 40, "sex": 0, "cp": 1, "trestbps": 110, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
    }
    disease_sample = {
        "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300,
        "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1,
        "oldpeak": 4.0, "slope": 0, "ca": 3, "thal": 3
    }

    print("Healthy sample →", predict(healthy_sample))
    print("Disease sample →", predict(disease_sample))
