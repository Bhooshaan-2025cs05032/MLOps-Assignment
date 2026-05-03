from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# Import your prediction function
from api.model_load_predict import predict



# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk with confidence score",
    version="1.0.0"
)

# -----------------------------
# Prometheus Instrumentation and Metrics
# -----------------------------
prediction_counter = Counter('model_predictions_total', 'Total number of predictions made')

Instrumentator().instrument(app).expose(app)

# -----------------------------
# Request Schema
# -----------------------------
from typing import Optional

class PatientData(BaseModel):
    age: Optional[float] = None
    sex: Optional[int] = None
    cp: Optional[int] = None
    trestbps: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[int] = None
    restecg: Optional[int] = None
    thalach: Optional[float] = None
    exang: Optional[int] = None
    oldpeak: Optional[float] = None
    slope: Optional[int] = None
    ca: Optional[float] = None
    thal: Optional[int] = None


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_endpoint(data: PatientData):
    try:
        logger.info("Received prediction request")

        data_dict = data.dict()

        # Check missing fields
        missing_fields = [k for k, v in data_dict.items() if v is None]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing_fields}"
            )

        result = predict(data_dict)

        prediction_counter.inc()

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))