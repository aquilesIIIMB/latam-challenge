import logging
import os
from typing import Dict, List

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_400_BAD_REQUEST

from challenge.model import DelayModel

# ----------------------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Import/Instantiate the Model
# ----------------------------------------------------------------------
model = DelayModel()

# ----------------------------------------------------------------------
# FastAPI App Setup
# ----------------------------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    description=(
        "This API uses an XGBoost-based classifier to predict whether flights "
        "will be delayed based on provided flight information."
    ),
    version="1.0.0",
)


# Add this exception handler to your code:
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Custom handler to transform Pydantic's default 422 errors into 400.
    """
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


# ----------------------------------------------------------------------
# Pydantic Models (Request/Response)
# ----------------------------------------------------------------------
class Flight(BaseModel):
    """
    Schema representing a single flight's features needed for prediction.
    """

    OPERA: str = Field(..., description="Airline operator name")
    TIPOVUELO: str = Field(
        ..., description="Flight type (I = International, N = National)"
    )
    MES: int = Field(..., ge=1, le=12, description="Month of the year (1-12)")

    @validator("MES")
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("Month must be between 1 and 12.")
        return v

    @validator("TIPOVUELO")
    def validate_flight_type(cls, v):
        if v not in ["I", "N"]:
            raise ValueError(
                "Flight type must be either 'I' (International) or 'N' (National)."
            )
        return v


class PredictRequest(BaseModel):
    """
    Schema for the prediction request body, which accepts a list of flights.
    """

    flights: List[Flight] = Field(..., description="List of flights to predict")


class PredictResponse(BaseModel):
    """
    Schema for the prediction response body.
    """

    predict: List[int] = Field(
        ..., description="List of delay predictions (1=Delayed, 0=On Time)"
    )


# ----------------------------------------------------------------------
# Startup Event (Load or Train Model)
# ----------------------------------------------------------------------
@app.on_event("startup")
def load_or_train_model():
    """
    Runs on API startup.
    Loads a pre-trained model if available. Otherwise, trains it on sample data.
    """
    try:
        if os.path.exists(model.JSON_MODEL_PATH):
            model.load_booster_from_json(model.JSON_MODEL_PATH)
            logger.info(f"Model loaded successfully from {model.JSON_MODEL_PATH}.")
        else:
            logger.warning(
                "No pre-trained model found. Attempting to train from sample data."
            )

            # Training from local CSV (adjust path to suit your environment)
            sample_data_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "data.csv",
            )

            if not os.path.exists(sample_data_path):
                raise FileNotFoundError(
                    f"Could not find training data at {sample_data_path}. "
                    "Provide a valid dataset to train the model."
                )

            df = pd.read_csv(sample_data_path)
            # Preprocess
            X, y = model.preprocess(df, target_column="delay")
            # Train
            model.fit(X, y)
            logger.info("Model was trained and saved on API startup.")
    except Exception as e:
        logger.error(f"Error during startup model loading/training: {str(e)}")
        raise RuntimeError("Failed to load or train the model.") from e


# ----------------------------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------------------------
@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """
    Health check endpoint to ensure the API is running.

    Returns:
        dict: {"status": "OK"}
    """
    return {"status": "OK"}


# ----------------------------------------------------------------------
# Prediction Endpoint
# ----------------------------------------------------------------------
@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(request: PredictRequest = Body(...)) -> PredictResponse:
    """
    Predict flight delays using the loaded or trained model.

    Args:
        request (PredictRequest): A list of flights as input.

    Returns:
        PredictResponse: A list of predictions (1=Delay, 0=No Delay).
    """
    try:
        # Convert request to DataFrame
        logger.info(f"Received prediction request for {len(request.flights)} flights.")
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)

        # Preprocess data
        features = model.preprocess(df)

        # Predict
        predictions = model.predict(features)

        logger.info(f"Predicted results for {len(predictions)} flights successfully.")
        return PredictResponse(predict=predictions)

    except HTTPException:
        # Pass through any HTTPException
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
