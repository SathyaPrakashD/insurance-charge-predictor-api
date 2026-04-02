from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Insurance Charge Predictor API",
    description=(
        "Predicts medical insurance charges from patient demographics. "
        "Powered by a GridSearchCV-tuned Random Forest pipeline (R\u00b2 = 0.8458). "
        "Visit /docs for the interactive Swagger UI."
    ),
    version="1.0.0",
)

# Load the serialised sklearn pipeline (preprocessor + tuned RandomForestRegressor)
model = joblib.load("model.pkl")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 19,
                    "sex": "female",
                    "bmi": 27.9,
                    "children": 0,
                    "smoker": "yes",
                    "region": "southwest",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_charges: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint — confirms the API is running."""
    return {"status": "ok", "message": "Insurance Charge Predictor API is live."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: InsuranceInput):
    """
    Predict medical insurance charges for a given patient.

    - Accepts a JSON body with patient demographics.
    - Runs the full sklearn pipeline (preprocessing + RandomForestRegressor).
    - Returns the predicted annual insurance charge in USD.
    """
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return PredictionResponse(predicted_charges=round(float(prediction), 2))
