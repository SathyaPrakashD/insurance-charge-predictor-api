from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd



# Load the saved model
model = joblib.load('model.pkl')

# Define the app
app = FastAPI()

# Define input shape
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Define the prediction endpoint
@app.post('/predict')
def predict(data: InsuranceInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return {'predicted_charges': round(prediction, 2)}
