# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model when the application starts
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Make sure it's in the same directory.")
    model = None

# Define the structure of the input data using Pydantic
# The names MUST match the columns in your training data
class TokenFeatures(BaseModel):
    liquidity_usd: float
    volume_h1: float
    price_change_h1: float
    top_10_hold_percentage: float
    is_honeypot: int
    sell_tax: float
    buy_tax: float
    creator_past_launches: int
    hype_score: int

# Define the prediction endpoint
@app.post("/predict")
async def predict(features: TokenFeatures):
    if model is None:
        return {"error": "Model not loaded."}
    
    # Convert the input data into a pandas DataFrame, as the model expects it
    data_df = pd.DataFrame([features.dict()])
    
    # Make a prediction (returns a probability score)
    # The [:, 1] gets the probability of the "success" (1) class
    prediction_proba = model.predict_proba(data_df)[:, 1]
    
    # Return the probability score as a JSON response
    return {"prediction_probability": float(prediction_proba[0])}

# A simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"status": "Model API is running."}
