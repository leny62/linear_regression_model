from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
from pathlib import Path
import warnings

# Ensure numpy is imported first
try:
    import numpy as np
except ImportError:
    raise ImportError("numpy must be installed first")

import pandas as pd
import joblib
from typing import Optional, Dict

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path to find the model
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

app = FastAPI(
    title="African GDP Growth Predictor API",
    description="Predicts GDP growth rates for African countries using machine learning",
    version="1.0.0",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    country: str = Field(
        ...,
        description="Name of the African country",
        example="Kenya"
    )
    year: int = Field(
        ...,
        description="Year for prediction",
        ge=2000,  # Minimum year
        le=2030,  # Maximum year
        example=2025
    )

class PredictionOutput(BaseModel):
    country: str
    year: int
    predicted_growth: float
    confidence_interval: Dict[str, float]
    region: str
    prediction_metadata: Dict[str, float]

def load_model_safely():
    """Safely load the model with error handling"""
    try:
        model_path = os.path.join(parent_dir, 'linear_regression', 'best_gdp_growth_model.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"Loading model from: {model_path}")

        # Load with numpy pickle handler
        return model_data

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Parent directory: {parent_dir}")
        print(f"Looking for model at: {model_path}")
        raise

# Load the model and preprocessor
try:
    model_data = load_model_safely()
    MODEL = model_data['model']
    PREPROCESSOR = model_data['preprocessor']
    PERFORMANCE = model_data['performance']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    MODEL = None
    PREPROCESSOR = None
    PERFORMANCE = {'r2_score': 0, 'mse': 0, 'cv_score': 0}

@app.get("/")
def read_root():
    """Root endpoint providing API information and model performance metrics."""
    return {
        "message": "Welcome to the African GDP Growth Prediction API",
        "description": "This API predicts GDP growth rates for African countries",
        "usage": "POST /predict with country name and year",
        "model_performance": {
            "r2_score": float(PERFORMANCE['r2_score']),
            "mse": float(PERFORMANCE['mse']),
            "cv_score": float(PERFORMANCE['cv_score'])
        }
    }

@app.get("/health")
def health_check():
    """Check if the model is loaded and ready for predictions"""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_gdp_growth(input_data: PredictionInput):
    """
    Predict GDP growth rate for a specific African country and year.

    Parameters:
    - country: Name of the African country
    - year: Year for prediction (between 2000 and 2030)

    Returns:
    - Prediction details including confidence intervals and metadata
    """
    try:
        # Create input DataFrame
        input_df = pd.DataFrame({
            'Year': [input_data.year],
            'Country': [input_data.country],
            'Region': [PREPROCESSOR.region_mapping.get(input_data.country, 'Other')],
            'Decade': [(input_data.year // 10) * 10],
            'Post_2000': [int(input_data.year > 2000)]
        })

        # Preprocess input
        processed_data = PREPROCESSOR.transform(input_df)

        # Make prediction
        prediction = float(MODEL.predict(processed_data)[0])

        # Calculate confidence interval based on model MSE
        mse = PERFORMANCE['mse']
        ci_range = 1.96 * (mse ** 0.5)  # 95% confidence interval

        return PredictionOutput(
            country=input_data.country,
            year=input_data.year,
            predicted_growth=prediction,
            confidence_interval={
                "lower": prediction - ci_range,
                "upper": prediction + ci_range
            },
            region=PREPROCESSOR.region_mapping.get(input_data.country, 'Other'),
            prediction_metadata={
                "model_r2_score": float(PERFORMANCE['r2_score']),
                "model_mse": float(PERFORMANCE['mse']),
                "confidence_level": 0.95
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
