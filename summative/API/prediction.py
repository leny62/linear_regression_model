from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
from pathlib import Path
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

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

# CategoryEncoder class needed for loading the model
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='Country', region_col='Region'):
        self.country_col = country_col
        self.region_col = region_col
        # These will be populated during fit or when the model is loaded
        self.countries = []
        self.regions = []

    def fit(self, X, y=None):
        self.countries = sorted(X[self.country_col].unique().tolist())
        self.regions = sorted(X[self.region_col].unique().tolist())
        return self

    def transform(self, X):
        X_copy = X.copy()
        # One-hot encode countries and regions
        country_dummies = pd.get_dummies(X_copy[self.country_col], prefix='country')
        region_dummies = pd.get_dummies(X_copy[self.region_col], prefix='region')

        # Add missing columns with zeros
        for country in self.countries:
            if f'country_{country}' not in country_dummies.columns:
                country_dummies[f'country_{country}'] = 0
        for region in self.regions:
            if f'region_{region}' not in region_dummies.columns:
                region_dummies[f'region_{region}'] = 0

        # Drop original columns and concatenate
        X_copy = X_copy.drop([self.country_col, self.region_col], axis=1)
        result = pd.concat([X_copy, country_dummies, region_dummies], axis=1)

        # Ensure feature names are preserved
        if hasattr(self, 'feature_names_'):
            # Reorder columns to match the expected feature names
            result = result.reindex(columns=self.feature_names_, fill_value=0)

        return result

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
        # Construct absolute path to the model file
        model_path = os.path.join(parent_dir, 'linear_regression', 'best_gdp_growth_model.pkl')

        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Parent directory: {parent_dir}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"Loading model from: {model_path}")

        # Load the model data (which should be a dictionary containing all components)
        model_data = joblib.load(model_path)
        print(f"Model data loaded successfully. Type: {type(model_data)}")

        if not isinstance(model_data, dict):
            raise ValueError("Model file does not contain expected dictionary structure")

        # Extract the required components from the model data
        result = {
            'model': model_data.get('model'),
            'preprocessor': model_data.get('preprocessor'),
            'performance': model_data.get('performance', {})
        }

        # Validate that required components are present
        if result['model'] is None:
            raise ValueError("Model component not found in the loaded data")
        if result['preprocessor'] is None:
            raise ValueError("Preprocessor component not found in the loaded data")

        print(f"Model components extracted successfully: {list(model_data.keys())}")
        return result

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Parent directory: {parent_dir}")
        print(f"Looking for model at: {model_path}")
        raise

MODEL = None
PREPROCESSOR = None
PERFORMANCE: Dict[str, float] = {}

@app.on_event("startup")
def load_model_event():
    """Load model and preprocessor at startup."""
    global MODEL, PREPROCESSOR, PERFORMANCE
    model_data = load_model_safely()
    MODEL = model_data['model']
    PREPROCESSOR = model_data['preprocessor']
    PERFORMANCE = model_data['performance']
    print("Model loaded successfully at startup")

@app.get("/")
def read_root():
    """Root endpoint providing API information and model performance metrics."""
    return {
        "message": "Welcome to the African GDP Growth Prediction API",
        "model_status": "loaded" if MODEL is not None else "not loaded",
        "model_performance": PERFORMANCE,
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """Check if the model and preprocessor are loaded and ready for predictions."""
    status = "healthy" if MODEL is not None and PREPROCESSOR is not None else "unhealthy"
    return {
        "status": status,
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_gdp(input_data: PredictionInput):
    """
    Predict GDP growth for a specific African country and year.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Check /health endpoint for status."
        )

    # Define region mapping
    region_mapping = {
        'Algeria': 'North Africa', 'Benin': 'West Africa', 'Botswana': 'Southern Africa',
        'Burkina Faso': 'West Africa', 'Burundi': 'East Africa', 'Cameroon': 'Central Africa',
        'Central African Republic': 'Central Africa', 'Chad': 'Central Africa',
        'Eswatini': 'Southern Africa', 'Ethiopia': 'East Africa', 'Gabon': 'Central Africa',
        'Ghana': 'West Africa', 'Kenya': 'East Africa', 'Lesotho': 'Southern Africa',
        'Liberia': 'West Africa', 'Libya': 'North Africa', 'Madagascar': 'East Africa',
        'Mauritius': 'East Africa', 'Morocco': 'North Africa', 'Niger': 'West Africa',
        'Nigeria': 'West Africa', 'Rwanda': 'East Africa', 'Senegal': 'West Africa',
        'Seychelles': 'East Africa', 'Sierra Leone': 'West Africa', 'Somalia': 'East Africa',
        'South Africa': 'Southern Africa', 'Sudan': 'North Africa', 'Tanzania': 'East Africa',
        'Togo': 'West Africa', 'Uganda': 'East Africa', 'Zambia': 'Southern Africa',
        'Zimbabwe': 'Southern Africa'
    }

    # Validate country
    if input_data.country not in region_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid country. Must be one of: {', '.join(sorted(region_mapping.keys()))}"
        )

    try:
        # Prepare input data
        input_df = pd.DataFrame({
            'Year': [input_data.year],
            'Country': [input_data.country],
            'Region': [region_mapping[input_data.country]],
            'Decade': [(input_data.year // 10) * 10],
            'Post_2000': [int(input_data.year > 2000)]
        })

        # Preprocess input
        input_processed = PREPROCESSOR.transform(input_df)

        # Make prediction
        prediction = float(MODEL.predict(input_processed)[0])

        # Calculate simple confidence interval based on model MSE
        mse = PERFORMANCE['mse']
        std_dev = np.sqrt(mse)
        confidence_interval = {
            "lower_bound": prediction - 1.96 * std_dev,
            "upper_bound": prediction + 1.96 * std_dev
        }

        return PredictionOutput(
            country=input_data.country,
            year=input_data.year,
            predicted_growth=prediction,
            confidence_interval=confidence_interval,
            region=region_mapping[input_data.country],
            prediction_metadata={
                "r2_score": PERFORMANCE['r2_score'],
                "mse": PERFORMANCE['mse'],
                "cv_score": PERFORMANCE['cv_score']
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
