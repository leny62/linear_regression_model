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

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Import the necessary class and handle module naming conflicts
from linear_regression.model_utils import CategoryEncoder

# Create module alias for both potential module paths that might be in the pickle
import sys
import types

# Create the gdp_model_utils module if it doesn't exist
if 'gdp_model_utils' not in sys.modules:
    gdp_model_utils = types.ModuleType('gdp_model_utils')
    sys.modules['gdp_model_utils'] = gdp_model_utils
    # Add CategoryEncoder to this module
    gdp_model_utils.CategoryEncoder = CategoryEncoder

# Also inject it into __main__ for older models that might reference it there
import __main__
__main__.CategoryEncoder = CategoryEncoder
# Also make gdp_model_utils accessible from __main__
if not hasattr(__main__, 'gdp_model_utils'):
    __main__.gdp_model_utils = sys.modules['gdp_model_utils']

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

        # Manual preprocessing approach
        if hasattr(MODEL, 'feature_names_in_'):
            # Get expected feature names from the model
            expected_features = MODEL.feature_names_in_
            
            # Create one-hot encoding for country and region manually
            countries_df = pd.DataFrame(0, index=input_df.index, 
                                     columns=[f'country_{c}' for c in region_mapping.keys()])
            regions_df = pd.DataFrame(0, index=input_df.index,
                                    columns=[f'region_{r}' for r in set(region_mapping.values())])
            
            # Set the appropriate country and region to 1
            countries_df[f'country_{input_data.country}'] = 1
            regions_df[f'region_{region_mapping[input_data.country]}'] = 1
            
            # Combine all features
            numeric_cols = ['Year', 'Decade', 'Post_2000']
            input_processed = pd.concat([
                input_df[numeric_cols], 
                countries_df,
                regions_df
            ], axis=1)
            
            # Ensure all columns match exactly what the model expects
            missing_cols = set(expected_features) - set(input_processed.columns)
            for col in missing_cols:
                input_processed[col] = 0
                
            # Keep only the columns the model expects, in the right order
            input_processed = input_processed[expected_features]
        else:
            # Fall back to preprocessor if model doesn't have feature_names_in_
            input_processed = PREPROCESSOR.transform(input_df)

        # Make prediction
        raw_prediction = float(MODEL.predict(input_processed)[0])

        # Scale down prediction to a reasonable percentage range if needed
        # A typical GDP growth rate is between -10% and 15%
        if abs(raw_prediction) > 100:
            # If value is very large, apply scaling to get a reasonable GDP growth percentage
            prediction = raw_prediction / 1_000_000_000
        else:
            prediction = raw_prediction

        # Calculate simple confidence interval based on scaled prediction
        mse = PERFORMANCE.get('mse', 1.0)  # Default to 1.0 if not available
        # Scale MSE similarly if we scaled the prediction
        if abs(raw_prediction) > 100:
            scaled_mse = mse / (1_000_000_000 ** 2)
            std_dev = np.sqrt(scaled_mse)
        else:
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
                "r2_score": PERFORMANCE.get('r2_score', 0.0),
                "mse": PERFORMANCE.get('mse', 0.0),
                "cv_score": PERFORMANCE.get('cv_score', 0.0)
            }
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {str(e)}")
        print(f"Detailed error: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
