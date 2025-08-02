import joblib
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path

# Filter warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def predict_gdp_growth(country, year, model_path=None):
    """Predict GDP growth for a specific country and year using the saved model.

    Parameters:
    country (str): Country name
    year (int): Year for prediction
    model_path (str): Path to the saved model (defaults to best_gdp_growth_model.pkl in current directory)

    Returns:
    dict: Dictionary containing prediction and metadata
    """
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_gdp_growth_model.pkl')
        
        print(f"Loading model from: {model_path}")
        
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Extract components from the model data
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        performance = model_data.get('performance', {})
        model_name = model_data.get('model_name', 'Unknown model')
        
        # Get region mapping (either from model_data or use default)
        region_mapping = model_data.get('region_mapping', {
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
        })

        # Validate country
        if country not in region_mapping:
            raise ValueError(f"Invalid country: {country}. Must be one of: {', '.join(sorted(region_mapping.keys()))}")

        # Determine the region
        region = region_mapping.get(country)

        # Calculate decade and post-2000 features
        decade = (year // 10) * 10
        post_2000 = int(year > 2000)

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Year': [year],
            'Country': [country],
            'Region': [region],
            'Decade': [decade],
            'Post_2000': [post_2000]
        })

        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = float(model.predict(input_processed)[0])
        
        # Calculate confidence interval
        mse = performance.get('mse', 0)
        std_dev = np.sqrt(mse)
        confidence_interval = {
            "lower_bound": prediction - 1.96 * std_dev,
            "upper_bound": prediction + 1.96 * std_dev
        }
        
        # Return prediction with metadata
        return {
            'country': country,
            'year': year,
            'predicted_growth': prediction,
            'confidence_interval': confidence_interval,
            'region': region,
            'model_name': model_name,
            'performance': performance
        }

    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        country = input("Enter country name: ")
        year = int(input("Enter year for prediction: "))

        prediction_result = predict_gdp_growth(country, year)
        
        print("\n===== GDP Growth Prediction =====")
        print(f"Country: {prediction_result['country']}")
        print(f"Year: {prediction_result['year']}")
        print(f"Region: {prediction_result['region']}")
        print(f"Predicted GDP Growth: {prediction_result['predicted_growth']:.2f}%")
        print(f"Confidence Interval: {prediction_result['confidence_interval']['lower_bound']:.2f}% to {prediction_result['confidence_interval']['upper_bound']:.2f}%")
        print(f"Model: {prediction_result['model_name']}")
        print("================================")
        
    except Exception as e:
        print(f"Error: {str(e)}")
