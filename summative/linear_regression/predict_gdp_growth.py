
import joblib
import pandas as pd
import numpy as np
import warnings

# Filter warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def predict_gdp_growth(country, year, model_path='best_gdp_growth_model.pkl',
                      preprocessor_path='gdp_growth_preprocessor.pkl'):
    """Predict GDP growth for a specific country and year using the saved model.

    Parameters:
    country (str): Country name
    year (int): Year for prediction
    model_path (str): Path to the saved model
    preprocessor_path (str): Path to the saved preprocessor

    Returns:
    float: Predicted GDP growth percentage
    """
    try:
        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        # Define region mapping
        region_mapping = {
            'Algeria': 'North Africa',
            'Benin': 'West Africa',
            'Botswana': 'Southern Africa',
            'Burkina Faso': 'West Africa',
            'Burundi': 'East Africa',
            'Cameroon': 'Central Africa',
            'Central African Republic': 'Central Africa',
            'Chad': 'Central Africa',
            'Eswatini': 'Southern Africa',
            'Ethiopia': 'East Africa',
            'Gabon': 'Central Africa',
            'Ghana': 'West Africa',
            'Kenya': 'East Africa',
            'Lesotho': 'Southern Africa',
            'Liberia': 'West Africa',
            'Libya': 'North Africa',
            'Madagascar': 'East Africa',
            'Mauritius': 'East Africa',
            'Morocco': 'North Africa',
            'Niger': 'West Africa',
            'Nigeria': 'West Africa',
            'Rwanda': 'East Africa',
            'Senegal': 'West Africa',
            'Seychelles': 'East Africa',
            'Sierra Leone': 'West Africa',
            'Somalia': 'East Africa',
            'South Africa': 'Southern Africa',
            'Sudan': 'North Africa',
            'Tanzania': 'East Africa',
            'Togo': 'West Africa',
            'Uganda': 'East Africa',
            'Zambia': 'Southern Africa',
            'Zimbabwe': 'Southern Africa'
        }

        # Determine the region
        region = region_mapping.get(country, 'Other')

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

        # Preprocess and make prediction
        input_processed = preprocessor.transform(input_data)

        # Convert to numpy array while preserving feature names
        if isinstance(input_processed, pd.DataFrame):
            # Ensure columns are in the same order as during training
            if hasattr(preprocessor, 'feature_names_'):
                input_processed = input_processed[preprocessor.feature_names_]
            input_processed = input_processed.to_numpy()

        prediction = model.predict(input_processed)[0]

        return prediction

    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        print(f"Input data shape: {input_processed.shape if 'input_processed' in locals() else 'N/A'}")
        print(f"Expected features: {preprocessor.feature_names_ if hasattr(preprocessor, 'feature_names_') else 'N/A'}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        country = input("Enter country name: ")
        year = int(input("Enter year for prediction: "))

        prediction = predict_gdp_growth(country, year)
        print(f"Predicted GDP growth for {country} in {year}: {prediction:.2f}%")
    except Exception as e:
        print(f"Error: {str(e)}")
