# African GDP Growth Prediction Model

A machine learning project that predicts GDP growth rates for African countries using linear regression techniques. This project includes model training, evaluation, an API for making predictions, and a Flutter mobile application.


## Mission

This prediction model supports education and job creation across Africa by helping potential investors evaluate economic performance and trends in target countries. By providing accurate GDP growth forecasts, the model offers valuable insights for investment decision-making, ultimately contributing to sustainable development and employment opportunities.

## Project Structure

```
linear_regression_model/
├── summative/
│   ├── API/                          # FastAPI application for serving predictions
│   │   ├── prediction.py             # API endpoint definitions
│   │   └── requirements.txt          # API dependencies
│   ├── FlutterApp/                   # Mobile application for GDP predictions
│   │   └── gdp_growth_predictor/     # Flutter app project directory
│   │       ├── lib/                  # Flutter app source code
│   │       │   └── main.dart         # Main Flutter application file
│   │       ├── android/              # Android platform-specific code
│   │       ├── ios/                  # iOS platform-specific code
│   │       └── pubspec.yaml          # Flutter dependencies and configuration
│   └── linear_regression/            # Model training and prediction scripts
│       ├── best_gdp_growth_model.pkl # Trained model saved as pickle file
│       ├── gdp_growth_preprocessor.pkl # Preprocessor for the model
│       ├── gdp-growth-of-african-countries.csv # Dataset used for training
│       ├── multivariate.ipynb        # Jupyter notebook with model development
│       ├── model_utils.py            # Helper utilities for model processing
│       ├── predict_gdp_growth.py     # Script for making predictions
│       └── retrain_model.py          # Script for training and saving the model
└── README.md                         # This file
```

## Overview

This project focuses on predicting GDP growth rates for various African countries based on historical data. The model incorporates features such as country, region, year, and temporal features (e.g., decade, post-2000 indicator) to make accurate predictions.

## Features

- **Data-driven GDP growth forecasting** for African countries
- **Region-based analysis** using a custom mapping of countries to regions (North, East, West, Central, and Southern Africa)
- **Investment decision support** for stakeholders focused on education and job creation initiatives
- **Confidence intervals** to indicate prediction reliability
- **Mobile application** for easy access to GDP growth predictions on the go

## Technology Stack

- **Python 3.x** for model development and API
- **scikit-learn** for machine learning algorithms
- **pandas** for data manipulation
- **FastAPI** for API development
- **joblib** for model serialization
- **uvicorn** for ASGI server
- **Flutter** for cross-platform mobile application
- **Dart** as the programming language for mobile development

## Installation

### Prerequisites

- Python 3.6+
- pip package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/leny62/linear_regression_model
   cd linear_regression_model
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies for the API:
   ```
   cd summative/API
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To retrain the model with updated data:

```
cd summative/linear_regression
python train_and_save_model.py
```

### Making Predictions with the Python Script

```python
from summative.linear_regression.predict_gdp_growth import predict_gdp_growth

# Predict GDP growth for Kenya in 2025
result = predict_gdp_growth("Kenya", 2025)
print(f"Predicted GDP growth: {result['predicted_growth']:.2f}%")
```

### Running the API

```
cd summative/API
uvicorn prediction:app --reload --port 9090
```

The API will be available at http://127.0.0.1:9090/docs for interactive Swagger documentation.

### API Endpoints

- **POST /predict/** - Predict GDP growth for a country in a specific year
  - Request body:
    ```json
    {
      "country": "Kenya",
      "year": 2025
    }
    ```
  - Response:
    ```json
    {
      "country": "Kenya",
      "year": 2025,
      "predicted_growth": 5.62,
      "confidence_interval": {"lower": 4.8, "upper": 6.4},
      "region": "East Africa",
      "prediction_metadata": {"model_type": "Linear Regression", "r2_score": 0.85}
    }
    ```

## API Access

The prediction API is publicly available at:
- **Swagger UI Documentation**: [https://linear-regression-model-pwvl.onrender.com/docs](https://linear-regression-model-pwvl.onrender.com/docs)

You can use this endpoint to make GDP growth predictions for any supported African country between 2000-2030.

## Video Demo

[Link to YouTube Demo Video - 5 minutes](https://youtu.be/6E5DttJb1FU)

## Mobile App Details

The GDP Growth Predictor Flutter application provides:

- **User-friendly interface** for predicting GDP growth
- **Country selection** from a dropdown of 33 African nations
- **Year input** for predictions between 2000-2030
- **Result display** showing predicted growth rate with confidence intervals
- **Regional context** indicating which region the selected country belongs to
- **Model metadata** showing R² score, MSE, and cross-validation score

The mobile app connects to the deployed API endpoint and presents predictions in an easy-to-understand format, making the economic forecasting technology accessible to investors, educators, and policymakers without requiring technical expertise.

## Mobile App Instructions

To run the GDP Growth Predictor mobile app:

1. **Prerequisites**:
   - Install [Flutter SDK](https://flutter.dev/docs/get-started/install)
   - Set up an emulator or connect a physical device

2. **Clone and Run**:
   ```
   git clone https://github.com/leny62/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp/gdp_growth_predictor
   flutter pub get
   flutter run
   ```

3. **Using the App**:
   - Launch the app and tap "Make a Prediction"
   - Select a country from the dropdown menu
   - Enter a year between 2000-2030
   - Tap "Predict" to get the GDP growth forecast
   - View the prediction results and confidence intervals

The app provides an intuitive interface for accessing predictions from our deployed model, with built-in validation to ensure inputs meet the required formats.

## Model Details

The project implements a Linear Regression model with custom preprocessing:

1. **Feature Engineering**:
   - Country and region one-hot encoding
   - Temporal features (decade, post-2000 indicator)

2. **Custom Transformers**:
   - `CategoryEncoder` for handling categorical variables

3. **Model Selection**:
   - The best performing model is selected based on cross-validation

## Understanding the Prediction Metrics

When making a prediction with our model, you'll receive several metrics that help interpret the results:

### Confidence Interval
```json
"confidence_interval": {
    "lower_bound": -32.69306942422564,
    "upper_bound": 30.965188019663483
}
```

The confidence interval represents the range within which the actual GDP growth is likely to fall with 95% confidence. In this example:
- **Lower bound**: -32.69% indicates the worst-case scenario for GDP growth
- **Upper bound**: 30.97% indicates the best-case scenario for GDP growth
- The wide interval suggests significant uncertainty in this particular prediction

### Region
```json
"region": "East Africa"
```

This identifies the geographic region of the selected country, which helps contextualize the prediction within regional economic trends.

### Prediction Metadata
```json
"prediction_metadata": {
    "r2_score": 0.1621001249079732,
    "mse": 263.71653352720654,
    "cv_score": -0.1080645301474232
}
```

These metrics describe the overall model performance:

- **R² Score (0.162)**: A measure of how well the model explains the variation in GDP growth. Values range from 0 to 1, with higher values indicating better fit. This relatively low value (0.162) indicates that the model explains about 16.2% of the variance in GDP growth, reflecting the inherent difficulty in predicting macroeconomic indicators.

- **Mean Squared Error (263.72)**: Measures the average squared difference between predicted and actual values. Lower values indicate better performance. This value indicates there's significant variance in the predictions.

- **Cross-Validation Score (-0.108)**: Measures how well the model generalizes to new data. The negative value suggests that the model struggles with certain predictions, which is common with economic data due to unpredictable external factors.

The Random Forest model was selected despite these metrics because it performed better than linear regression and decision trees for this particular prediction task, especially when dealing with the complex relationships between regional factors and GDP growth in African economies.

## Troubleshooting

If you encounter the error `Can't get attribute 'CategoryEncoder' on <module '__main__'>` when loading the model:

1. Make sure you're running the API from the correct directory
2. Verify that the `CategoryEncoder` class is defined in the same scope as where the model is loaded
3. Consider rebuilding the model if necessary

## License

MIT License

## Contributors

Leny Pascal IHIRWE
