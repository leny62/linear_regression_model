import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import joblib
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Define CategoryEncoder class with a proper module path to avoid pickle issues
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='Country', region_col='Region'):
        self.country_col = country_col
        self.region_col = region_col
        self.countries = []
        self.regions = []
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.countries = sorted(X[self.country_col].unique().tolist())
        self.regions = sorted(X[self.region_col].unique().tolist())

        # Generate and store expected feature names to ensure consistency
        X_transformed = self.transform(X)
        self.feature_names_ = X_transformed.columns.tolist()

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

        # Ensure feature names are preserved and in the correct order
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            # Create missing columns with zeros
            for col in self.feature_names_:
                if col not in result.columns:
                    result[col] = 0
            # Reorder columns to match training data
            result = result[self.feature_names_]

        return result

# Load and prepare data
print("Loading dataset...")
df = pd.read_csv('gdp-growth-of-african-countries.csv')

# Transform to long format
print("Converting to long format...")
df_melted = pd.melt(df, id_vars=['Year'], var_name='Country', value_name='GDP')

# Calculate GDP growth rates properly
print("Calculating growth rates...")
df_sorted = df_melted.sort_values(['Country', 'Year'])
df_sorted['GDP_prev'] = df_sorted.groupby('Country')['GDP'].shift(1)
df_sorted['GDP_Growth'] = ((df_sorted['GDP'] - df_sorted['GDP_prev']) / df_sorted['GDP_prev']) * 100

# Drop rows with missing growth rates (first year for each country)
df_growth = df_sorted.dropna(subset=['GDP_Growth'])

print(f"Data with growth rates shape: {df_growth.shape}")

# Feature Engineering
df_growth['Decade'] = (df_growth['Year'] // 10) * 10
df_growth['Post_2000'] = (df_growth['Year'] > 2000).astype(int)

REGION_MAPPING = {
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

df_growth['Region'] = df_growth['Country'].map(lambda x: REGION_MAPPING.get(x, 'Other'))

# Prepare data for modeling
print("Preparing data for modeling...")
features = ['Year', 'Country', 'Region', 'Decade', 'Post_2000']
X = df_growth[features]
y = df_growth['GDP_Growth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit preprocessor
preprocessor = CategoryEncoder()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'SGD Regression': SGDRegressor(max_iter=10000, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
best_model = None
best_score = float('-inf')

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Fit model
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred_test = model.predict(X_test_processed)

    # Calculate metrics
    test_score = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5)

    print(f"Test R²: {test_score:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Update best model if current model has better test score
    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_model_name = name
        best_metrics = {
            'r2_score': test_score,
            'mse': test_mse,
            'cv_score': cv_scores.mean()
        }

print(f"\nBest performing model: {best_model_name}")
print(f"Best R² score: {best_score:.4f}")

# Save the best model with all necessary components
model_info = {
    'model': best_model,
    'preprocessor': preprocessor,
    'performance': best_metrics,
    'model_name': best_model_name,
    'feature_names': X_train_processed.columns.tolist(),
    'region_mapping': REGION_MAPPING
}

# Save the model information, including the CategoryEncoder class definition
model_path = 'best_gdp_growth_model.pkl'
with open(model_path, 'wb') as f:
    joblib.dump(model_info, f)

print(f"\nBest model ({best_model_name}) saved with preprocessor and performance metrics to {model_path}")

# Create a helper script for the API
with open('model_utils.py', 'w') as f:
    f.write("""
# model_utils.py - Helper module for model loading in API
from sklearn.base import BaseEstimator, TransformerMixin

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='Country', region_col='Region'):
        self.country_col = country_col
        self.region_col = region_col
        self.countries = []
        self.regions = []
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.countries = sorted(X[self.country_col].unique().tolist())
        self.regions = sorted(X[self.region_col].unique().tolist())

        # Generate and store expected feature names to ensure consistency
        X_transformed = self.transform(X)
        self.feature_names_ = X_transformed.columns.tolist()

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

        # Ensure feature names are preserved and in the correct order
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            # Create missing columns with zeros
            for col in self.feature_names_:
                if col not in result.columns:
                    result[col] = 0
            # Reorder columns to match training data
            result = result[self.feature_names_]

        return result
""")

print("Created model_utils.py for API compatibility")

# Test the model loading
print("\nTesting model loading...")
loaded_model_info = joblib.load(model_path)
print("Model loaded successfully!")
print(f"Model type: {type(loaded_model_info['model'])}")
print(f"Preprocessor type: {type(loaded_model_info['preprocessor'])}")
