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

warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('gdp-growth-of-african-countries.csv')
df = pd.melt(df, id_vars=['Year'], var_name='Country', value_name='GDP_Growth')

# Feature Engineering
df['Decade'] = (df['Year'] // 10) * 10
df['Post_2000'] = (df['Year'] > 2000).astype(int)

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

df['Region'] = df['Country'].map(lambda x: REGION_MAPPING.get(x, 'Other'))

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='Country', region_col='Region'):
        self.country_col = country_col
        self.region_col = region_col
        self.countries = sorted(df['Country'].unique().tolist())
        self.regions = sorted(list(set(REGION_MAPPING.values())))

    def fit(self, X, y=None):
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
        
        # Store feature names
        self.feature_names_ = result.columns.tolist()
        return result

# Prepare data for modeling
df_clean = df.dropna(subset=['GDP_Growth'])
features = ['Year', 'Country', 'Region', 'Decade', 'Post_2000']
X = df_clean[features]
y = df_clean['GDP_Growth']

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
    'performance': best_metrics
}

# Save as a single pickle file
joblib.dump(model_info, 'best_gdp_growth_model.pkl')
print(f"\nBest model ({best_model_name}) saved with preprocessor and performance metrics")
