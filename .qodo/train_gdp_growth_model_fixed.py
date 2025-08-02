import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path to find the utils module
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Import CategoryEncoder from shared module
from utils import CategoryEncoder

warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('ggplot')

# Load the data
print("Loading dataset...")
df = pd.read_csv('../summative/linear_regression/gdp-growth-of-african-countries.csv')

# Examine the structure
print(f"Original dataset shape: {df.shape}")
print(f"First few rows of original data:")
print(df.head())

# Convert to a long format for easier processing
print("Melting data to long format...")
df_melted = pd.melt(df, id_vars=['Year'], var_name='Country', value_name='GDP')

# Calculate GDP growth rates
print("Calculating annual GDP growth rates...")
# Group by country and sort by year
df_sorted = df_melted.sort_values(['Country', 'Year'])

# Calculate year-over-year percentage growth
df_sorted['GDP_prev'] = df_sorted.groupby('Country')['GDP'].shift(1)
df_sorted['GDP_Growth'] = ((df_sorted['GDP'] - df_sorted['GDP_prev']) / df_sorted['GDP_prev']) * 100
df_sorted.dropna(subset=['GDP_Growth'], inplace=True)

print("GDP growth rate calculation complete.")
print(f"Data after calculating growth rates (first few rows):")
print(df_sorted.head())

# Basic statistics
print("\nGDP Growth Statistics:")
print(df_sorted['GDP_Growth'].describe())

# Plot histograms to see distribution of growth rates
plt.figure(figsize=(12, 6))
sns.histplot(df_sorted['GDP_Growth'], kde=True, bins=50)
plt.title('Distribution of GDP Growth Rates (%)')
plt.xlabel('GDP Growth Rate (%)')
plt.ylabel('Frequency')
plt.savefig('gdp_growth_distribution.png')
plt.close()

# Detect and handle outliers
Q1 = df_sorted['GDP_Growth'].quantile(0.25)
Q3 = df_sorted['GDP_Growth'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nOutlier thresholds: Lower bound = {lower_bound:.2f}%, Upper bound = {upper_bound:.2f}%")

# Filter out extreme outliers for modeling purposes
df_filtered = df_sorted[(df_sorted['GDP_Growth'] >= lower_bound) &
                         (df_sorted['GDP_Growth'] <= upper_bound)]

print(f"Removed {len(df_sorted) - len(df_filtered)} outliers")
print(f"Remaining data points: {len(df_filtered)}")

# Feature Engineering
print("\nPerforming feature engineering...")
df_filtered['Decade'] = (df_filtered['Year'] // 10) * 10
df_filtered['Post_2000'] = (df_filtered['Year'] > 2000).astype(int)

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

df_filtered['Region'] = df_filtered['Country'].map(lambda x: REGION_MAPPING.get(x, 'Other'))

# Data Visualization: GDP Growth by Region
plt.figure(figsize=(15, 7))
sns.boxplot(x='Region', y='GDP_Growth', data=df_filtered)
plt.title('GDP Growth by Region (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gdp_growth_by_region.png')
plt.close()

# Data Visualization: GDP Growth by Decade
plt.figure(figsize=(12, 6))
sns.boxplot(x='Decade', y='GDP_Growth', data=df_filtered)
plt.title('GDP Growth by Decade (%)')
plt.tight_layout()
plt.savefig('gdp_growth_by_decade.png')
plt.close()

# Data Visualization: Correlation Matrix
numeric_cols = ['Year', 'GDP', 'GDP_Growth', 'Decade', 'Post_2000']
plt.figure(figsize=(10, 8))
corr_matrix = df_filtered[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Prepare data for modeling
print("\nPreparing data for modeling...")
features = ['Year', 'Country', 'Region', 'Decade', 'Post_2000']
X = df_filtered[features]
y = df_filtered['GDP_Growth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create and fit preprocessor
preprocessor = CategoryEncoder()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define models
print("\nTraining and evaluating models...")
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

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual GDP Growth (%)')
    plt.ylabel('Predicted GDP Growth (%)')
    plt.title(f'{name}: Actual vs Predicted GDP Growth')
    plt.savefig(f'{name.lower().replace(" ", "_")}_predictions.png')
    plt.close()

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

# Plot feature importance for Random Forest
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importances = pd.DataFrame({
        'feature': X_train_processed.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Save the best model with all necessary components
model_info = {
    'model': best_model,
    'preprocessor': preprocessor,
    'performance': best_metrics,
    'model_name': best_model_name,
    'feature_names': X_train_processed.columns.tolist(),
    'region_mapping': REGION_MAPPING
}

# Save as a single pickle file
output_path = '../summative/linear_regression/best_gdp_growth_model.pkl'
joblib.dump(model_info, output_path)
print(f"\nBest model ({best_model_name}) saved with preprocessor and performance metrics to {output_path}")

# Example prediction
print("\nExample prediction:")
sample_country = X_test['Country'].iloc[0]
sample_year = X_test['Year'].iloc[0]
sample_actual = y_test.iloc[0]

sample_input = pd.DataFrame({
    'Year': [sample_year],
    'Country': [sample_country],
    'Region': [REGION_MAPPING[sample_country]],
    'Decade': [(sample_year // 10) * 10],
    'Post_2000': [int(sample_year > 2000)]
})

sample_processed = preprocessor.transform(sample_input)
sample_prediction = best_model.predict(sample_processed)[0]

print(f"Country: {sample_country}")
print(f"Year: {sample_year}")
print(f"Actual GDP Growth: {sample_actual:.2f}%")
print(f"Predicted GDP Growth: {sample_prediction:.2f}%")
print(f"Difference: {abs(sample_actual - sample_prediction):.2f}%")
