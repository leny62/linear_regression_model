# model_utils.py - Helper module for model loading in API
import pandas as pd
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
        country_dummies = pd.get_dummies(X[self.country_col], prefix='country')
        region_dummies = pd.get_dummies(X[self.region_col], prefix='region')

        # Store all possible feature names
        X_copy = X.drop([self.country_col, self.region_col], axis=1)
        all_features = list(X_copy.columns)
        all_features.extend(country_dummies.columns)
        all_features.extend(region_dummies.columns)
        self.feature_names_ = all_features

        return self

    def transform(self, X):
        X_copy = X.copy()
        # One-hot encode countries and regions
        country_dummies = pd.get_dummies(X_copy[self.country_col], prefix='country')
        region_dummies = pd.get_dummies(X_copy[self.region_col], prefix='region')

        # Make sure all expected country and region columns are present
        for country in self.countries:
            col_name = f'country_{country}'
            if col_name not in country_dummies.columns:
                country_dummies[col_name] = 0

        for region in self.regions:
            col_name = f'region_{region}'
            if col_name not in region_dummies.columns:
                region_dummies[col_name] = 0

        # Drop original columns and concatenate
        X_copy = X_copy.drop([self.country_col, self.region_col], axis=1)
        result = pd.concat([X_copy, country_dummies, region_dummies], axis=1)

        # Ensure all expected feature names are present in the correct order
        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            for col in self.feature_names_:
                if col not in result.columns:
                    result[col] = 0

            # Make sure columns are in the exact same order as during training
            result = result[self.feature_names_]

        return result
