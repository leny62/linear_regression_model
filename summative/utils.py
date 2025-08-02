# summative/utils.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

"""
This file contains shared utility classes and functions for the project,
ensuring that both training and prediction environments use the exact same code.
"""

class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer to one-hot encode categorical features (Country and Region)
    while ensuring consistency between training and prediction.
    """
    def __init__(self, country_col='Country', region_col='Region'):
        self.country_col = country_col
        self.region_col = region_col
        self.countries_ = []
        self.regions_ = []
        self.feature_names_in_ = []
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        """
        Learns the unique categories for countries and regions from the training data.
        """
        self.countries_ = sorted(X[self.country_col].unique().tolist())
        self.regions_ = sorted(X[self.region_col].unique().tolist())
        
        # Store the feature names seen during fit
        self.feature_names_in_ = X.columns.tolist()
        
        # Generate the output feature names
        X_transformed = self.transform(X.copy())
        self.feature_names_out_ = X_transformed.columns.tolist()
        
        return self

    def transform(self, X):
        """
        Transforms the data by one-hot encoding the categorical features.
        It ensures the output has the same columns as the fitted data.
        """
        X_copy = X.copy()
        
        # One-hot encode countries and regions
        country_dummies = pd.get_dummies(X_copy[self.country_col], prefix='country', dtype=int)
        region_dummies = pd.get_dummies(X_copy[self.region_col], prefix='region', dtype=int)

        # Ensure all learned columns are present
        for country in self.countries_:
            col_name = f'country_{country}'
            if col_name not in country_dummies.columns:
                country_dummies[col_name] = 0
        
        for region in self.regions_:
            col_name = f'region_{region}'
            if col_name not in region_dummies.columns:
                region_dummies[col_name] = 0
        
        # Drop original categorical columns and concatenate new ones
        X_copy = X_copy.drop([self.country_col, self.region_col], axis=1)
        result = pd.concat([X_copy, country_dummies, region_dummies], axis=1)

        # Ensure the final output has columns in the correct order
        if self.feature_names_out_:
            # Add any missing columns that were in the training set
            for col in self.feature_names_out_:
                if col not in result.columns:
                    result[col] = 0
            # Reorder to match the order during fit
            result = result[self.feature_names_out_]
            
        return result

    def get_feature_names_out(self, input_features=None):
        """Returns the feature names of the transformed data."""
        return self.feature_names_out_

