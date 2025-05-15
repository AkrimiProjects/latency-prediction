import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_missing_values(data):
    """
    Handle missing values in the dataset
    
    Args:
        data (pandas.DataFrame): The original dataset
        
    Returns:
        pandas.DataFrame: Dataset with handled missing values
    """
    if data is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Get lists of numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numeric_cols:
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled {col} missing values with median: {median_value}")
        
        # Fill categorical missing values with most frequent value
        for col in categorical_cols:
            if df[col].isna().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        return data

def encode_categorical_variables(data):
    """
    Encode categorical variables in the dataset
    
    Args:
        data (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Dataset with encoded categorical variables
    """
    if data is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Get list of categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # One-hot encode categorical variables
        for col in categorical_cols:
            # Skip if the column is 'id' or 'timestamp'
            if col.lower() in ['id', 'timestamp']:
                continue
            
            # Create dummies
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            
            # Add to dataframe
            df = pd.concat([df, dummies], axis=1)
            
            # Drop original column
            df = df.drop(col, axis=1)
            
            logger.info(f"Encoded categorical column: {col}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error encoding categorical variables: {str(e)}")
        return data

def scale_features(X_train, X_test=None):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame, optional): Test features
        
    Returns:
        tuple: Scaled training and test features
    """
    try:
        # Create a copy
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy() if X_test is not None else None
        
        # Get numerical columns
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        # Create and fit scaler
        scaler = StandardScaler()
        
        # Scale training data
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        
        # Scale test data if provided
        if X_test is not None:
            X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Save the scaler
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        logger.info(f"Scaled {len(numeric_cols)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        return X_train, X_test

def prepare_data(data, test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting into train and test sets
    
    Args:
        data (pandas.DataFrame): The dataset
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Remove ID and timestamp columns if they exist
        drop_cols = ['id', 'timestamp']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
                logger.info(f"Dropped column: {col}")
        
        # Check if target column exists
        if 'target' not in df.columns:
            logger.error("Target column 'target' not found in the dataset")
            return None, None, None, None
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples)")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return None, None, None, None

def create_preprocessing_pipeline(categorical_cols, numeric_cols):
    """
    Create a preprocessing pipeline for categorical and numerical features
    
    Args:
        categorical_cols (list): List of categorical column names
        numeric_cols (list): List of numerical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    try:
        # Numeric pipeline with imputer and scaler
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline with imputer and one-hot encoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        return preprocessor
    
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        return None
