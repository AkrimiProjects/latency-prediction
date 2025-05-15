import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data
from src.data_preprocessing import (handle_missing_values, prepare_data,
                                    scale_features)
from src.feature_engineering import engineer_features
from src.mlflow_utils import log_model_metrics, setup_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model_type="random_forest", params=None, X_train=None, y_train=None, X_test=None, y_test=None):
    """Train a machine learning model

    Args:
        model_type (str): Type of model to train (random_forest, gradient_boosting, xgboost, linear)
        params (dict): Parameters for the model
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Test features
        y_test (Series): Test target

    Returns:
        Trained model and evaluation metrics
    """
    try:
        # Validate inputs
        if X_train is None or y_train is None:
            logger.error("Training data not provided")
            return None, None
        
        # Set default parameters if none provided
        if params is None:
            params = get_default_params(model_type)
        
        # Create the model based on model_type
        model = create_model(model_type, params)
        
        if model is None:
            logger.error(f"Failed to create model of type: {model_type}")
            return None, None
        
        # Train the model
        logger.info(f"Training {model_type} model")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Log model to MLflow
        log_model_to_mlflow(model, params, metrics, model_type)
        
        # Save the model
        save_model(model, model_type)
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, None

def get_default_params(model_type):
    """Get default parameters for the specified model type"""
    if model_type == "random_forest":
        return {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42
        }
    elif model_type == "gradient_boosting":
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "tree_method": "hist",  # CPU-based histogram method
            "device": "cpu",        # Force CPU usage
            "random_state": 42
        }
    elif model_type == "linear":
        return {}
    else:
        logger.warning(f"Unknown model type: {model_type}, using random forest defaults")
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }

def create_model(model_type, params):
    """Create a model instance based on model_type and params"""
    try:
        if model_type == "random_forest":
            return RandomForestRegressor(**params)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        elif model_type == "xgboost":
            # Remove all GPU-related parameters to ensure CPU usage
            gpu_params = ['gpu_id', 'device', 'single_precision_histogram']
            for param in gpu_params:
                if param in params:
                    del params[param]
            
            # Explicitly set CPU-only parameters
            params['tree_method'] = 'hist'  # CPU histogram-based approach
            params['device'] = 'cpu'        # Force CPU usage
            
            return xgb.XGBRegressor(**params)
        elif model_type == "linear":
            return LinearRegression(**params)
        else:
            logger.warning(f"Unknown model type: {model_type}, using random forest")
            return RandomForestRegressor(**get_default_params("random_forest"))
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return None

def evaluate_model(model, X_train, y_train, X_test=None, y_test=None):
    """Evaluate a trained model on train and test data"""
    try:
        # Predictions on training data
        y_pred_train = model.predict(X_train)
        
        # Calculate training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        metrics = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2
        }
        
        logger.info(f"Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        # If test data is provided, calculate test metrics
        if X_test is not None and y_test is not None:
            y_pred_test = model.predict(X_test)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            metrics.update({
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2
            })
            
            logger.info(f"Test metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {}

def log_model_to_mlflow(model, params, metrics, model_type):
    """Log model, parameters, and metrics to MLflow"""
    try:
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{model_type}")
        
        logger.info(f"Model {model_type} logged to MLflow")
    
    except Exception as e:
        logger.error(f"Error logging model to MLflow: {str(e)}")

def save_model(model, model_type):
    """Save the trained model to disk"""
    try:
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
        joblib.dump(model, model_path)
        
        # Also save as best_model.joblib for the API
        best_model_path = os.path.join(model_dir, "best_model.joblib")
        joblib.dump(model, best_model_path)
        
        logger.info(f"Model saved to {model_path} and {best_model_path}")
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")

# Legacy function names for backward compatibility
def train_random_forest(X_train, y_train, **params):
    """Train a random forest model (legacy function)"""
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, **params):
    """Train a linear regression model (legacy function)"""
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    train_model()
