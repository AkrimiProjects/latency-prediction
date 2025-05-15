import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.data_preprocessing import (handle_missing_values, prepare_data,
                                    scale_features)
from src.feature_engineering import engineer_features
from src.mlflow_utils import log_model_metrics, setup_mlflow


def train_and_save_model(model_type="random_forest", hyperparameter_tuning=False):
    """
    Main function to train and save a model for the telecom dataset
    
    Args:
        model_type (str): The type of model to train (random_forest, gradient_boosting, xgboost)
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        Trained model and metrics
    """
    # Set up MLflow tracking
    setup_mlflow("Latency_prediction")
    
    # Load the data
    data = load_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return None, None
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Feature engineering
    data = engineer_features(data)
    
    # Prepare the data (split into train/test)
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Get categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    numerical_transformer = Pipeline(steps=[
        ('imputer', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Define the model
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Hyperparameter tuning if requested
    if hyperparameter_tuning:
        if model_type == "random_forest":
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 15, 20],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_type == "gradient_boosting":
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        elif model_type == "xgboost":
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 6, 9],
                'model__min_child_weight': [1, 3, 5],
                'model__subsample': [0.6, 0.8, 1.0]
            }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Train the model
        pipeline.fit(X_train, y_train)
    
    # Save the preprocessor
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    
    # Evaluate the model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training score (R²): {train_score:.4f}")
    print(f"Test score (R²): {test_score:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Log metrics to MLflow
    metrics = {
        "train_r2": train_score,
        "test_r2": test_score,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }
    
    params = pipeline.get_params()
    
    log_model_metrics(metrics, params, pipeline, f"telecom_{model_type}")
    
    # Save the model
    joblib.dump(pipeline, os.path.join(model_dir, "best_model.joblib"))
    
    print(f"Model saved to {os.path.join(model_dir, 'best_model.joblib')}")
    
    # Get feature importances if available
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        # Get feature names from preprocessor
        cat_features = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
        all_features = numerical_cols + list(cat_features)
        
        feature_importances = pd.DataFrame(
            pipeline.named_steps['model'].feature_importances_,
            index=all_features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importances.head(10))
    
    return pipeline, metrics

if __name__ == "__main__":
    train_and_save_model(model_type="xgboost", hyperparameter_tuning=False) 