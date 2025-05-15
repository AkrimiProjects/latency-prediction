import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_correlation_features(data):
    """
    Create new features based on correlations
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Create interaction features between highly correlated variables
    if all(col in df.columns for col in ['PCell_RSRP_max', 'PCell_RSSI_max']):
        df['RSRP_RSSI_ratio'] = df['PCell_RSRP_max'] / df['PCell_RSSI_max'].replace(0, np.nan)
        df['RSRP_RSSI_ratio'] = df['RSRP_RSSI_ratio'].fillna(df['RSRP_RSSI_ratio'].median())
    
    if all(col in df.columns for col in ['PCell_RSRP_max', 'PCell_SNR_max']):
        df['RSRP_SNR_product'] = df['PCell_RSRP_max'] * df['PCell_SNR_max']
    
    return df


def create_signal_quality_index(data):
    """
    Create a single signal quality index from multiple signal metrics
    """
    df = data.copy()
    
    # Define the columns needed for the index
    signal_cols = ['PCell_RSRP_max', 'PCell_RSRQ_max', 'PCell_SNR_max']
    
    # Check if all required columns exist
    if all(col in df.columns for col in signal_cols):
        # Standardize the columns
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[signal_cols]),
            columns=signal_cols,
            index=df.index
        )
        
        # Create a weighted signal quality index
        df['signal_quality_index'] = (
            0.4 * df_scaled['PCell_RSRP_max'] + 
            0.3 * df_scaled['PCell_RSRQ_max'] + 
            0.3 * df_scaled['PCell_SNR_max']
        )
    
    return df


def create_bandwidth_efficiency(data):
    """
    Create bandwidth efficiency metrics
    """
    df = data.copy()
    
    if all(col in df.columns for col in ['PCell_Downlink_Average_MCS', 'PCell_Downlink_bandwidth_MHz']):
        df['bandwidth_efficiency'] = df['PCell_Downlink_Average_MCS'] / df['PCell_Downlink_bandwidth_MHz']
        df['bandwidth_efficiency'] = df['bandwidth_efficiency'].fillna(df['bandwidth_efficiency'].median())
    
    return df


def create_environmental_impact(data):
    """
    Create features related to environmental impact on signal
    """
    df = data.copy()
    
    # Temperature and humidity impact
    if all(col in df.columns for col in ['temperature', 'humidity']):
        df['temp_humid_interaction'] = df['temperature'] * df['humidity']
    
    # Area-specific speed impact
    if all(col in df.columns for col in ['speed_kmh', 'area']):
        # Create area-specific speed impact
        df['mobility_factor'] = df['speed_kmh']
        
        # Adjust for different areas
        area_impact = {
            'Urban': 1.2,      # Higher impact in urban areas
            'Residential': 0.9, # Moderate impact
            'Rural': 0.5,       # Lower impact
            'Industrial': 1.0,  # Standard impact
            'Commercial': 1.1   # Slightly higher impact
        }
        
        for area, impact in area_impact.items():
            mask = df['area'] == area
            if mask.any():
                df.loc[mask, 'mobility_factor'] = df.loc[mask, 'speed_kmh'] * impact
    
    return df


def create_feature_engineering_pipeline():
    """
    Create a pipeline for feature engineering on the telecom dataset
    """
    # Identify numerical and categorical columns
    numerical_cols = [ 'PCell_RSRP_max', 'PCell_RSRQ_max', 'PCell_RSSI_max', 'PCell_SNR_max',
    'PCell_Downlink_Num_RBs', 'PCell_Downlink_Average_MCS',
    'SCell_RSRP_max', 'SCell_RSRQ_max', 'SCell_RSSI_max', 'SCell_SNR_max',
    'speed_kmh', 'Traffic Jam Factor', 'temperature', 'humidity', 'windSpeed',
    'target']
    
    categorical_cols = ['device', 'area']
    
    # Numerical transformer pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor


def engineer_features(data):
    """
    Apply all feature engineering steps
    """
    df = data.copy()
    
    # Apply all feature engineering functions
    df = create_correlation_features(df)
    df = create_signal_quality_index(df)
    df = create_bandwidth_efficiency(df)
    df = create_environmental_impact(df)
    
    # Handle NaN values created during feature engineering
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df
