import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path=None):
    try:
        # Default data path
        if data_path is None:
            data_path = os.path.join(os.getcwd(), 'data', 'Train.csv')
        
        # Check if the file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return None
        
        # Load the data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Basic data validation
        if data.shape[0] == 0:
            logger.warning("Loaded data has 0 rows")
        else:
            logger.info(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None
        
def load_test_data(test_path=None):
    """
    Load the test data for predictions
    
    Args:
        test_path (str, optional): Path to the test data file
        
    Returns:
        pandas.DataFrame: Test dataset
    """
    try:
        # Default test path
        if test_path is None:
            test_path = os.path.join(os.getcwd(), 'data', 'Test.csv')
        
        # Check if the file exists
        if not os.path.exists(test_path):
            logger.error(f"Test data file not found: {test_path}")
            return None
        
        # Load the data
        logger.info(f"Loading test data from {test_path}")
        test_data = pd.read_csv(test_path)
        
        logger.info(f"Loaded test data with {test_data.shape[0]} rows and {test_data.shape[1]} columns")
        
        return test_data
    
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return None
