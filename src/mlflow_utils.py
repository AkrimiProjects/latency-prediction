import logging
import os

import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name):
    """
    Set up MLflow tracking
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        str: ID of the experiment
    """
    try:
        # Set up tracking URI (use local directory by default)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
        
        # Set the experiment as active
        mlflow.set_experiment(experiment_name)
        
        return experiment_id
    
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        return None

def log_model_metrics(metrics, params, model, model_name):
    """
    Log model metrics, parameters, and the model itself to MLflow
    
    Args:
        metrics (dict): Dictionary of metrics to log
        params (dict): Dictionary of parameters to log
        model: Trained model to log
        model_name (str): Name of the model
    """
    try:
        # Start a new MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log the model
            mlflow.sklearn.log_model(model, model_name)
            
            # Log artifact URI
            artifact_uri = mlflow.get_artifact_uri()
            logger.info(f"Model saved in run {run.info.run_id}")
            logger.info(f"Model artifacts stored at: {artifact_uri}")
            
            return run.info.run_id
    
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def get_best_model(experiment_name, metric_name="test_rmse", ascending=False):
    """
    Get the best model from MLflow based on a metric
    
    Args:
        experiment_name (str): Name of the experiment
        metric_name (str): Name of the metric to use for comparison
        ascending (bool): Whether to sort in ascending order (True for metrics like RMSE, False for R2)
        
    Returns:
        dict: Information about the best run
    """
    try:
        # Set the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return None
        
        # Get all runs for the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning(f"No runs found for experiment '{experiment_name}'")
            return None
        
        # Sort runs by metric
        if ascending:
            best_run = runs.sort_values(f"metrics.{metric_name}").iloc[0]
        else:
            best_run = runs.sort_values(f"metrics.{metric_name}", ascending=False).iloc[0]
        
        logger.info(f"Found best run {best_run.run_id} with {metric_name} = {best_run[f'metrics.{metric_name}']}")
        
        return {
            "run_id": best_run.run_id,
            "metrics": {col.replace("metrics.", ""): best_run[col] for col in best_run.keys() if col.startswith("metrics.")},
            "params": {col.replace("params.", ""): best_run[col] for col in best_run.keys() if col.startswith("params.")},
            "artifact_uri": best_run.artifact_uri
        }
    
    except Exception as e:
        logger.error(f"Error getting best model: {str(e)}")
        return None

def load_model(run_id: str, model_name: str):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
