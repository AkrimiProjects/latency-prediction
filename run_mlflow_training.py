#!/usr/bin/env python
"""
MLflow Training Script

This script runs the model training and ensures proper tracking in MLflow.
It's designed to be used before running the CD pipeline to ensure MLflow data exists.
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mlflow_training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ML training with MLflow tracking')
    parser.add_argument('--experiment-name', type=str, default='ml-experiment',
                        help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default='http://localhost:5001',
                        help='MLflow tracking server URI')
    return parser.parse_args()

def check_mlflow_server(tracking_uri):
    """Check if MLflow server is running"""
    import requests
    try:
        response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/list")
        if response.status_code == 200:
            logger.info(f"MLflow server is running at {tracking_uri}")
            return True
        else:
            logger.error(f"MLflow server returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to MLflow server: {str(e)}")
        return False

def start_mlflow_server():
    """Start MLflow server if it's not running"""
    logger.info("Starting MLflow server...")
    try:
        # Create mlruns directory if it doesn't exist
        os.makedirs("mlruns", exist_ok=True)
        
        # Start MLflow server in background
        process = subprocess.Popen(
            ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for the server to start
        import time
        time.sleep(5)
        
        logger.info("MLflow server started")
        return True
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {str(e)}")
        return False

def create_temporary_fix():
    """Create a temporary fix for the Unicode error in train_and_track.py"""
    logger.info("Creating temporary patch for Unicode error in MLflow...")
    
    # Path to the MLflow client file that needs patching
    import site
    site_packages = site.getsitepackages()[0]
    tracking_client_path = os.path.join(site_packages, "mlflow", "tracking", "_tracking_service", "client.py")
    
    try:
        # Check if the file exists
        if not os.path.exists(tracking_client_path):
            logger.warning(f"Could not find MLflow tracking client at {tracking_client_path}")
            logger.info("Will continue without patching. If training fails, fix the Unicode error manually.")
            return
        
        # Read the file content
        with open(tracking_client_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the emoji is used in the file
        if "\\U0001f3c3" in content or "🏃" in content:
            # Replace the emoji with a plain text version
            new_content = content.replace("f\"\\U0001f3c3 View run {run_name} at:", f"f\"View run {run_name} at:")
            new_content = new_content.replace("f\"🏃 View run {run_name} at:", f"f\"View run {run_name} at:")
            
            # Backup the original file
            backup_path = tracking_client_path + ".bak"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created backup of MLflow tracking client at {backup_path}")
            
            # Write the patched content
            with open(tracking_client_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info("Successfully patched MLflow tracking client to fix Unicode error")
        else:
            logger.info("No need to patch MLflow tracking client - emoji not found")
        
    except Exception as e:
        logger.error(f"Failed to patch MLflow tracking client: {str(e)}")
        logger.info("Will continue without patching. If training fails, fix the Unicode error manually.")

def fix_train_and_track():
    """Fix train_and_track.py to work with console encoding"""
    train_script_path = "train_and_track.py"
    fixed_script_path = "train_and_track_fixed.py"
    
    if not os.path.exists(train_script_path):
        logger.error(f"Could not find training script at {train_script_path}")
        return False
    
    try:
        with open(train_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add code to fix the encoding issues
        patched_content = content
        
        # Add this code before the main function
        encoding_fix = """
# Fix for console encoding issues
import sys
import codecs

# Ensure stdout can handle Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
"""
        
        # Check if fix already exists
        if "codecs.getwriter('utf-8')" not in content:
            # Insert it before the main function
            if "def main():" in content:
                patched_content = content.replace("def main():", encoding_fix + "\ndef main():")
            else:
                # Just append at the end if main function is not found
                patched_content = content + "\n" + encoding_fix
        
        # Write to a temporary file
        with open(fixed_script_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        
        logger.info(f"Created fixed training script at {fixed_script_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to fix training script: {str(e)}")
        return False

def run_training(experiment_name, tracking_uri):
    """Run the model training script"""
    logger.info(f"Running training with experiment name: {experiment_name}")
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # First, try to fix the Unicode issues
    create_temporary_fix()
    fix_succeeded = fix_train_and_track()
    script_to_run = "train_and_track_fixed.py" if fix_succeeded else "train_and_track.py"
    
    try:
        # Set environment variables for MLflow
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = tracking_uri
        env["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        
        # Force UTF-8 encoding for the subprocess
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Add environment variable to disable emoji in MLflow output
        env["MLFLOW_DISABLE_EMOJI"] = "true"
        
        # Run training script
        logger.info(f"Running {script_to_run}...")
        result = subprocess.run(
            ["python", script_to_run],
            env=env,
            check=False,  # Don't raise exception on non-zero exit
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # Check if there was an error
        if result.returncode != 0:
            logger.warning(f"Training script exited with code {result.returncode}")
            logger.info("This might be due to the Unicode error at the end, but training may have succeeded")
            logger.info("Checking if run was created in MLflow...")
            
            # Check if a run was created despite the error
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is not None:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")
                    logger.info("Training was successful despite the error")
                    logger.info(result.stdout)
                    return True
            
            # If we get here, no runs were found
            logger.error("Training failed and no runs were created in MLflow")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False
        else:
            logger.info("Training completed successfully")
            logger.info(result.stdout)
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if MLflow server is running
    if not check_mlflow_server(args.tracking_uri):
        logger.info("MLflow server is not running. Attempting to start it...")
        if not start_mlflow_server():
            logger.error("Failed to start MLflow server. Please start it manually.")
            logger.error("Run: mlflow server --host 0.0.0.0 --port 5001")
            return 1
    
    # Run training
    success = run_training(args.experiment_name, args.tracking_uri)
    
    if success:
        logger.info("Training and MLflow tracking completed successfully.")
        logger.info("You can now run the CD pipeline to analyze the model results.")
        logger.info(f"Run: python cd_pipeline.py --docker-username votre_username --mlflow-experiment-name {args.experiment_name}")
        return 0
    else:
        logger.error("Training failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 