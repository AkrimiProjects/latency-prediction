import argparse
import os

import uvicorn

from data_loader import load_data
from data_preprocessing import handle_missing_values
from model_training import train_model


def create_directories():
    """Create necessary directories for the project"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("templates", exist_ok=True)  
    os.makedirs("static", exist_ok=True)  

def main():
    """
    Main entry point for the MLOps pipeline
    """
    parser = argparse.ArgumentParser(description='Telecom Throughput Prediction Pipeline')
    parser.add_argument('--data-path', type=str, default=None, 
                        help='Path to the dataset (default: data/telecom_data.csv)')
    parser.add_argument('--model-type', type=str, default='random_forest', 
                        choices=['random_forest', 'gradient_boosting', 'linear'],
                        help='Type of model to train')
    parser.add_argument('--train', action='store_true', 
                        help='Train a new model')
    parser.add_argument('--serve', action='store_true', 
                        help='Start the prediction API server')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port for the API server (default: 8000)')
    args = parser.parse_args()

    # Create necessary directories
    create_directories()

    # Train a new model if requested
    if args.train:
        print("=== Training a new model ===")
        print(f"Model type: {args.model_type}")
        
        # Load data
        data = load_data(args.data_path)
        if data is None:
            print("Failed to load data. Exiting.")
            return
        
        # Handle missing values
        data = handle_missing_values(data)
        
        # Train the model
        model, metrics = train_model(model_type=args.model_type)
        
        print("\n=== Training Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    # Start the API server if requested
    if args.serve:
        print(f"=== Starting API server on port {args.port} ===")
        
        # Import here to avoid circular import
        from api import app
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    # If no action specified, print help
    if not (args.train or args.serve):
        parser.print_help()

if __name__ == "__main__":
    main() 