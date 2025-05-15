# ML Latency Prediction System with CI/CD Pipeline

This project implements a machine learning system for predicting network latency with a complete CI/CD pipeline, monitoring, and security features.

## Features

- **Machine Learning Model**: XGBoost regressor to predict network latency
- **Web Interface**: Flask-based web app for making predictions
- **MLFlow Integration**: Experiment tracking and model registry
- **CI Pipeline**: Automated code quality checks triggered by file changes
- **CD Pipeline**: Automated Docker image building and model deployment
- **Monitoring**: Prometheus and Grafana for metrics collection and visualization
- **Security Enhancements**: Security headers, environment-based configurations

## Architecture

```
┌─────────────────┐    ┌───────────────┐    ┌────────────────┐
│                 │    │               │    │                │
│  CI Pipeline    ├───►│  CD Pipeline  ├───►│  Docker Image  │
│                 │    │               │    │                │
└─────────────────┘    └───────────────┘    └────────────────┘
                                                   │
                                                   ▼
┌─────────────────┐    ┌───────────────┐    ┌────────────────┐
│                 │    │               │    │                │
│    MLFlow UI    │◄───┤ MLFlow Server ◄────┤  Application   │
│                 │    │               │    │    Server      │
└─────────────────┘    └───────────────┘    └────────────────┘
                                                   │
                                                   ▼
┌─────────────────┐    ┌───────────────┐    ┌────────────────┐
│                 │    │               │    │                │
│     Grafana     ◄────┤  Prometheus   ◄────┤   Metrics      │
│                 │    │               │    │                │
└─────────────────┘    └───────────────┘    └────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd latency-predictor
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
# Create a .env file
echo "SECRET_KEY=your-secret-key" > .env
echo "MLFLOW_TRACKING_URI=http://localhost:5001" >> .env
```

### Running the Application

1. Train a model:
```bash
make train
```

2. Start the MLFlow server:
```bash
make mlflow-ui
```

3. Start the application:
```bash
make run
```

4. Open in browser:
```
http://localhost:5000
```

## CI/CD Pipeline

### Setting Up CI

1. Install CI dependencies:
```bash
make ci-setup
```

2. Start the CI watcher:
```bash
make ci-watch
```

Now, any changes to the code will automatically trigger:
- Code linting (flake8)
- Code formatting (black)
- Security checks (bandit)
- Unit tests (pytest)

### Running CD Pipeline

To trigger the CD pipeline manually:

```bash
make cd-pipeline DOCKER_USERNAME=yourusername
```

This will:
1. Build a Docker image
2. Tag the image
3. Push the image to DockerHub (if credentials are provided)
4. Analyze MLFlow models
5. Deploy the application

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
make docker-build DOCKER_USERNAME=yourusername

# Run the container
make docker-run DOCKER_USERNAME=yourusername
```

### Deploy with Docker Compose

```bash
# Start all services
make start-all

# Stop all services
make stop-all
```

## Monitoring

### Setup Monitoring

```bash
# Setup monitoring infrastructure
make setup-monitoring

# Start monitoring services
make start-monitoring
```

### Access Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Application Metrics**: http://localhost:5000/metrics

## MLFlow Integration

### MLFlow UI

Access the MLFlow UI at http://localhost:5001

### Managing Models

```bash
# Deploy the latest model to production
make mlflow-deploy
```

## API Endpoints

- **Web UI**: `GET /`
- **Prediction API**: `POST /predict` with JSON payload
- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics`
- **Model Info**: `GET /model_info`
- **Reload Model**: `GET /reload_model`

## Security Features

- Content Security Policy headers
- XSS Protection
- Environment-based configuration
- Session secret management
- Secure headers

## Development

### Project Structure

```
├── ci_pipeline.py              # CI pipeline script
├── cd_pipeline.py              # CD pipeline script
├── Dockerfile                  # Main Dockerfile
├── Makefile                    # Makefile with commands
├── README.md                   # This file
├── bandit.yaml                 # Security scan config
├── docker-compose.yml          # Docker Compose config
├── requirements.txt            # Python dependencies
├── ci_results/                 # CI pipeline results
├── cd_results/                 # CD pipeline results
├── data/                       # Data files
├── mlflow-data/                # MLFlow data
├── models/                     # Saved models
├── monitoring/                 # Monitoring configs
│   ├── grafana/                # Grafana dashboards
│   └── prometheus/             # Prometheus config
├── src/                        # Source code
│   ├── main.py                 # Web application
│   ├── metrics.py              # Prometheus metrics
│   └── train_model.py          # Model training
├── static/                     # Static assets
└── templates/                  # HTML templates
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting MLflow Artifacts

### Common MLflow Issues in Docker

When using MLflow in a Docker environment, it's common to encounter issues with model artifacts. The most common problem is that model metadata exists in the MLflow database, but the actual model artifacts are missing or inaccessible. This can happen due to:

1. **Volume mounting issues**: Docker volumes not correctly configured
2. **Environment variable issues**: MLflow needs proper environment variables set for artifact paths
3. **Permission issues**: Container users may not have permission to write to artifact directories
4. **Path inconsistency**: Train and serve containers may use different paths

### Debugging Tools

This repository includes the following tools to help debug and fix MLflow issues:

1. **`make debug-mlflow`**: Command to check MLflow artifact paths, logs, and connectivity
2. **`fix_mlflow.sh`**: Comprehensive debugging script
3. **`src/fix_mlflow_artifacts.py`**: Python script to diagnose MLflow connection and artifact issues
4. **`src/retrain_with_artifacts.py`**: Script to train a model with proper artifact handling

### How to Fix MLflow Artifact Issues

If you encounter "No Artifacts Recorded" or model loading issues:

1. Run the debugging script:
   ```bash
   bash fix_mlflow.sh
   ```

2. If the diagnostic script identifies issues, you can retrain the model with proper artifact handling:
   ```bash
   docker-compose exec app python /app/retrain_with_artifacts.py
   ```

3. If manually creating a model:
   - Make sure to set `MLFLOW_ARTIFACT_ROOT` properly
   - Use `mlflow.log_model()` with an explicit artifact path
   - Verify artifacts are saved by listing them with `MlflowClient().list_artifacts(run_id)`

### MLflow Configuration

For proper artifact handling in Docker, ensure:

1. **Docker Compose Volume Configuration**:
   ```yaml
   volumes:
     - mlflow-artifacts:/mlflow-artifacts
   ```

2. **Environment Variables**:
   ```yaml
   environment:
     - MLFLOW_TRACKING_URI=http://mlflow:5001
     - MLFLOW_ARTIFACT_ROOT=file:///mlflow-artifacts
   ```

3. **MLflow Server Configuration**:
   ```yaml
   command: mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow-data/mlflow.db --default-artifact-root file:///mlflow-artifacts
   ```

4. **Health Check**:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```
