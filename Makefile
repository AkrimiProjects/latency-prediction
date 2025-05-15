# Variables
PYTHON = python
PIP = pip
VENV = venv
SRC_DIR = src
TEST_DIR = tests

# Commandes virtuelles
.PHONY: all setup clean lint format security test coverage docker-build docker-push mlflow-track deploy monitor watch-ci

# Configuration variables
DOCKER_USERNAME ?= $(shell echo $$DOCKER_USERNAME)
IMAGE_NAME = ml-app
VERSION ?= $(shell git describe --tags --always --dirty)
DOCKER_IMAGE = $(DOCKER_USERNAME)/$(IMAGE_NAME):$(VERSION)

# MLflow configuration
MLFLOW_TRACKING_URI ?= http://localhost:5000
EXPERIMENT_NAME ?= ml-experiment

# Cible par défaut
all: setup lint format security test docker-build docker-push deploy

# Configuration de l'environnement virtuel
setup:
	if not exist $(VENV) $(PYTHON) -m venv $(VENV)
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pip install --upgrade pip}"
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pip install -r requirements.txt}"
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pip install black flake8 pylint mypy bandit pytest pytest-cov safety watchdog}"

# Nettoyage
clean:
	if exist $(VENV) rmdir /s /q $(VENV)
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .coverage del /f .coverage
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist .tox rmdir /s /q .tox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

# Vérification de la qualité du code
lint:
	@echo Running Flake8...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; flake8 $(SRC_DIR)}"
	@echo Running Pylint...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pylint $(SRC_DIR)}"
	@echo Running MyPy...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; mypy $(SRC_DIR)}"

# Formatage du code
format:
	@echo Formatting code with Black...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; black $(SRC_DIR)}"

# Vérification de la sécurité
security:
	@echo Running Bandit security checks...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; bandit -r $(SRC_DIR)}"
	@echo Checking dependencies for known security vulnerabilities...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; safety check}"

# Tests unitaires
test:
	@echo Running tests...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pytest $(TEST_DIR) -v}"

# Couverture des tests
coverage:
	@echo Generating coverage report...
	powershell -Command "& {. .\$(VENV)\Scripts\Activate.ps1; pytest --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=term-missing --cov-report=html}"

# Docker build
docker-build:
	@echo "Building Docker image $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$(shell git rev-parse HEAD) \
		.

# Docker push
docker-push:
	@echo "Pushing Docker image $(DOCKER_IMAGE)"
	docker push $(DOCKER_IMAGE)

# MLflow track
mlflow-track:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	python train_and_track.py --experiment-name $(EXPERIMENT_NAME)

# Deploy
deploy:
	docker-compose up -d

# Monitor
monitor:
	@echo "Starting monitoring stack..."
	docker-compose -f docker-compose.monitoring.yml up -d

# Watch CI
watch-ci:
	python ci_watcher.py

# Helper targets
setup-dev:
	pip install -r requirements-dev.txt
	pre-commit install

setup-hooks:
	cp hooks/* .git/hooks/
	chmod +x .git/hooks/*

check-security: security
	@echo "Running additional security checks..."
	docker scan $(DOCKER_IMAGE)
	trivy image $(DOCKER_IMAGE)

# CD pipeline
cd-pipeline: docker-build docker-push mlflow-track deploy monitor
	@echo "CD pipeline completed successfully"

# Aide
help:
	@echo "Available targets:"
	@echo "  all          - Run complete pipeline (clean, lint, test, security, docker-build, docker-push, deploy)"
	@echo "  clean        - Clean up temporary files"
	@echo "  lint         - Run code linting"
	@echo "  test         - Run tests with coverage"
	@echo "  security     - Run security checks"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-push  - Push Docker image to registry"
	@echo "  mlflow-track - Track ML experiment with MLflow"
	@echo "  deploy       - Deploy application"
	@echo "  monitor      - Start monitoring stack"
	@echo "  watch-ci     - Start CI file watcher"
	@echo "  cd-pipeline  - Run complete CD pipeline" 