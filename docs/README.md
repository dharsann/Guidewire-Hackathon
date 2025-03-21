# Kubernetes Self-Healing Service

This project provides a self-healing service for Kubernetes environments using FastAPI. It leverages machine learning models to predict metrics, detect anomalies, and take corrective actions based on the detected issues. It is under development.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features

- Predicts Kubernetes performance metrics using an LSTM model.
- Detects anomalies in the predicted metrics.
- Classifies the type of issue and takes appropriate actions (e.g., restarting pods, scaling deployments).
- Integrates with Kubernetes to retrieve active pods, deployments, and nodes.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

    Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies:

    pip install -r requirements.txt

    Download Pre-trained Models:
        Ensure the pre-trained LSTM model (lstm_k8s_model_extended.pth) is in the project directory.
        Ensure the Isolation Forest model (isolation_forest.pkl) is in the project directory.

Usage

    Run the FastAPI Application:

    uvicorn your_module_name\:app --reload

    Replace your_module_name with the name of your Python file (without the .py extension).

    Access the API:
        The API will be available at http://127.0.0.1:8000.
        Use tools like Postman or cURL to interact with the API.

API Endpoints

    /self_heal:
        Method: POST
        Description: Endpoint to predict Kubernetes metrics, detect anomalies, and take corrective actions.
        Request Body:

{
  "input_features": {
    "cpu_allocation_efficiency": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6],
    "memory_allocation_efficiency": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5],
    "disk_io": [100, 150, 200, 250, 300, 350, 300, 250, 200, 150],
    "network_latency": [20, 25, 30, 35, 40, 45, 40, 35, 30, 25],
    "node_cpu_usage": [70, 75, 80, 85, 90, 95, 90, 85, 80, 75],
    "node_memory_usage": [50, 55, 60, 65, 70, 75, 70, 65, 60, 55]
  },
  "pod_name": "example-pod",
  "deployment_name": "example-deployment",
  "node_name": "example-node"
}

Response:

        {
          "predicted_values": [...],
          "status": "Anomaly or Normal",
          "issue_type": "Type of issue",
          "action": "Action taken"
        }

Model Training

    The LSTM model is trained to predict Kubernetes performance metrics.
    The Isolation Forest model is used for anomaly detection.
    Ensure the models are trained and saved in the project directory before running the application.
