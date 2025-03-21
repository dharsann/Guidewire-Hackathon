# Kubernetes Self-Healing Service

This project provides a self-healing service for Kubernetes environments using FastAPI. It leverages machine learning models to predict metrics, detect anomalies, and take corrective actions based on detected issues. The project is currently under development.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Data Source](#data-source)
- [Contributing](#contributing)
- [License](#license)

## Features

- Predicts Kubernetes performance metrics using an LSTM model.
- Detects anomalies in the predicted metrics.
- Classifies the type of issue and takes appropriate actions (e.g., restarting pods, scaling deployments).
- Integrates with Kubernetes to retrieve active pods, deployments, and nodes.

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
Ensure the following pre-trained models are in the project directory:
- **LSTM Model:** `lstm_k8s_model_extended.pth`
- **Isolation Forest Model:** `isolation_forest.pkl`

## Usage

### Run the FastAPI Application
```bash
uvicorn your_module_name:app --reload
```
Replace `your_module_name` with the name of your Python file (without the `.py` extension).

### Access the API
- The API will be available at **http://127.0.0.1:8000**.
- Use tools like **Postman** or `cURL` to interact with the API.

## API Endpoints

### `/self_heal`
- **Method:** `POST`
- **Description:** Endpoint to predict Kubernetes metrics, detect anomalies, and take corrective actions.
- **Request Body:**
    ```json
    {
        "input_features": [0.8, 0.7, 400],  
        "pod_lifetime_seconds": 45,
        "event_type": "Warning",
        "pod_name": "nginx-pod",
        "node_name": "minikube"
    }
    ```
- **Response:**
    ```json
    {
        "predicted_values": [...],
        "status": "Anomaly or Normal",
        "issue_type": "Type of issue",
        "action": "Action taken"
    }
    ```

## Model Training

- The **LSTM model** is trained to predict Kubernetes performance metrics.
- The **Isolation Forest model** is used for anomaly detection.
- Ensure the models are trained and saved in the project directory before running the application.

## Data Source

The dataset used for training the models is sourced from Kaggle. It contains performance metrics and other relevant data from Kubernetes environments. The dataset is preprocessed and used to train the LSTM model for predicting future metrics and the Isolation Forest model for detecting anomalies.

