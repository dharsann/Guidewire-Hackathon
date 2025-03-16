from fastapi import FastAPI
import numpy as np
import torch
import torch.nn as nn
import joblib
from app.utils import get_active_pods, get_active_nodes, predict_kubernetes_metrics, detect_anomaly, restart_pod, get_active_deployments, scale_deployment_hpa, migrate_pods_from_node, classify_issue_type
from app.models import prediction_model

app = FastAPI()

@app.post("/self_heal")
async def self_heal(data: dict):
    input_features = torch.tensor(data["input_features"], dtype=torch.float32).unsqueeze(0)
    active_deployments = get_active_deployments()
    active_pods = get_active_pods()
    active_nodes = get_active_nodes()
    predicted_values = predict_kubernetes_metrics(input_features)
    anomaly_status = detect_anomaly(predicted_values)
    issue_type = classify_issue_type(predicted_values, data)  
    action = "No action needed"
    if anomaly_status == "Anomaly":
        deployment_name = data.get("deployment_name", active_deployments[0]) 
        pod_name = data.get("pod_name", active_pods[0])
        node_name = data.get("node_name", active_nodes[0])
        if issue_type == "pod_crash":
            action = restart_pod(pod_name)
        elif issue_type == "high_resource_usage":
            action = scale_deployment_hpa(deployment_name)
        elif issue_type == "node_failure":
            action = migrate_pods_from_node(node_name)
        elif issue_type == "network_issue":
            action = restart_pod(pod_name)  
        elif issue_type == "unknown":
            action = "No action taken, needs manual review"
    return {
        "predicted_values": predicted_values,
        "status": anomaly_status,
        "issue_type": issue_type,
        "action": action
    }