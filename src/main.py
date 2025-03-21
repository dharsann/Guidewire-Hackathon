from fastapi import FastAPI, HTTPException
import torch
from src.utils import (
    get_active_pods, get_active_nodes, restart_pod, get_active_deployments,
    scale_deployment_hpa, migrate_pods_from_node, classify_issue_type
)
from src.models import predict_kubernetes_metrics, detect_anomaly

app = FastAPI()

@app.post("/self_heal")
async def self_heal(data: dict):
    try:
        if "input_features" not in data:
            raise HTTPException(status_code=400, detail="Missing required field: input_features")
        input_features = torch.tensor(data["input_features"], dtype=torch.float32).unsqueeze(0)
        try:
            active_pods = get_active_pods()
            active_deployments = get_active_deployments()
            active_nodes = get_active_nodes()
        except Exception as e:
            active_pods = []
            active_deployments = []
            active_nodes = []
        pod_name = data.get("pod_name")
        if not pod_name and active_pods:
            pod_name = active_pods[0]
        elif not pod_name:
            pod_name = "unknown"
        deployment_name = data.get("deployment_name")
        if not deployment_name and active_deployments:
            deployment_name = active_deployments[0]
        elif not deployment_name:
            deployment_name = "unknown"    
        node_name = data.get("node_name")
        if not node_name and active_nodes:
            node_name = active_nodes[0]
        elif not node_name:
            node_name = "unknown"
        predicted_values = predict_kubernetes_metrics(input_features)
        anomaly_status = detect_anomaly(predicted_values)
        issue_type = classify_issue_type(predicted_values, data)
        action = "No action needed"
        if anomaly_status == "Anomaly":
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
        else:
            return {
                "predicted_values": predicted_values,
                "status": anomaly_status,
                "issue_type": issue_type,
                "action": action
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")