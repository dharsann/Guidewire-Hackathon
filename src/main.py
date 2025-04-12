import asyncio
import time
import torch
import google.generativeai as genai
import logging
import prometheus_client
from prometheus_client import Gauge, generate_latest
from fastapi import FastAPI, HTTPException
from src.utils import (
    get_active_pods, get_active_nodes, restart_pod, get_active_deployments,
    scale_deployment_hpa, migrate_pods_from_node, classify_issue_type, generate_remediation_suggestion
)
from src.models import predict_kubernetes_metrics, detect_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

genai.configure(api_key="AIzaSyDlp8BfQuAlfCJmrLWbDQQd08Jt8uxJ5ME")

prediction_metric = Gauge('k8s_prediction', 'Predicted Kubernetes issue type', ['issue_type'])
remediation_metric = Gauge('k8s_remediation', 'Remediation action taken', ['action'])

async def monitor_kubernetes():
    while True:
        try:
            active_pods = get_active_pods()
            active_deployments = get_active_deployments()
            active_nodes = get_active_nodes()
            pod_name = active_pods[0] if active_pods else "unknown"
            deployment_name = active_deployments[0] if active_deployments else "unknown"
            node_name = active_nodes[0] if active_nodes else "unknown"
            input_features = torch.rand(1, 6)  
            predicted_values = predict_kubernetes_metrics(input_features)
            anomaly_status = detect_anomaly(predicted_values)
            issue_type = classify_issue_type(predicted_values, actual_data={})
            prediction_metric.labels(issue_type=issue_type).set(1)
            action = "No action needed"
            if anomaly_status == "Anomaly":
                logger.info(f"Anomaly detected: {issue_type}")
                if issue_type == "pod_crash":
                    logger.info(f"Restarting pod: {pod_name}")
                    action = restart_pod(pod_name)
                elif issue_type == "high_resource_usage":
                    logger.info(f"Scaling deployment: {deployment_name}")
                    action = scale_deployment_hpa(deployment_name)
                elif issue_type == "node_failure":
                    logger.info(f"Migrating pods from node: {node_name}")
                    action = migrate_pods_from_node(node_name)
                elif issue_type == "network_issue":
                    logger.info(f"Restarting pod due to network issue: {pod_name}")
                    action = restart_pod(pod_name)
                elif issue_type == "unknown":
                    logger.warning("Unknown issue detected. Requesting AI analysis.")
                    action = "No automated action taken, needs manual review."
                remediation_metric.labels(action=action).set(1)
            ai_suggestion = generate_remediation_suggestion(issue_type, pod_name, node_name, deployment_name)
            logger.info(f"AI Suggestion: {ai_suggestion}")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
        await asyncio.sleep(30)  

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_kubernetes())

@app.get("/metrics")
def get_metrics():
    return generate_latest()

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "Self-healing Kubernetes API is running!"}
