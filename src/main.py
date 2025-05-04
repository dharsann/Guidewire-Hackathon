import time
import uvicorn
import logging
import threading
import google.generativeai as genai
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from src.utils import (
    load_kube_config, 
    detect_anomaly, 
    generate_remediation_suggestion,
    apply_remediation,
    collect_metrics,
    scan_cluster,
    get_deployment_for_pod
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minikube-self-healing-api")

class KubernetesMetric(BaseModel):
    timestamp: str
    pod_name: str
    namespace: str
    cpu_allocation_efficiency: float
    memory_allocation_efficiency: float
    disk_io: float
    network_latency: float
    node_temperature: float
    node_cpu_usage: float
    node_memory_usage: float
    event_type: str
    event_message: str
    scaling_event: bool
    pod_lifetime_seconds: float

class RemediationAction(BaseModel):
    action_type: str
    target: str
    target_name: str
    namespace: str
    parameters: Optional[Dict] = None
    status: str = "pending"
    message: str = ""

class ClusterState(BaseModel):
    total_pods: int
    unhealthy_pods: int
    node_states: Dict
    resource_usage: Dict
    metrics: List[Dict]
    recent_actions: List[Dict]

app = FastAPI(title="Minikube Self-Healing Cluster Manager")
genai.configure(api_key="AIzaSyDlp8BfQuAlfCJmrLWbDQQd08Jt8uxJ5ME")

recent_actions = []
last_cluster_scan = {}

@app.post("/self-heal")
async def self_heal(input_data: KubernetesMetric, background_tasks: BackgroundTasks):
    metrics_dict = input_data.dict()

    logger.info("Starting self-heal process")
    
    anomaly = detect_anomaly(metrics_dict)
    logger.info(f"Anomaly detection result: {anomaly}")

    if anomaly:
        suggestion = generate_remediation_suggestion(anomaly, input_data.pod_name, input_data.namespace)
        logger.info(f"Generated suggestion: {suggestion}")

        action = RemediationAction(
            action_type=suggestion["action"],
            target=suggestion["target"],
            target_name=input_data.pod_name,
            namespace=input_data.namespace,
            parameters={"issue_type": anomaly},
            status="pending",
            message=suggestion["recommendation"]
        )

        if action.target == "deployment" and action.target_name == input_data.pod_name:
            deployment_name = get_deployment_for_pod(input_data.pod_name, input_data.namespace)
            if deployment_name:
                action.target_name = deployment_name
                logger.info(f"Mapped pod to deployment: {deployment_name}")

        def remediation_task():
            result = apply_remediation(action)
            action.status = result["status"]
            action.message = result["message"]

            recent_actions.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": action.dict(),
                "metrics": metrics_dict
            })

            logger.info(f"Remediation result: {result}")

        background_tasks.add_task(remediation_task)

        return {
            "status": "Anomaly detected",
            "anomaly": anomaly,
            "remediation_suggestion": suggestion,
            "message": "Remediation task started in background."
        }

    else:
        return {
            "status": "Healthy",
            "message": "No anomalies detected"
        }

@app.get("/cluster-status")
async def get_cluster_status():
    try:
        metrics = collect_metrics()
        
        return {
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cluster_state": {
                "nodes": len(metrics["node_metrics"]),
                "pods": metrics["total_pods"],
                "unhealthy_pods": metrics["unhealthy_pods"]
            },
            "node_metrics": metrics["node_metrics"],
            "recent_issues": metrics["recent_warnings"],
            "recent_actions": recent_actions[-10:] if recent_actions else []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/scan-cluster")
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(scan_cluster, background_tasks, recent_actions)
    return {
        "status": "scanning",
        "message": "Cluster scan started in background"
    }

@app.post("/remediate")
async def manual_remediate(action: RemediationAction):
    result = apply_remediation(action)
    
    recent_actions.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action.dict(),
        "result": result,
        "trigger": "manual"
    })
    
    return {
        "status": result["status"],
        "message": result["message"],
        "action": action.dict()
    }

@app.get("/recent-actions")
async def get_recent_actions():
    return {
        "actions": recent_actions
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_kube_config()
    logger.info("Self-healing service started")

def run_periodic_scan():
    while True:
        try:
            from fastapi import BackgroundTasks
            scan_cluster(BackgroundTasks(), recent_actions)
        except Exception as e:
            logger.error(f"Error in periodic scan: {str(e)}")
        
        time.sleep(300)

if __name__ == "__main__":
    scan_thread = threading.Thread(target=run_periodic_scan, daemon=True)
    scan_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)