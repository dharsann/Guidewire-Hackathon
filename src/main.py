import time
import logging
import threading
import google.generativeai as genai
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# Import functions from utility module
from src.utils import (
    load_kube_config, 
    detect_anomaly, 
    generate_remediation_suggestion,
    apply_remediation,
    collect_metrics,
    scan_cluster,
    get_deployment_for_pod
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minikube-self-healing-api")

# Model definitions
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

# Initialize FastAPI app
app = FastAPI(title="Minikube Self-Healing Cluster Manager")
genai.configure(api_key="AIzaSyDlp8BfQuAlfCJmrLWbDQQd08Jt8uxJ5ME")

# Global variables to track state
recent_actions = []
last_cluster_scan = {}

# API Endpoints
@app.post("/self-heal")
async def self_heal(input_data: KubernetesMetric, background_tasks: BackgroundTasks):
    """Process incoming metrics and apply self-healing if needed."""
    metrics_dict = input_data.dict()

    logger.info("Starting self-heal process")
    
    anomaly = detect_anomaly(metrics_dict)
    logger.info(f"Anomaly detection result: {anomaly}")

    if anomaly:
        suggestion = generate_remediation_suggestion(anomaly, input_data.pod_name, input_data.namespace)
        logger.info(f"Generated suggestion: {suggestion}")

        # Build the action object
        action = RemediationAction(
            action_type=suggestion["action"],
            target=suggestion["target"],
            target_name=input_data.pod_name,
            namespace=input_data.namespace,
            parameters={"issue_type": anomaly},
            status="pending",
            message=suggestion["recommendation"]
        )

        # If targeting a deployment, map pod to deployment
        if action.target == "deployment" and action.target_name == input_data.pod_name:
            deployment_name = get_deployment_for_pod(input_data.pod_name, input_data.namespace)
            if deployment_name:
                action.target_name = deployment_name
                logger.info(f"Mapped pod to deployment: {deployment_name}")

        # Background remediation task
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

        # Add to background
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
    """Get the current status of the cluster."""
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
    """Trigger a manual cluster scan."""
    background_tasks.add_task(scan_cluster, background_tasks, recent_actions)
    return {
        "status": "scanning",
        "message": "Cluster scan started in background"
    }

@app.post("/remediate")
async def manual_remediate(action: RemediationAction):
    """Manually apply a remediation action."""
    result = apply_remediation(action)
    
    # Record the action
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
    """Get the list of recent remediation actions."""
    return {
        "actions": recent_actions
    }

# Background task to periodically scan the cluster
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_kube_config()
    logger.info("Self-healing service started")

# Run periodic scan in a separate thread
def run_periodic_scan():
    """Run periodic cluster scans."""
    while True:
        try:
            from fastapi import BackgroundTasks
            scan_cluster(BackgroundTasks(), recent_actions)
        except Exception as e:
            logger.error(f"Error in periodic scan: {str(e)}")
        
        # Sleep for 5 minutes before next scan
        time.sleep(300)

if __name__ == "__main__":
    import uvicorn
    
    # Start periodic scan in a background thread
    scan_thread = threading.Thread(target=run_periodic_scan, daemon=True)
    scan_thread.start()
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)