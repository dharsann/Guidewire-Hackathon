import time
import uvicorn
import logging
import threading
import google.generativeai as genai
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from kubernetes import client, config, watch

from src.utils import (
    load_kube_config, 
    detect_anomaly, 
    generate_remediation_suggestion,
    collect_metrics,
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

recent_actions = []

app = FastAPI(title="Minikube Self-Healing Cluster Manager")
genai.configure(api_key="AIzaSyDlp8BfQuAlfCJmrLWbDQQd08Jt8uxJ5ME")

def apply_remediation(action: RemediationAction):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()

    try:
        if action.action_type == "restart_pod":
            v1.delete_namespaced_pod(name=action.target_name, namespace=action.namespace)
            return {"status": "success", "message": f"Pod {action.target_name} restarted."}

        elif action.action_type == "scale_up":
            current = apps_v1.read_namespaced_deployment_scale(action.target_name, action.namespace)
            new_replicas = current.spec.replicas + 1
            body = {"spec": {"replicas": new_replicas}}
            apps_v1.patch_namespaced_deployment_scale(name=action.target_name, namespace=action.namespace, body=body)
            return {"status": "success", "message": f"Scaled up {action.target_name} to {new_replicas} replicas."}

        elif action.action_type == "scale_down":
            current = apps_v1.read_namespaced_deployment_scale(action.target_name, action.namespace)
            new_replicas = max(current.spec.replicas - 1, 1)
            body = {"spec": {"replicas": new_replicas}}
            apps_v1.patch_namespaced_deployment_scale(name=action.target_name, namespace=action.namespace, body=body)
            return {"status": "success", "message": f"Scaled down {action.target_name} to {new_replicas} replicas."}

        else:
            return {"status": "ignored", "message": f"No implementation for action: {action.action_type}"}

    except Exception as e:
        return {"status": "failure", "message": str(e)}

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

        if action.target == "deployment":
            deployment_name = get_deployment_for_pod(input_data.pod_name, input_data.namespace)
            if deployment_name:
                action.target_name = deployment_name

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

    return {"status": "Healthy", "message": "No anomalies detected"}

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
            "recent_actions": recent_actions[-10:]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
    return {"actions": recent_actions}

@app.on_event("startup")
async def startup_event():
    load_kube_config()
    threading.Thread(target=monitor_k8s_events, daemon=True).start()
    logger.info("Self-healing service started")

def monitor_k8s_events():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    w = watch.Watch()
    for event in w.stream(v1.list_event_for_all_namespaces, timeout_seconds=0):
        obj = event['object']
        involved = obj.involved_object
        if involved.kind == "Pod" and any(kw in obj.message for kw in ["BackOff", "Unhealthy", "CrashLoopBackOff"]):
            logger.warning(f"Detected issue: {obj.message}")
            anomaly = "CrashLoopBackOff"
            suggestion = generate_remediation_suggestion(anomaly, involved.name, involved.namespace)
            action = RemediationAction(
                action_type=suggestion["action"],
                target=suggestion["target"],
                target_name=involved.name,
                namespace=involved.namespace,
                parameters={"issue_type": anomaly},
                status="pending",
                message=suggestion["recommendation"]
            )
            if action.target == "deployment":
                deployment_name = get_deployment_for_pod(involved.name, involved.namespace)
                if deployment_name:
                    action.target_name = deployment_name
            result = apply_remediation(action)
            logger.info(f"Auto-remediation result: {result}")
            recent_actions.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": action.dict(),
                "result": result,
                "trigger": "auto-event"
            })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
