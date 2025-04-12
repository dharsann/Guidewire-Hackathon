import subprocess
import google.generativeai as genai
from src.models import prediction_model, anomaly_detection_model, scaler

def get_active_pods():
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-o", "jsonpath={.items[*].metadata.name}"],
            capture_output=True, text=True, check=True
        )
        pods = result.stdout.split()
        return pods if pods else []
    except subprocess.CalledProcessError:
        return []

def get_active_nodes():
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "jsonpath={.items[*].metadata.name}"],
            capture_output=True, text=True, check=True
        )
        nodes = result.stdout.split()
        return nodes if nodes else []
    except subprocess.CalledProcessError:
        return []

def get_active_deployments():
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployments", "-o", "jsonpath={.items[*].metadata.name}"],
            capture_output=True, text=True, check=True
        )
        deployments = result.stdout.split()
        return deployments if deployments else []
    except subprocess.CalledProcessError:
        return []

def restart_pod(pod_name, namespace="default"):
    active_pods = get_active_pods()
    if pod_name not in active_pods:
        return f"Error: Pod {pod_name} not found!"
    try:
        subprocess.run(
            ["kubectl", "delete", "pod", pod_name, "-n", namespace],
            capture_output=True, text=True, check=True
        )
        return f"Restarted pod {pod_name}"
    except subprocess.CalledProcessError as e:
        return f"Failed to restart {pod_name}: {e.stderr.strip()}"

def scale_deployment(deployment_name, replicas):
    try:
        subprocess.run(
            ["kubectl", "scale", "deployment", deployment_name, f"--replicas={replicas}"],
            check=True
        )
        return f"Scaled deployment {deployment_name} to {replicas} replicas"
    except subprocess.CalledProcessError as e:
        return f"Scaling failed: {e.stderr.strip()}"

def scale_deployment_hpa(deployment_name):
    try:
        hpa_status = subprocess.run(
            ["kubectl", "get", "hpa", deployment_name, "-o", "jsonpath={.status.conditions[*].status}"],
            capture_output=True, text=True, check=True
        )
        return f"HPA managing scaling for {deployment_name}" if "True" in hpa_status.stdout else scale_deployment(deployment_name, 5)
    except subprocess.CalledProcessError as e:
        return f"Failed to check HPA: {e.stderr.strip()}"

def migrate_pods_from_node(node_name):
    try:
        subprocess.run(["kubectl", "drain", node_name, "--ignore-daemonsets"], check=True)
        return f"Drained node {node_name}, moving pods to other nodes"
    except subprocess.CalledProcessError as e:
        return f"Failed to drain node {node_name}: {e.stderr.strip()}"

def replace_failed_node(node_name):
    try:
        subprocess.run(["kubectl", "delete", "node", node_name], check=True)
        return f"Deleted failed node {node_name}, allowing cluster autoscaler to replace it"
    except subprocess.CalledProcessError as e:
        return f"Failed to delete node {node_name}: {e.stderr.strip()}"

def rollback_deployment(deployment_name):
    try:
        subprocess.run(["kubectl", "rollout", "undo", "deployment", deployment_name], check=True)
        return f"Rolled back deployment {deployment_name} to previous version"
    except subprocess.CalledProcessError as e:
        return f"Rollback failed: {e.stderr.strip()}"

def classify_issue_type(predicted_values, actual_data):
    cpu, memory, disk_io = predicted_values[0]
    if cpu > 0.9 or memory > 0.9:
        return "high_resource_usage"
    if actual_data.get("pod_lifetime_seconds", 100) < 60:
        return "pod_crash"
    if actual_data.get("event_type") in ["Warning", "Error"]:
        return "node_failure"
    if actual_data.get("network_latency", 0) > 100:
        return "network_issue"
    return "unknown"

def generate_remediation_suggestion(issue_type, pod_name="N/A", node_name="N/A", deployment_name="N/A"):
    try:
        prompt = f"""
        Kubernetes cluster failure detected.
        Issue Type: {issue_type}
        Affected Pod: {pod_name}
        Affected Node: {node_name}
        Affected Deployment: {deployment_name}

        Suggest a remediation strategy for this issue.
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Failed to generate remediation suggestion: {str(e)}"
