import subprocess
from src.models import prediction_model, anomaly_detection_model, scaler

def get_active_pods():
    result = subprocess.run(["kubectl", "get", "pods", "-o", "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True)
    pods = result.stdout.split()
    return pods if pods else ["unknown"]

def get_active_nodes():
    result = subprocess.run(["kubectl", "get", "nodes", "-o", "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True)
    nodes = result.stdout.split()
    return nodes if nodes else ["unknown"]

def get_active_deployments():
    result = subprocess.run(["kubectl", "get", "deployments", "-o", "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True)
    deployments = result.stdout.split()
    return deployments if deployments else ["unknown"]

def restart_pod(pod_name, namespace="default"):
    active_pods = get_active_pods()
    if pod_name not in active_pods:
        return f"Error: Pod {pod_name} not found!"
    try:
        result = subprocess.run(
            ["kubectl", "delete", "pod", pod_name, "-n", namespace],
            capture_output=True,
            text=True,
            check=True
        )
        return f"Restarted pod {pod_name}"
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip()
        return f"Failed to restart {pod_name}: {error_message}"

def scale_deployment(deployment_name, replicas):
    subprocess.run(["kubectl", "scale", "deployment", deployment_name, f"--replicas={replicas}"], check=True)
    return f"Scaled deployment {deployment_name} to {replicas} replicas"

def scale_deployment_hpa(deployment_name):
    hpa_status = subprocess.run(
        ["kubectl", "get", "hpa", deployment_name, "-o", "jsonpath={.status.conditions[*].status}"],
        capture_output=True, text=True
    )
    return f"HPA managing scaling for {deployment_name}" if "True" in hpa_status.stdout else scale_deployment(deployment_name, 5)

def migrate_pods_from_node(node_name):
    subprocess.run(["kubectl", "drain", node_name, "--ignore-daemonsets"], check=True)
    return f"Drained node {node_name}, moving pods to other nodes"

def replace_failed_node(node_name):
    subprocess.run(["kubectl", "delete", "node", node_name], check=True)
    return f"Deleted failed node {node_name}, allowing cluster autoscaler to replace it"

def rollback_deployment(deployment_name):
    subprocess.run(["kubectl", "rollout", "undo", "deployment", deployment_name], check=True)
    return f"Rolled back deployment {deployment_name} to previous version"

def classify_issue_type(predicted_values, actual_data):
    cpu, memory, disk_io = predicted_values[0]
    if cpu > 0.9 or memory > 0.9:
        return "high_resource_usage"
    if actual_data["pod_lifetime_seconds"] < 60:
        return "pod_crash"
    if actual_data["event_type"] in ["Warning", "Error"]:
        return "node_failure"
    if actual_data["network_latency"] > 100:
        return "network_issue"
    return "unknown"
