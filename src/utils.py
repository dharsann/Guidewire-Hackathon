import time
import logging
import json
import subprocess
import google.generativeai as genai
from typing import Dict, List, Optional
from kubernetes import client, config
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minikube-cluster-utils")

# For type hints
class RemediationAction(BaseModel):
    action_type: str
    target: str
    target_name: str
    namespace: str
    parameters: Optional[Dict] = None
    status: str = "pending"
    message: str = ""

def load_kube_config():
    """Load Kubernetes configuration for in-cluster or local development."""
    try:
        # Try to load in-cluster config (when running in a pod)
        config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes configuration")
    except config.config_exception.ConfigException:
        # Fall back to local config
        config.load_kube_config()
        logger.info("Loaded local Kubernetes configuration")

def run_kubectl_command(command: List[str]) -> str:
    """Run kubectl command and get output."""
    try:
        logger.debug(f"Running kubectl command: {' '.join(command)}")
        output = subprocess.check_output(['kubectl'] + command, stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running kubectl command: {e.output.decode('utf-8')}"
        logger.error(error_msg)
        return error_msg

def detect_anomaly(metrics: Dict) -> Optional[str]:
    """Detect anomalies in cluster metrics."""
    # CPU efficiency too low
    if metrics["cpu_allocation_efficiency"] < 0.3:
        return "low_cpu_efficiency"
    # Memory efficiency too low
    if metrics["memory_allocation_efficiency"] < 0.3:
        return "low_memory_efficiency"
    # High resource usage
    if metrics["node_cpu_usage"] > 85 or metrics["node_memory_usage"] > 85:
        return "high_resource_usage"
    # High disk IO
    if metrics["disk_io"] > 90:
        return "high_disk_io"
    # Network latency issues
    if metrics["network_latency"] > 200:  # ms
        return "network_issue"
    # Node temperature too high
    if metrics["node_temperature"] > 75:  # arbitrary threshold
        return "high_node_temperature"
    # Unusual event reported
    if metrics["event_type"] and metrics["event_type"] != "Normal":
        return "abnormal_event"
    # Pod crash or restart
    if "CrashLoopBackOff" in metrics.get("event_message", ""):
        return "pod_crash"
    
    return None

def generate_remediation_suggestion(anomaly_type: str, pod_name: str = "", namespace: str = "default", extra_context: Optional[Dict] = None) -> Dict:
    """Use Gemini to generate a remediation suggestion."""
    prompt = f"""
    A Kubernetes anomaly was detected:
    - Anomaly Type: {anomaly_type}
    - Pod Name: {pod_name}
    - Namespace: {namespace}
    - Additional Context: {json.dumps(extra_context or {}, indent=2)}

    As a Kubernetes expert, suggest a remediation action in this format:
    {{
        "action": "<short_action>",
        "target": "<target_type>",
        "recommendation": "<detailed human-readable suggestion>"
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        suggestion = json.loads(response.text.strip())
        return suggestion
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {
            "action": "manual_review",
            "target": "cluster",
            "recommendation": "Gemini failed to generate a suggestion. Manual review needed."
        }

def scale_deployment(deployment_name: str, replicas: int, namespace: str = "default") -> str:
    """Scale a deployment to specified number of replicas."""
    try:
        logger.info(f"Scaling deployment {deployment_name} to {replicas} replicas in namespace {namespace}")
        apps_v1 = client.AppsV1Api()
        body = {'spec': {'replicas': replicas}}
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        return f"Scaled deployment '{deployment_name}' to {replicas} replicas"
    except Exception as e:
        error_msg = f"Error scaling deployment: {str(e)}"
        logger.error(error_msg)
        return error_msg

def restart_pod(pod_name: str, namespace: str = "default") -> str:
    """Restart a pod by deleting it (K8s will recreate it)."""
    try:
        logger.info(f"Restarting pod {pod_name} in namespace {namespace}")
        v1 = client.CoreV1Api()
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
        return f"Pod '{pod_name}' deleted. It will be recreated automatically."
    except Exception as e:
        error_msg = f"Error restarting pod: {str(e)}"
        logger.error(error_msg)
        return error_msg

def cordon_node(node_name: str) -> str:
    """Cordon a node to prevent new pods from being scheduled on it."""
    try:
        logger.info(f"Cordoning node {node_name}")
        return run_kubectl_command(['cordon', node_name])
    except Exception as e:
        error_msg = f"Error cordoning node: {str(e)}"
        logger.error(error_msg)
        return error_msg

def drain_node(node_name: str) -> str:
    """Drain a node to relocate pods."""
    try:
        logger.info(f"Draining node {node_name}")
        return run_kubectl_command(['drain', node_name, '--ignore-daemonsets', '--delete-emptydir-data'])
    except Exception as e:
        error_msg = f"Error draining node: {str(e)}"
        logger.error(error_msg)
        return error_msg

def get_deployment_for_pod(pod_name: str, namespace: str = "default") -> Optional[str]:
    """Find the deployment that manages a pod."""
    try:
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        
        # Check for owner references
        if pod.metadata.owner_references:
            for owner in pod.metadata.owner_references:
                if owner.kind == "ReplicaSet":
                    # Get the ReplicaSet to find its owner deployment
                    apps_v1 = client.AppsV1Api()
                    rs = apps_v1.read_namespaced_replica_set(
                        name=owner.name, 
                        namespace=namespace
                    )
                    if rs.metadata.owner_references:
                        for rs_owner in rs.metadata.owner_references:
                            if rs_owner.kind == "Deployment":
                                return rs_owner.name
        return None
    except Exception as e:
        logger.error(f"Error finding deployment for pod: {str(e)}")
        return None

def get_node_metrics() -> Dict:
    """Get metrics for all nodes."""
    try:
        # Try to get metrics using kubectl top
        output = run_kubectl_command(['top', 'nodes', '--no-headers'])
        
        metrics = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                node_name = parts[0]
                cpu_usage = parts[1].rstrip('%')
                memory_usage = parts[3].rstrip('%')
                
                metrics[node_name] = {
                    'cpu_usage_percent': float(cpu_usage),
                    'memory_usage_percent': float(memory_usage),
                }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting node metrics: {str(e)}")
        return {}

def get_pod_metrics(namespace: str = "") -> Dict:
    """Get metrics for all pods, optionally filtered by namespace."""
    try:
        cmd = ['top', 'pods', '--no-headers']
        if namespace:
            cmd.extend(['-n', namespace])
            
        output = run_kubectl_command(cmd)
        
        metrics = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                pod_name = parts[0]
                cpu_usage = parts[1].rstrip('m')
                memory_usage = parts[2].rstrip('Mi')
                
                metrics[pod_name] = {
                    'cpu_usage_millicores': float(cpu_usage),
                    'memory_usage_mb': float(memory_usage),
                }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting pod metrics: {str(e)}")
        return {}

def get_pod_statuses(namespace: str = "") -> List[Dict]:
    """Get status information for all pods."""
    try:
        v1 = client.CoreV1Api()
        
        if namespace:
            pods = v1.list_namespaced_pod(namespace)
        else:
            pods = v1.list_pod_for_all_namespaces()
            
        pod_info = []
        for pod in pods.items:
            # Calculate age in seconds
            creation_time = pod.metadata.creation_timestamp
            if creation_time:
                age_seconds = (time.time() - creation_time.timestamp())
            else:
                age_seconds = 0
                
            # Get container statuses
            restart_count = 0
            container_statuses = []
            
            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    restart_count += cs.restart_count
                    container_statuses.append({
                        'name': cs.name,
                        'ready': cs.ready,
                        'restarts': cs.restart_count,
                        'state': next(iter(cs.state.to_dict().keys()), 'unknown')
                    })
            
            pod_info.append({
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "status": pod.status.phase,
                "node": pod.spec.node_name if pod.spec.node_name else "unknown",
                "age_seconds": age_seconds,
                "restarts": restart_count,
                "containers": container_statuses
            })
            
        return pod_info
    except Exception as e:
        logger.error(f"Error getting pod statuses: {str(e)}")
        return [{"error": str(e)}]

def get_events(namespace: str = "") -> List[Dict]:
    """Get recent Kubernetes events."""
    try:
        v1 = client.CoreV1Api()
        
        if namespace:
            events = v1.list_namespaced_event(namespace)
        else:
            events = v1.list_event_for_all_namespaces()
            
        event_list = []
        for event in events.items:
            # Calculate how recent the event is
            last_timestamp = event.last_timestamp
            if last_timestamp:
                seconds_ago = (time.time() - last_timestamp.timestamp())
            else:
                seconds_ago = 0
                
            # Only include events from the last hour
            if seconds_ago <= 3600:
                event_list.append({
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "involved_object": {
                        "kind": event.involved_object.kind,
                        "name": event.involved_object.name,
                        "namespace": event.involved_object.namespace
                    },
                    "count": event.count,
                    "seconds_ago": seconds_ago
                })
                
        return event_list
    except Exception as e:
        logger.error(f"Error getting events: {str(e)}")
        return [{"error": str(e)}]

def apply_remediation(action: RemediationAction) -> Dict:
    """Apply remediation action based on the action type."""
    result = {"status": "failed", "message": "Unknown action type"}
    
    try:
        if action.action_type == "restart_pod":
            result["message"] = restart_pod(action.target_name, action.namespace)
            result["status"] = "completed"
            
        elif action.action_type == "scale_up":
            # Get current replicas and increase by 1
            deployment_name = action.target_name
            if not deployment_name and action.parameters and "pod_name" in action.parameters:
                # Find deployment that owns this pod
                deployment_name = get_deployment_for_pod(
                    action.parameters["pod_name"], 
                    action.namespace
                )
                
            if deployment_name:
                apps_v1 = client.AppsV1Api()
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=action.namespace
                )
                current_replicas = deployment.spec.replicas
                new_replicas = current_replicas + 1
                
                result["message"] = scale_deployment(
                    deployment_name, 
                    new_replicas, 
                    action.namespace
                )
                result["status"] = "completed"
            else:
                result["message"] = "Could not find deployment to scale"
                
        elif action.action_type == "scale_down":
            # Get current replicas and decrease by 1 (min 1)
            deployment_name = action.target_name
            if not deployment_name and action.parameters and "pod_name" in action.parameters:
                # Find deployment that owns this pod
                deployment_name = get_deployment_for_pod(
                    action.parameters["pod_name"], 
                    action.namespace
                )
                
            if deployment_name:
                apps_v1 = client.AppsV1Api()
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=action.namespace
                )
                current_replicas = deployment.spec.replicas
                new_replicas = max(1, current_replicas - 1)
                
                result["message"] = scale_deployment(
                    deployment_name, 
                    new_replicas, 
                    action.namespace
                )
                result["status"] = "completed"
            else:
                result["message"] = "Could not find deployment to scale"
                
        elif action.action_type == "cordon_node":
            result["message"] = cordon_node(action.target_name)
            result["status"] = "completed"
            
        elif action.action_type == "drain_node":
            result["message"] = drain_node(action.target_name)
            result["status"] = "completed"
            
        elif action.action_type in ["check_network", "optimize_storage", "manual_review", "investigate_event"]:
            result["message"] = f"Action '{action.action_type}' requires manual intervention"
            result["status"] = "manual_action_required"
            
        else:
            result["message"] = f"Unsupported action type: {action.action_type}"
    
    except Exception as e:
        result["message"] = f"Error applying remediation: {str(e)}"
        logger.error(result["message"])
        
    return result

def collect_metrics() -> Dict:
    """Collect comprehensive metrics about the cluster."""
    node_metrics = get_node_metrics()
    pod_statuses = get_pod_statuses()
    events = get_events()
    
    # Count unhealthy pods
    unhealthy_pods = sum(1 for pod in pod_statuses if pod["status"] != "Running")
    
    # Find pods with high restart counts
    high_restart_pods = [pod for pod in pod_statuses if pod.get("restarts", 0) > 5]
    
    # Find recent warning/error events
    recent_warnings = [e for e in events if e["type"] != "Normal" and e.get("seconds_ago", 0) < 600]
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node_metrics": node_metrics,
        "pod_statuses": pod_statuses,
        "events": events,
        "unhealthy_pods": unhealthy_pods,
        "high_restart_pods": high_restart_pods,
        "recent_warnings": recent_warnings,
        "total_pods": len(pod_statuses)
    }

def scan_cluster(background_tasks, recent_actions: List[Dict]):
    """Scan the cluster for issues and apply remediations automatically."""
    try:
        logger.info("Starting cluster scan")
        
        # Collect comprehensive metrics
        metrics = collect_metrics()
        
        # Check for issues
        issues_found = []
        
        # Check for pods not in Running state
        for pod in metrics["pod_statuses"]:
            if pod["status"] != "Running" and pod["status"] != "Succeeded":
                issue = {
                    "type": "unhealthy_pod",
                    "severity": "high",
                    "target": "pod",
                    "target_name": pod["name"],
                    "namespace": pod["namespace"],
                    "message": f"Pod {pod['name']} is in {pod['status']} state"
                }
                issues_found.append(issue)
        
        # Check for pods with high restart counts
        for pod in metrics["high_restart_pods"]:
            issue = {
                "type": "pod_crash",
                "severity": "medium",
                "target": "pod",
                "target_name": pod["name"],
                "namespace": pod["namespace"],
                "message": f"Pod {pod['name']} has {pod['restarts']} restarts"
            }
            issues_found.append(issue)
        
        # Check for node resource usage
        for node_name, node_data in metrics["node_metrics"].items():
            if node_data["cpu_usage_percent"] > 80:
                issue = {
                    "type": "high_resource_usage",
                    "severity": "medium",
                    "target": "node",
                    "target_name": node_name,
                    "namespace": "",
                    "message": f"Node {node_name} has high CPU usage: {node_data['cpu_usage_percent']}%"
                }
                issues_found.append(issue)
                
            if node_data["memory_usage_percent"] > 80:
                issue = {
                    "type": "high_resource_usage",
                    "severity": "medium",
                    "target": "node",
                    "target_name": node_name,
                    "namespace": "",
                    "message": f"Node {node_name} has high memory usage: {node_data['memory_usage_percent']}%"
                }
                issues_found.append(issue)
        
        # Process each issue
        for issue in issues_found:
            # Generate remediation suggestion
            remediation = generate_remediation_suggestion(issue["type"], issue["target_name"], issue["namespace"])
            
            # Create remediation action
            action = RemediationAction(
                action_type=remediation["action"],
                target=remediation["target"],
                target_name=issue["target_name"],
                namespace=issue["namespace"],
                parameters={"issue_type": issue["type"]},
                status="pending",
                message=remediation["recommendation"]
            )
            
            # Apply the remediation
            result = apply_remediation(action)
            
            # Update action with result
            action.status = result["status"]
            action.message = result["message"]
            
            # Add to recent actions
            recent_actions.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": action.dict(),
                "issue": issue
            })
            
            # Keep only the most recent 50 actions
            if len(recent_actions) > 50:
                del recent_actions[0:-50]
            
            logger.info(f"Applied remediation for {issue['type']} on {issue['target']} {issue['target_name']}: {result['status']}")
            
        logger.info(f"Cluster scan completed. Found {len(issues_found)} issues.")
        
    except Exception as e:
        logger.error(f"Error during cluster scan: {str(e)}")