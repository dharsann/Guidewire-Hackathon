import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from kubernetes import client, config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    config.load_incluster_config()
except config.ConfigException:
    try:
        config.load_kube_config()
    except config.ConfigException:
        logger.warning("Could not configure Kubernetes client. Using limited functionality.")

try:
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    autoscaling_api = client.AutoscalingV1Api()
except NameError:
    logger.warning("Kubernetes client not fully initialized")

def get_active_pods() -> List[str]:
    try:
        pods = core_api.list_pod_for_all_namespaces(watch=False)
        return [f"{pod.metadata.namespace}/{pod.metadata.name}" for pod in pods.items]
    except Exception as e:
        logger.error(f"Error getting pods: {str(e)}")
        return []

def get_active_nodes() -> List[str]:
    try:
        nodes = core_api.list_node(watch=False)
        return [node.metadata.name for node in nodes.items]
    except Exception as e:
        logger.error(f"Error getting nodes: {str(e)}")
        return []

def get_active_deployments() -> List[str]:
    try:
        deployments = apps_api.list_deployment_for_all_namespaces(watch=False)
        return [f"{deploy.metadata.namespace}/{deploy.metadata.name}" for deploy in deployments.items]
    except Exception as e:
        logger.error(f"Error getting deployments: {str(e)}")
        return []

def restart_pod(pod_name: str) -> str:
    try:
        if "/" in pod_name:
            namespace, name = pod_name.split("/", 1)
        else:
            namespace = "default"
            name = pod_name
        core_api.delete_namespaced_pod(name=name, namespace=namespace)
        
        logger.info(f"Pod {namespace}/{name} deleted for restart")
        return f"Pod {namespace}/{name} restarted successfully"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to restart pod {pod_name}: {error_msg}")
        return f"Failed to restart pod {pod_name}: {error_msg}"

def scale_deployment_hpa(deployment_name: str, replicas: Optional[int] = None) -> str:
    try:
        if "/" in deployment_name:
            namespace, name = deployment_name.split("/", 1)
        else:
            namespace = "default"
            name = deployment_name
        
        try:
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(name=name, namespace=namespace)
            
            if replicas is not None:
                hpa.spec.min_replicas = max(1, replicas - 1)
                hpa.spec.max_replicas = replicas + 2
                autoscaling_api.replace_namespaced_horizontal_pod_autoscaler(
                    name=name,
                    namespace=namespace,
                    body=hpa
                )
                return f"HPA for {namespace}/{name} updated with min={hpa.spec.min_replicas}, max={hpa.spec.max_replicas}"
            else:
                return f"HPA for {namespace}/{name} exists with min={hpa.spec.min_replicas}, max={hpa.spec.max_replicas}"
                
        except client.exceptions.ApiException as api_e:
            if api_e.status == 404:
                if replicas is None:
                    deployment = apps_api.read_namespaced_deployment(name=name, namespace=namespace)
                    current = deployment.spec.replicas
                    replicas = max(current + 1, 2)  
                
                apps_api.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body={"spec": {"replicas": replicas}}
                )
                
                return f"Deployment {namespace}/{name} scaled to {replicas} replicas"
            else:
                raise
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to scale deployment {deployment_name}: {error_msg}")
        return f"Failed to scale deployment {deployment_name}: {error_msg}"

def migrate_pods_from_node(node_name: str) -> str:
    try:
        body = {
            "spec": {
                "unschedulable": True
            }
        }
        core_api.patch_node(name=node_name, body=body)
        logger.info(f"Node {node_name} cordoned")
        
        field_selector = f"spec.nodeName={node_name}"
        pods = core_api.list_pod_for_all_namespaces(watch=False, field_selector=field_selector)
        
        for pod in pods.items:
            if pod.metadata.owner_references and any(
                ref.kind == "DaemonSet" for ref in pod.metadata.owner_references
            ):
                logger.info(f"Skipping DaemonSet pod {pod.metadata.namespace}/{pod.metadata.name}")
                continue
                
            try:
                core_api.delete_namespaced_pod(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=60,
                        propagation_policy='Foreground'
                    )
                )
                logger.info(f"Deleted pod {pod.metadata.namespace}/{pod.metadata.name} from node {node_name}")
            except Exception as pod_e:
                logger.error(f"Error deleting pod {pod.metadata.namespace}/{pod.metadata.name}: {str(pod_e)}")
        
        return f"Node {node_name} cordoned and pods migrated"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to migrate pods from node {node_name}: {error_msg}")
        return f"Failed to migrate pods from node {node_name}: {error_msg}"

def classify_issue_type(model_predictions: Any, actual_data: Dict[str, Any]) -> str:
    import torch
    
    if not isinstance(model_predictions, torch.Tensor):
        logger.warning("model_predictions is not a tensor")
        return "unknown"
    
    if len(model_predictions.shape) > 1:
        pred_values = model_predictions[0] 
    else:
        pred_values = model_predictions
    
    pred_array = pred_values.cpu().numpy()
    
    metrics = {}
    metric_names = ["cpu_usage", "memory_usage", "network_latency", "disk_io", "error_rate"]
    for i, name in enumerate(metric_names):
        if i < len(pred_array):
            metrics[name] = float(pred_array[i])
        else:
            metrics[name] = 0.0
    
    if metrics.get("error_rate", 0) > 0.8:
        return "pod_crash"
    elif metrics.get("cpu_usage", 0) > 0.85 or metrics.get("memory_usage", 0) > 0.9:
        return "high_resource_usage"
    elif metrics.get("disk_io", 0) < 0.2 and metrics.get("cpu_usage", 0) < 0.1:
        return "node_failure"
    elif metrics.get("network_latency", 0) > 0.7:
        return "network_issue"
    else:
        return "unknown"

def generate_remediation_suggestion(
    issue_type: str, 
    pod_name: Optional[str] = None, 
    node_name: Optional[str] = None, 
    deployment_name: Optional[str] = None
) -> str:
    try:
        context = f"Issue type: {issue_type}\n"
        if pod_name and pod_name != "unknown":
            context += f"Pod: {pod_name}\n"
        if node_name and node_name != "unknown":
            context += f"Node: {node_name}\n"
        if deployment_name and deployment_name != "unknown":
            context += f"Deployment: {deployment_name}\n"
            
        if issue_type == "pod_crash":
            context += "The pod is repeatedly crashing, potentially due to application errors or resource constraints."
        elif issue_type == "high_resource_usage":
            context += "The resource usage (CPU or memory) is exceeding healthy thresholds."
        elif issue_type == "node_failure":
            context += "The node appears to be unresponsive or facing hardware/connectivity issues."
        elif issue_type == "network_issue":
            context += "Network connectivity or latency issues are affecting communication."
        
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 256,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        
        prompt = f"""
        You are a Kubernetes expert tasked with providing remediation suggestions for issues in a Kubernetes cluster.
        Please provide a concise, actionable remediation suggestion for the following issue:
        
        {context}
        
        Your suggestion should:
        1. Be specific and actionable
        2. Include commands or steps where appropriate
        3. Suggest both immediate fixes and longer-term preventative measures
        4. Be concise (3-5 sentences)
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        logger.error(f"Error generating remediation suggestion: {str(e)}")
        return f"Unable to generate suggestion at this time. Manual investigation recommended for this {issue_type} issue."