import asyncio
import time
import torch
import google.generativeai as genai
import logging
import prometheus_client
from prometheus_client import Gauge, generate_latest
from fastapi import FastAPI, HTTPException, Body, Query, Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.utils import (
    get_active_pods, get_active_nodes, restart_pod, get_active_deployments,
    scale_deployment_hpa, migrate_pods_from_node, classify_issue_type, generate_remediation_suggestion
)
from src.models import predict_kubernetes_metrics, detect_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="Kubernetes Self-Healing API", 
              description="API for Kubernetes monitoring, self-healing, and remediation recommendations")

genai.configure(api_key="AIzaSyDlp8BfQuAlfCJmrLWbDQQd08Jt8uxJ5ME")
app = FastAPI()

prediction_metric = Gauge('k8s_prediction', 'Predicted Kubernetes issue type', ['issue_type'])
remediation_metric = Gauge('k8s_remediation', 'Remediation action taken', ['action'])

class RemediationRequest(BaseModel):
    resource_type: str
    resource_name: str
    issue_type: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None

class RemediationResponse(BaseModel):
    success: bool
    action_taken: str
    details: str
    timestamp: str

class ResourceMetrics(BaseModel):
    resource_name: str
    resource_type: str
    metrics: Dict[str, float]
    status: str

class AnomalyDetectionResponse(BaseModel):
    resource_name: str
    anomaly_detected: bool
    issue_type: Optional[str] = None
    confidence: float
    metrics: Dict[str, float]
    recommendation: Optional[str] = None

class RecommendationResponse(BaseModel):
    resource_name: str
    resource_type: str
    recommendations: List[Dict[str, Any]]
    generated_at: str
    priority: str

async def get_kubernetes_metrics():
    import numpy as np
    
    metrics = np.random.rand(6)
    
    metrics[0] = metrics[0] * 0.9  
    metrics[1] = metrics[1] * 0.8  
    metrics[2] = metrics[2] * 0.5  
    metrics[3] = metrics[3] * 0.7  
    metrics[4] = metrics[4] * 0.3  
    metrics[5] = metrics[5] * 1.0  
    
    return metrics

async def monitor_kubernetes():
    while True:
        try:
            active_pods = get_active_pods()
            active_deployments = get_active_deployments()
            active_nodes = get_active_nodes()
            
            pod_name = active_pods[0] if active_pods else "unknown"
            deployment_name = active_deployments[0] if active_deployments else "unknown"
            node_name = active_nodes[0] if active_nodes else "unknown"
            
            metrics = await get_kubernetes_metrics()
            
            input_features = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0)
            
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

@app.get("/api/v1/resources/{resource_type}", response_model=List[str])
async def get_resources(resource_type: str = Path(..., description="Type of resource (pods, nodes, deployments)")):
    try:
        if resource_type == "pods":
            return get_active_pods()
        elif resource_type == "nodes":
            return get_active_nodes()
        elif resource_type == "deployments":
            return get_active_deployments()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported resource type: {resource_type}")
    except Exception as e:
        logger.error(f"Error retrieving {resource_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve {resource_type}: {str(e)}")

@app.post("/api/v1/analyze", response_model=AnomalyDetectionResponse)
async def analyze_resource(
    resource_type: str = Query(..., description="Type of resource to analyze"),
    resource_name: str = Query(..., description="Name of the resource to analyze")
):
    try:
        metrics = await get_kubernetes_metrics()
        
        input_features = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0)
        
        predicted_values = predict_kubernetes_metrics(input_features)
        anomaly_status = detect_anomaly(predicted_values)
        issue_type = classify_issue_type(predicted_values, actual_data={})
        
        recommendation = None
        if anomaly_status == "Anomaly":
            recommendation = generate_remediation_suggestion(
                issue_type, 
                resource_name if resource_type == "pods" else "unknown",
                resource_name if resource_type == "nodes" else "unknown",
                resource_name if resource_type == "deployments" else "unknown"
            )
        
        metrics_dict = {f"metric_{i}": float(metrics[i]) for i in range(len(metrics))}
        
        return AnomalyDetectionResponse(
            resource_name=resource_name,
            anomaly_detected=anomaly_status == "Anomaly",
            issue_type=issue_type if anomaly_status == "Anomaly" else None,
            confidence=0.85, 
            metrics=metrics_dict,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Error analyzing resource {resource_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/remediate", response_model=RemediationResponse)
async def remediate_resource(request: RemediationRequest = Body(...)):
    try:
        resource_type = request.resource_type
        resource_name = request.resource_name
        issue_type = request.issue_type or "unknown"
        
        action = "No action taken"
        details = "Resource evaluated but no action was necessary"
        
        if issue_type == "pod_crash" or (resource_type == "pods" and issue_type == "unknown"):
            action = restart_pod(resource_name)
            details = f"Pod {resource_name} restarted due to crash or issue"
        
        elif issue_type == "high_resource_usage" or (resource_type == "deployments" and issue_type == "unknown"):
            action = scale_deployment_hpa(resource_name)
            details = f"Deployment {resource_name} scaled due to high resource usage"
        
        elif issue_type == "node_failure" or (resource_type == "nodes" and issue_type == "unknown"):
            action = migrate_pods_from_node(resource_name)
            details = f"Pods migrated from node {resource_name} due to node failure"
        
        elif issue_type == "network_issue":
            if resource_type == "pods":
                action = restart_pod(resource_name)
                details = f"Pod {resource_name} restarted due to network issue"
            else:
                details = f"Network issue detected but no appropriate action for {resource_type}"
        
        remediation_metric.labels(action=action).set(1)
        
        return RemediationResponse(
            success=True,
            action_taken=action,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error(f"Error remediating {request.resource_type}/{request.resource_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Remediation failed: {str(e)}")

@app.get("/api/v1/recommendations/{resource_type}/{resource_name}", response_model=RecommendationResponse)
async def get_recommendations(
    resource_type: str = Path(..., description="Type of resource (pods, nodes, deployments)"),
    resource_name: str = Path(..., description="Name of the resource to get recommendations for")
):
    try:
        metrics = await get_kubernetes_metrics()
        
        input_features = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0)
        
        predicted_values = predict_kubernetes_metrics(input_features)
        issue_type = classify_issue_type(predicted_values, actual_data={})
        
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Generate remediation and optimization recommendations for a Kubernetes {resource_type} named '{resource_name}'.
        
        Current metrics:
        - CPU Usage: {metrics[0]:.2f}
        - Memory Usage: {metrics[1]:.2f}
        - Network Latency: {metrics[2]:.2f}
        - Disk I/O: {metrics[3]:.2f}
        - Error Rate: {metrics[4]:.2f}
        
        Detected issue type: {issue_type if issue_type != "unknown" else "None detected"}
        
        Provide 3 recommendations with the following details for each:
        1. Title of recommendation
        2. Detailed description
        3. Expected benefit
        4. Implementation complexity (Low, Medium, High)
        5. Priority (Low, Medium, High)
        
        Format as JSON.
        """
        
        response = model.generate_content(prompt)
        try:
            import json
            recommendations = json.loads(response.text)
        except Exception:
            recommendations = [
                {
                    "title": f"Optimize resource limits for {resource_name}",
                    "description": "Review and adjust CPU and memory limits based on actual usage patterns.",
                    "expected_benefit": "Improved resource utilization and reduced costs.",
                    "complexity": "Medium",
                    "priority": "Medium"
                },
                {
                    "title": "Implement health checks",
                    "description": f"Add liveness and readiness probes to {resource_name}.",
                    "expected_benefit": "Faster recovery from failures and improved reliability.",
                    "complexity": "Low",
                    "priority": "High"
                },
                {
                    "title": "Upgrade Kubernetes version",
                    "description": "Consider upgrading to the latest stable Kubernetes version for improved features and security.",
                    "expected_benefit": "Access to latest features and security patches.",
                    "complexity": "High",
                    "priority": "Low"
                }
            ]
        
        # Determine overall priority based on issues detected
        priority = "Low"
        if issue_type != "unknown":
            priority = "High" if issue_type in ["pod_crash", "node_failure"] else "Medium"
        
        return RecommendationResponse(
            resource_name=resource_name,
            resource_type=resource_type,
            recommendations=recommendations,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            priority=priority
        )
    except Exception as e:
        logger.error(f"Error generating recommendations for {resource_type}/{resource_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/api/v1/cluster/health", response_model=Dict[str, Any])
async def get_cluster_health():
    """
    Get overall cluster health status with summarized metrics.
    """
    try:
        # In a real implementation, this would query actual cluster metrics
        # For demonstration, we'll generate synthetic data
        active_pods = get_active_pods()
        active_nodes = get_active_nodes()
        active_deployments = get_active_deployments()
        
        # Get metrics from different resources
        pod_metrics = []
        node_metrics = []
        deployment_metrics = []
        
        for pod in active_pods[:5]:  # Limit to 5 pods for demo
            metrics = await get_kubernetes_metrics()
            pod_metrics.append({
                "name": pod,
                "cpu": metrics[0],
                "memory": metrics[1],
                "status": "Healthy" if metrics[4] < 0.1 else "Warning"
            })
        
        for node in active_nodes:
            metrics = await get_kubernetes_metrics()
            node_metrics.append({
                "name": node,
                "cpu": metrics[0],
                "memory": metrics[1],
                "disk": metrics[3],
                "status": "Healthy" if all(m < 0.7 for m in metrics) else "Warning"
            })
        
        for deployment in active_deployments:
            metrics = await get_kubernetes_metrics()
            deployment_metrics.append({
                "name": deployment,
                "desired_replicas": 3,
                "available_replicas": 3 if metrics[4] < 0.1 else 2,
                "status": "Healthy" if metrics[4] < 0.1 else "Warning"
            })
        
        # Calculate overall health scores
        pod_health = sum(1 for p in pod_metrics if p["status"] == "Healthy") / max(1, len(pod_metrics))
        node_health = sum(1 for n in node_metrics if n["status"] == "Healthy") / max(1, len(node_metrics))
        deployment_health = sum(1 for d in deployment_metrics if d["status"] == "Healthy") / max(1, len(deployment_metrics))
        
        overall_health = (pod_health + node_health + deployment_health) / 3
        health_status = "Healthy" if overall_health > 0.9 else "Warning" if overall_health > 0.7 else "Critical"
        
        return {
            "status": health_status,
            "health_score": overall_health,
            "components": {
                "pods": {
                    "health_score": pod_health,
                    "total": len(active_pods),
                    "healthy": sum(1 for p in pod_metrics if p["status"] == "Healthy"),
                    "warning": sum(1 for p in pod_metrics if p["status"] == "Warning"),
                    "critical": sum(1 for p in pod_metrics if p["status"] == "Critical"),
                    "samples": pod_metrics
                },
                "nodes": {
                    "health_score": node_health,
                    "total": len(active_nodes),
                    "healthy": sum(1 for n in node_metrics if n["status"] == "Healthy"),
                    "warning": sum(1 for n in node_metrics if n["status"] == "Warning"),
                    "critical": sum(1 for n in node_metrics if n["status"] == "Critical"),
                    "samples": node_metrics
                },
                "deployments": {
                    "health_score": deployment_health,
                    "total": len(active_deployments),
                    "healthy": sum(1 for d in deployment_metrics if d["status"] == "Healthy"),
                    "warning": sum(1 for d in deployment_metrics if d["status"] == "Warning"),
                    "critical": sum(1 for d in deployment_metrics if d["status"] == "Critical"),
                    "samples": deployment_metrics
                }
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting cluster health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster health: {str(e)}")

@app.post("/api/v1/ai/analyze")
async def ai_analyze_cluster():
    """
    Perform a deep AI-based analysis of the cluster and generate comprehensive recommendations.
    """
    try:
        # Get cluster health data
        cluster_health = await get_cluster_health()
        
        # Get sample metrics from different resources
        pod_metrics = [p for p in cluster_health["components"]["pods"]["samples"]]
        node_metrics = [n for n in cluster_health["components"]["nodes"]["samples"]]
        deployment_metrics = [d for d in cluster_health["components"]["deployments"]["samples"]]
        
        # Use Gemini AI for comprehensive analysis
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Perform a deep analysis of a Kubernetes cluster based on the following data:
        
        Overall Cluster Health: {cluster_health['status']} (Score: {cluster_health['health_score']:.2f})
        
        Pod Health:
        {pod_metrics}
        
        Node Health:
        {node_metrics}
        
        Deployment Health:
        {deployment_metrics}
        
        Provide a comprehensive analysis including:
        1. Summary of cluster health
        2. Identified issues and potential root causes
        3. Short-term remediation actions
        4. Long-term recommendations for improving reliability and performance
        5. Resource optimization strategies

        Structure the response in markdown format with clear sections and bullet points.
        """
        
        response = model.generate_content(prompt)
        analysis_text = response.text
        
        # Return the analysis
        return {
            "analysis": analysis_text,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cluster_health": cluster_health["status"],
            "recommendations_count": analysis_text.lower().count("recommend")
        }
    except Exception as e:
        logger.error(f"Error in AI cluster analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)