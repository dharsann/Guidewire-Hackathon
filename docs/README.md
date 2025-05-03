---

# 🚑 Kubernetes Self-Healing API

An intelligent FastAPI-based service that monitors a Kubernetes cluster and performs **self-healing**, **anomaly detection**, and **AI-powered remediation recommendations** using LSTM, Isolation Forest, and Gemini Pro.

---

## 📦 Features

- Predict Kubernetes failures using LSTM
- Detect anomalies in pod, node, and deployment metrics
- Automate remediation actions (restart pod, scale deployment, migrate pods)
- Generate AI-based remediation suggestions (via Google Gemini Pro)
- Export Prometheus metrics
- Cluster-wide health monitoring and scoring

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Ensure that your environment includes:
- `torch`
- `google-generativeai`
- `prometheus_client`
- `fastapi`
- `uvicorn`
- A valid Google Generative AI API Key

---

## 🚀 Endpoints

### Health & Metrics

#### `GET /health`
Check API status.

#### `GET /metrics`
Returns Prometheus-formatted metrics.

---

### Resource Discovery

#### `GET /api/v1/resources/{resource_type}`
List active Kubernetes resources (pods, nodes, deployments).

- **Path Params**:
  - `resource_type`: `pods` | `nodes` | `deployments`

**Example**:
```bash
curl http://localhost:8000/api/v1/resources/pods
```

---

### 🔍 Anomaly Detection

#### `POST /api/v1/analyze`
Analyze a Kubernetes resource for anomalies and get a recommendation.

- **Query Params**:
  - `resource_type`: `pods`, `nodes`, `deployments`
  - `resource_name`: name of the resource

**Response**:
```json
{
  "resource_name": "nginx-pod-1",
  "anomaly_detected": true,
  "issue_type": "pod_crash",
  "confidence": 0.85,
  "metrics": { "metric_0": 0.45, ... },
  "recommendation": "Restart the pod immediately"
}
```

---

### 🔧 Remediation

#### `POST /api/v1/remediate`
Trigger remediation for a resource based on detected or provided issue.

- **Body**:
```json
{
  "resource_type": "pods",
  "resource_name": "nginx-pod-1",
  "issue_type": "pod_crash"
}
```

**Response**:
```json
{
  "success": true,
  "action_taken": "Pod restarted",
  "details": "Pod nginx-pod-1 restarted due to crash",
  "timestamp": "2025-04-29 14:20:31"
}
```

---

### 💡 Recommendations

#### `GET /api/v1/recommendations/{resource_type}/{resource_name}`
AI-generated Kubernetes best-practice and remediation recommendations using Gemini Pro.

**Response**:
```json
{
  "resource_name": "nginx-deployment",
  "resource_type": "deployments",
  "recommendations": [
    {
      "title": "Implement Health Checks",
      "description": "Add liveness and readiness probes...",
      "expected_benefit": "Faster recovery from failures",
      "complexity": "Low",
      "priority": "High"
    }
  ],
  "generated_at": "2025-04-29 14:22:00",
  "priority": "High"
}
```

---

### 🧠 Cluster Health Overview

#### `GET /api/v1/cluster/health`
Returns overall cluster health score and metrics summary.

**Response**:
```json
{
  "status": "Healthy",
  "health_score": 0.93,
  "components": {
    "pods": {
      "health_score": 0.9,
      "total": 10,
      "healthy": 9,
      "warning": 1,
      ...
    },
    ...
  },
  "timestamp": "2025-04-29 14:25:00"
}
```

---

## 📊 Prometheus Integration

Metrics are exposed at `/metrics` and include:
- `k8s_prediction{issue_type="pod_crash"}`
- `k8s_remediation{action="restart_pod"}`

---

## 🧠 Powered by

- **PyTorch** (LSTM predictions)
- **Isolation Forest** (anomaly detection)
- **Gemini Pro API** (remediation suggestions)
- **Prometheus** (metrics)
- **FastAPI** (API framework)

---

## 🧪 Development Mode

To simulate metrics and test functionality without a live cluster, adjust the `get_kubernetes_metrics()` function to return synthetic values (already implemented).

---
