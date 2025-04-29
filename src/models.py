import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Union, Tuple

class KubernetesMetricsPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=5):
        super(KubernetesMetricsPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

_model = KubernetesMetricsPredictor(input_size=6, output_size=5)

def predict_kubernetes_metrics(input_data: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
    if isinstance(input_data, list):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    elif isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    elif isinstance(input_data, torch.Tensor):
        input_tensor = input_data.clone().detach().float()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    if input_tensor.size(-1) != _model.input_size:
        if input_tensor.size(-1) > _model.input_size:
            input_tensor = input_tensor[..., :_model.input_size]
        else:
            padding_size = _model.input_size - input_tensor.size(-1)
            padding = torch.zeros(input_tensor.size(0), padding_size, dtype=torch.float32)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
    
    _model.eval()
    
    with torch.no_grad():
        predictions = _model(input_tensor)
    
    return predictions

def detect_anomaly(predictions: torch.Tensor, threshold: float = 0.7) -> str:
    max_value = torch.max(torch.abs(predictions)).item()
    
    if max_value > threshold:
        return "Anomaly"
    else:
        return "Normal"

def classify_issue_type(predictions: torch.Tensor, actual_data: Dict[str, Any]) -> str:
    if len(predictions.shape) > 1:
        pred_values = predictions[0]  
    else:
        pred_values = predictions
    
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