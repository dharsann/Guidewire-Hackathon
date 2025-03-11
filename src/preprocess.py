import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_k8s_metrics(pods=30, hours=24, interval_mins=5):
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(minutes=i*interval_mins) for i in range(int(hours*60/interval_mins))]
    
    namespaces = ['default', 'kube-system', 'app', 'monitoring', 'database']
    services = ['api', 'web', 'db', 'cache', 'queue', 'auth', 'worker']
    statuses = ['Running', 'Running', 'Running', 'Running', 'Pending', 'Failed']
    
    pod_data = []
    for i in range(pods):
        service = random.choice(services)
        namespace = random.choice(namespaces)
        
        cpu_profiles = {
            'api': 0.2, 
            'web': 0.15, 
            'db': 0.4, 
            'cache': 0.1, 
            'queue': 0.05, 
            'auth': 0.1, 
            'worker': 0.3
        }
        cpu_profile = cpu_profiles.get(service, 0.2)
        
        mem_profiles = {
            'api': 200, 
            'web': 150, 
            'db': 800, 
            'cache': 600, 
            'queue': 200, 
            'auth': 100, 
            'worker': 400
        }
        mem_profile = mem_profiles.get(service, 200)
        
        pod_data.append({
            'pod_name': f"{service}-{namespace}-{random.randint(1000, 9999)}",
            'namespace': namespace,
            'service': service,
            'node': f"node-{random.randint(1, 5)}",
            'status': random.choice(statuses),
            'cpu_limit': round(cpu_profile * 4, 1),
            'cpu_request': round(cpu_profile * 2, 1),
            'memory_limit': int(mem_profile * 2),
            'memory_request': int(mem_profile)
        })
    
    pods_df = pd.DataFrame(pod_data)
    
    metrics = []
    for timestamp in timestamps:
        hour = timestamp.hour
        time_factor = 0.7 + 0.5 * np.sin(np.pi * hour / 12)
        
        for _, pod in pods_df.iterrows():
            if pod['status'] == 'Running' or random.random() < 0.05:
                cpu_usage = pod['cpu_request'] * (0.5 + 0.7 * time_factor * random.random())
                if random.random() < 0.05:
                    cpu_usage *= random.uniform(1.5, 3.0)
                
                memory_usage = pod['memory_request'] * (0.7 + 0.4 * random.random())
                
                network_factor = 2.0 if pod['service'] in ['api', 'web'] else 1.0
                network_rx = random.lognormvariate(8, 0.8) * time_factor * network_factor
                network_tx = network_rx * random.uniform(0.3, 0.7)
                
                error_rate = 0
                if pod['service'] in ['api', 'web'] and random.random() < 0.1:
                    error_rate = random.uniform(0.01, 0.05)
                
                restart_count = 0
                if random.random() < 0.01:
                    restart_count = random.randint(1, 3)
                
                metrics.append({
                    'timestamp': timestamp,
                    'pod_name': pod['pod_name'],
                    'namespace': pod['namespace'],
                    'service': pod['service'],
                    'status': pod['status'],
                    'cpu_usage': round(cpu_usage, 2),
                    'cpu_utilization': round(cpu_usage / pod['cpu_limit'] * 100, 1),
                    'memory_usage': int(memory_usage),
                    'memory_utilization': round(memory_usage / pod['memory_limit'] * 100, 1),
                    'network_rx': int(network_rx),
                    'network_tx': int(network_tx),
                    'error_rate': round(error_rate, 3),
                    'restart_count': restart_count
                })
    
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    k8s_metrics = generate_k8s_metrics(pods=30, hours=24, interval_mins=5)
    
    k8s_metrics.to_csv("k8s_metrics.csv", index=False)
    
    print("\nSAMPLE DATA:")
    print(k8s_metrics.head(3))
    
    print(f"\nTotal records: {len(k8s_metrics)}")
    print(f"Unique pods: {k8s_metrics['pod_name'].nunique()}")
    print(f"Time range: {k8s_metrics['timestamp'].min()} to {k8s_metrics['timestamp'].max()}")
    print(f"Avg CPU utilization: {k8s_metrics['cpu_utilization'].mean():.1f}%")
    print(f"Avg memory utilization: {k8s_metrics['memory_utilization'].mean():.1f}%")