import requests
import pandas as pd

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

def fetch_prometheus_metrics(query):
    response = requests.get(PROMETHEUS_URL, params={"query": query})
    return response.json()

cpu_data = fetch_prometheus_metrics('rate(container_cpu_usage_seconds_total[5m])')
df = pd.DataFrame(cpu_data['data']['result'])
print(df.head()) 
