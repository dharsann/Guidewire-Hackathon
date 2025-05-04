import subprocess
import time
import requests
import json

FASTAPI_URL = "http://localhost:8000/self-heal"
NAMESPACE = "default"

def run_kubectl_cmd(command):
    full_cmd = ["kubectl"] + command
    result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr.strip())
    return result.stdout.strip()

def deploy_nginx():
    print("Deploying nginx-test...")
    run_kubectl_cmd(["create", "deployment", "nginx-test", "--image=nginx"])
    time.sleep(5)

def get_pod_name():
    print("Getting pod name...")
    pods = run_kubectl_cmd(["get", "pods", "-n", NAMESPACE, "-l", "app=nginx-test", "-o", "json"])
    pods_json = json.loads(pods)
    return pods_json["items"][0]["metadata"]["name"]

def delete_pod(pod_name):
    print(f"Deleting pod {pod_name} to simulate failure...")
    run_kubectl_cmd(["delete", "pod", pod_name, "-n", NAMESPACE])

def send_self_heal_request(pod_name):
    print("Sending simulated anomaly to /self-heal...")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pod_name": pod_name,
        "namespace": NAMESPACE,
        "cpu_allocation_efficiency": 0.02,
        "memory_allocation_efficiency": 0.6,
        "disk_io": 30,
        "network_latency": 50,
        "node_temperature": 75,
        "node_cpu_usage": 92,
        "node_memory_usage": 78,
        "event_type": "Warning",
        "event_message": "High CPU usage with low efficiency",
        "scaling_event": False,
        "pod_lifetime_seconds": 200
    }

    response = requests.post(FASTAPI_URL, json=payload)
    print("Response:", response.status_code)
    print(response.json())

if __name__ == "__main__":
    deploy_nginx()
    time.sleep(10)
    pod = get_pod_name()
    delete_pod(pod)
    time.sleep(5)
    send_self_heal_request(pod)
