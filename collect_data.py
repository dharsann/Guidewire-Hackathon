import requests
import json

url = "http://localhost:9090/api/v1/query?query=container_memory_usage_bytes"
response = requests.get(url)
data = response.json()

# Save as JSON file
with open("metrics.json", "w") as f:
    json.dump(data, f)

# Convert to CSV
import csv

with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pod", "memory_usage_bytes"])
    for result in data["data"]["result"]:
        writer.writerow([result["metric"]["pod"], result["value"][1]])

print("Exported metrics to metrics.json and metrics.csv")
