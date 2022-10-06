import json
import requests

url = "http://localhost:9696/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
response = requests.post(url, json=client)
result = response.json()

print(json.dumps(result, indent=2))
