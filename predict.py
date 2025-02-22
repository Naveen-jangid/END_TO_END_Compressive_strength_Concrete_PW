import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "cement": 300,
    "blast_furnace_slag": 0,
    "fly_ash": 0,
    "water": 180,
    "superplasticizer": 0,
    "coarse_aggregate": 1000,
    "fine_aggregate": 800,
    "age": 28
}

response = requests.post(url, json=data)
print(response.json())  # Should return predicted strength