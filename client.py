import requests

body = {
    "Gender": 1, 
    "Age": 68,
    "Height": 190.0,
    "Weight": 94.0,
    "Duration": 29.0,
    "Heart_Rate": 105.0,
    "Body_Temp": 40.8
}

response = requests.post(url='http://127.0.0.1:8000/predict', json=body)
print(response.json())
# output esperado: {'calories': 228.7}

