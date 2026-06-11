"""
Send local GET and POST requests to the FastAPI application.
"""

import requests


BASE_URL = "http://127.0.0.1:8000"


get_response = requests.get(f"{BASE_URL}/", timeout=10)

print("GET request")
print("Status Code:", get_response.status_code)
print("Result:", get_response.json())
print()


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

post_response = requests.post(
    f"{BASE_URL}/data/",
    json=data,
    timeout=10,
)

print("POST request")
print("Status Code:", post_response.status_code)
print("Result:", post_response.json())
