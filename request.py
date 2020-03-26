import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'przeznaczenie':2, 'wiek':9, 'kubatura':6})

print(r.json())