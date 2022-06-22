import requests

url = "http://127.0.0.1:5000/image/predict"

payload = {}
headers = {}

files = [
    (
        "image",
        (
            "maceration-present-48393.png",
            open(
                "images/maceration-present-48393.png",
                "rb",
            ),
            "image/png",
        ),
    )
]

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print("Prediction for DFU with maceration")
print(response.text)


files = [
    (
        "image",
        (
            "maceration-absent-46478.png",
            open(
                "images/maceration-absent-0497.png",
                "rb",
            ),
            "image/png",
        ),
    )
]

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print("Prediction for DFU without maceration")
print(response.text)
