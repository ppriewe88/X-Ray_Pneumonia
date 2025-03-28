import requests
from pathlib import Path
from enum import Enum

class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

base_url = "http://127.0.0.1:8000"
endpoint = "/upload_image"

url_with_endpoint = base_url + endpoint

# Pfad zum Ordner mit den Bildern
image_folder = Path(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\NORMAL")

# Wähle das erste Bild aus dem Ordner
image_file = next(image_folder.glob("*"))

# Sammeln der ersten 10 Bilddateien
image_files = list(image_folder.glob("*"))[:3]

for i, image_file in enumerate(image_files):
    with open(image_file, "rb") as img:
        
        # prepare image in suitable format for endpoint
        files = {"file": (image_file.name, img, "image/jpeg")}
        
        # prepare label in suitable format for endpoint (query-param)
        params = {"label": Label.NEGATIVE.value}  # 0

        # send request to endpoint
        response = requests.post(
            url_with_endpoint, 
            files=files, 
            params=params  # query params
        )

        # keep status_code
        status_code = response.status_code

        # quick error handling
        if status_code == 200:
            print(f"Image {files["file"][0]} sent successfully. Response:", response.json())
        else:
            print(f"Error during sending. status code: {status_code}")
            print("Error details:", response.text)

    print(f"Call no. {i+1} done.")
