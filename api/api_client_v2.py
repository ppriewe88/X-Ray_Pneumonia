import requests
from pathlib import Path
from enum import Enum
import random

class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

base_url = "http://127.0.0.1:8000"
endpoint = "/upload_image"
url_with_endpoint = base_url + endpoint

samples = 40  # Gesamtanzahl der Samples (10 pro Klasse)

# Pfade zu den Bilderordnern
normal_folder = Path(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\NORMAL")
pneumonia_folder = Path(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\PNEUMONIA")

# Bilder für beide Klassen laden
normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))

# Zufällige Auswahl von Bildern für jede Klasse
selected_normal = random.sample(normal_images, samples // 2)
selected_pneumonia = random.sample(pneumonia_images, samples // 2)

# Kombinieren und mischen der ausgewählten Bilder
all_selected_images = selected_normal + selected_pneumonia
random.shuffle(all_selected_images)

for i, image_file in enumerate(all_selected_images):
    with open(image_file, "rb") as img:
        files = {"file": (image_file.name, img, "image/jpeg")}
        
        # Bestimme die Klasse basierend auf dem Ordnernamen
        data_class = image_file.parent.name
        params = {"label": Label.NEGATIVE.value if data_class == "NORMAL" else Label.POSITIVE.value}

        response = requests.post(url_with_endpoint, files=files, params=params)
        status_code = response.status_code

        if status_code == 200:
            print(f"Image {files['file'][0]} sent successfully. Response:", response.json())
        else:
            print(f"Error during sending. status code: {status_code}")
            print("Error details:", response.text)

    print(f"Call no. {i+1} with class {data_class} done.")
