import requests
from pathlib import Path
from enum import Enum
import random

' ################ configuration of API info (= given API endpoint requirements) ###########'
class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

base_url = "http://127.0.0.1:8000"
endpoint = "/upload_image"
url_with_endpoint = base_url + endpoint

' ################################ configure nr. of samples to generate #############'
# samples (prediction runs) to be generated
samples = 50  

' ################################ get images ########################################'
# Get absolute path of the project dir
project_folder = Path(__file__).resolve().parent.parent

# paths to image folders
normal_folder = project_folder / "data" / "test" / "NORMAL"
pneumonia_folder = project_folder / "data" / "test" / "PNEUMONIA"
print("Normal folder:", normal_folder)
print("Pneumonia folder:", pneumonia_folder)

# load images of both classes
normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))

# random choice of images for each class
selected_normal = random.sample(normal_images, samples // 2)
selected_pneumonia = random.sample(pneumonia_images, samples // 2)

# combine and shuffle selected images
all_selected_images = selected_normal + selected_pneumonia
random.shuffle(all_selected_images)

' ################################ generate predictions via API call ###############################'

for i, image_file in enumerate(all_selected_images):
    with open(image_file, "rb") as img:
        files = {"file": (image_file.name, img, "image/jpeg")}
        
        # get class from parent folder name
        data_class = image_file.parent.name
        params = {"label": Label.NEGATIVE.value if data_class == "NORMAL" else Label.POSITIVE.value}

        # make API call
        response = requests.post(url_with_endpoint, files=files, params=params)
        status_code = response.status_code

        # quick response logging
        if status_code == 200:
            print(f"Image {files['file'][0]} sent successfully. Response:", response.json())
        else:
            print(f"Error during sending. status code: {status_code}")
            print("Error details:", response.text)

    print(f"Call no. {i+1} of {samples} with class {data_class} done.")
