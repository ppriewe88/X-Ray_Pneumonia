import requests
from pathlib import Path
from enum import Enum


' ################# class for labels; as in endpoint for prediciton ###############'
class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

' ################# configure communication with API endpoint ############'
base_url = "http://127.0.0.1:8000"
endpoint = "/upload_image"

url_with_endpoint = base_url + endpoint

' ################# make n=samples calls for each class  #########################'
samples = 5

for data_class in ["NORMAL", "PNEUMONIA"]:
    ' ################### folder to data (images) #################################'
    # path to images; currently absolute path
    image_folder = Path(rf"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\{data_class}")

    ' ######################## take first n images and make prediction + logging ###########'
    # get first n=samples images
    image_files = list(image_folder.glob("*"))[:samples]

    for i, image_file in enumerate(image_files):
        with open(image_file, "rb") as img:
            
            # prepare image in suitable format for endpoint
            files = {"file": (image_file.name, img, "image/jpeg")}
            
            # prepare label in suitable format for endpoint (query-param)
            if data_class == "NORMAL":
                params = {"label": Label.NEGATIVE.value}  # 0
            elif data_class == "PNEUMONIA":
                params = {"label": Label.POSITIVE.value}  # 1

            # send request to endpoint
            response = requests.post(
                url_with_endpoint, 
                files=files, 
                params=params  # query params
            )

            # keep status_code
            status_code = response.status_code

            # simple error handling
            if status_code == 200:
                print(f"Image {files["file"][0]} sent successfully. Response:", response.json())
            else:
                print(f"Error during sending. status code: {status_code}")
                print("Error details:", response.text)

        print(f"Call no. {i+1} with class {data_class} done.")
