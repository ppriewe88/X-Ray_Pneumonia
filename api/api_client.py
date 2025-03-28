import os
import io
import requests
import numpy as np
from IPython.display import Image, display



base_url = "http://127.0.0.1:8080"
endpoint = "/upload_image"

url_with_endpoint = base_url + endpoint

file = "insert path of image here"

response = requests.post(url_with_endpoint, files = file, label= 1)
status_code = response.status_code