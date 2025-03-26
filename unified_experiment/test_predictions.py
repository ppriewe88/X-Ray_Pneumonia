from tensorflow.keras.models import load_model
import time


start_loading = time.time()
model = load_model("model_transferlearning.keras")  # model_smallCNN, model_transferlearning
end_loading = time.time()

print("loading time: ", end_loading - start_loading)

import numpy as np
from PIL import Image

# Bild laden
img = Image.open('IM-0001-0001.jpeg')

# Bild auf 256x256 Pixel resizen
img_resized = img.resize((256, 256))

# In NumPy-Array konvertieren
img_array = np.array(img_resized)

# Dimensionen anpassen
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array.reshape(-1, 256,256,1)

# Überprüfen der Shape
print(img_array.shape)  # Sollte (1, 256, 256, 3) ausgeben

prediction = model.predict(img_array)
pred_reshap = prediction.flatten()

print(prediction, type(prediction), pred_reshap, type(pred_reshap))
print(np.around(prediction))