import mlflow.pyfunc
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


model_name = "Small_CNN"
model_version = 1

"    mlflow server --host 127.0.0.1 --port 8080     "
mlflow.set_tracking_uri("http://127.0.0.1:8080")
print("tracking uri set")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
print("model loaded")
prediction = model.predict(img_array)
pred_reshap = prediction.flatten()

print(prediction, type(prediction), pred_reshap, type(pred_reshap))
print(np.around(prediction))