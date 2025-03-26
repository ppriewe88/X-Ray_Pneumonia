import mlflow.pyfunc
import numpy as np
from PIL import Image
from tensorflow import keras
import time

# ' ##################################################'
# # Bild laden
# img = Image.open('IM-0001-0001.jpeg')

# # Bild auf 256x256 Pixel resizen
# img_resized = img.resize((256, 256))

# # In NumPy-Array konvertieren
# img_array = np.array(img_resized)

# # Dimensionen anpassen
# img_array = np.expand_dims(img_array, axis=0)
# img_array = img_array.reshape(-1, 256,256,1)
# img_array = img_array.astype(np.float32)

# # Überprüfen der Shape
# print(img_array.shape)  # Sollte (1, 256, 256, 1) ausgeben


' ####################### modell laden -> EIGENE FUNKTION MACHEN!!!  ###################################'
model_name = "MobileNet_transfer_learning"  # Small_CNN, MobileNet_transfer_learning
model_version = 1

"    mlflow server --host 127.0.0.1 --port 8080     "
mlflow.set_tracking_uri("http://127.0.0.1:8080")
print("tracking uri set")
start_loading = time.time()
print("start loading model")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
end_loading = time.time()
print("loading time: ", end_loading - start_loading)
# Signature abrufen
signature = model.metadata.signature
input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']

# Signature anzeigen
# print(' # signature # \n', signature, type(signature), signature.inputs, signature.outputs, type(signature.inputs), type(signature.outputs))


' ############################# image resizen #########################'
def resize_image_new(
    image,
    signature_shape,
    signature_dtype
    ):

    # convert image to numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape((*image_array.shape,1))

    # shape according to signature
    if signature_shape[-1] > 1:
        # populate each channel with the same pixel values
        img_array_tuple = tuple([image_array for i in signature_shape[-1]])
        image_array = np.concatenate(img_array_tuple, axis = -1)

    # resizing according to signature_shape
    resized_image = keras.ops.image.resize(
        image_array,
        size = (signature_shape[1], signature_shape[2]),
        interpolation="bilinear",
        )
    
    # converting to numpy and retyping according to signature_type
    image_array = resized_image.numpy().reshape(signature_shape)
    image_array = image_array.astype(signature_dtype)

    return image_array

# Bild laden
img = Image.open('IM-0001-0001.jpeg')


print("############ call resizing function according to signature ##############")
formatted_tensor = resize_image_new(image=img, signature_shape = input_shape, signature_dtype=input_type)
print(formatted_tensor.shape)


print("############# make prediction with pyfunc model -> EIGENE FUNKTION MACHEN!!! #############")
prediction = model.predict(formatted_tensor)
pred_reshaped = prediction.flatten()

print(prediction, type(prediction), pred_reshaped, type(pred_reshaped))
print(np.around(prediction))