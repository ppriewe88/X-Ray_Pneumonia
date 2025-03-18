import time


import os

# You can use 'tensorflow', 'torch' or 'jax' as backend. Make sure to set the environment variable before importing Keras.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mlflow
from PIL import Image


# params to log: model_summary (as string), parameters from get_model() function, learning curves + metrics, 
# infer model signature (gives input shape and output shape)

'''
img = Image.open("./data/train/PNEUMONIA/person2_bacteria_4.jpeg")
img_array = np.array(img) 

print('image shape = ', img_array.shape)
print('max pixel value = ', img_array.max())
'''

'''

# params for training
BATCHSIZE = 1000
IMGSIZE = 256
CHOSEN_EPOCHS = 5
dense_layer_top_neurons = 128
dense_layer_top_activation = "relu"
dropout_rate_top = 0.1
chosen_loss = "binary_crossentropy"
chosen_optimizer = "adam"
chosen_learning_rate = 0.001
data_path = os.path.join("..","data/train/")
tag = "ResNet + Dense Top"

# params for mlflow
params = {"batch size": BATCHSIZE,
          "image size": IMGSIZE,
          "epochs": CHOSEN_EPOCHS,
          "top dense layer neurons": dense_layer_top_neurons,
          "top dense layer activation": dense_layer_top_activation,
          "top dropout rate": dropout_rate_top,
          "loss": chosen_loss,
          "optimizer": chosen_optimizer,
          "learning rate": chosen_learning_rate,
          "momentum": momentum
          "data": data_path,
          "tag": tag}
mlflow_tracking = True

'''




BATCH_SIZE = 128
IMG_SIZE = 256
EPOCHS = 3 

# images are grayscale, max pixel value is 255

train_data, val_data = keras.utils.image_dataset_from_directory(
    # relative path to images
    r'./data/train',
    labels='inferred',              # labels are generated from the directory structure
    label_mode='binary',            # 'binary' => binary cross-entropy
    # class_names=class_names_list, # such that i can control the order of the class names
    color_mode='grayscale',         # alternatives: 'grayscale', 'rgba'
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,                   # shuffle images before each epoch
    seed=0,                         # shuffle seed
    validation_split=0.2,           # percentage of validation data
    subset='both',                  # return a tuple of datasets (train, val)
    interpolation='bilinear',       # interpolation method used when resizing images
    follow_links=False,             # follow folder structure?
    crop_to_aspect_ratio=False
    )


'''
batch = train_data.take(1)

print('batch shape = ', list(batch)[0][0].numpy().shape) # (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)
'''



def get_model(add_dropout = True, dropout_rate = 0.3, add_batch_normalization = True):
    
    inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Rescaling
    x = keras.layers.Rescaling(scale = 1./255)(inputs)
    
    # First block
    x = keras.layers.Conv2D(
        filters=8,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same', # such that the output has the same size as the input
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMG_SIZE, IMG_SIZE, 8)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMG_SIZE/4, IMG_SIZE/4, 8) 
    
    # Second block 
    x = keras.layers.Conv2D(
        filters=16,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMG_SIZE/4, IMG_SIZE/4, 16)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMG_SIZE/16, IMG_SIZE/16, 16) 
    
    # Third block 
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMG_SIZE/16, IMG_SIZE/16, 32)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMG_SIZE/64, IMG_SIZE/64, 32) = (4,4,32)
    
    # Head (Flatten + Dense Layer)
    
    x = keras.layers.Flatten()(x) # output shape = 512
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer = None)(x)
    
    #Final Layer (Output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model


model = get_model()

model.compile(
    loss="binary_crossentropy", 
    metrics=["accuracy", "f1_score"],
    optimizer=keras.optimizers.SGD(learning_rate=0.1)
    )

model.summary()

# TO DO:
# add class weights
# add callbacks


# first run    mlflow server --host 127.0.0.1 --port 8080   in a different terminal to open the server
mlflow.set_tracking_uri("http://127.0.0.1:8080")



start = time.time()

# 188 sec for 5 epochs => ~ 37.6 s/epoch

'''
history = model.fit(
    train_data, 
    epochs = EPOCHS, 
    verbose = True,
    # class_weight = class_weights_dict,
    validation_data = val_data,
    # callbacks = [lr_reduction, model_checkpoint_callback]
    )
'''



# i should do the logging the classic way, as mlflow.keras.MlflowCallback() is an experimental...

run = mlflow.start_run()

model.fit(
    train_data, 
    epochs = EPOCHS, 
    verbose = True,
    # class_weight = class_weights_dict,
    validation_data = val_data,
    callbacks=[mlflow.keras.MlflowCallback(run)],
)

mlflow.end_run()


print(f'Training time = {time.time() - start} sec')