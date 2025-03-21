"""
This script contains the training routine for transfer learning with MobileNet and/or ResNet.
It also contains the setup for experiment tracking via mlflow.
If you run this script to train, and you want to log with mlflow, you have to start the tracking server of mlflow. 
To do so, in the directory of this script (i.e. folder transfer_learning), run the following command in terminal to start mlflow server for tracking experiments:
mlflow server --host 127.0.0.1 --port 8080
Then check the localhost port to access the MLFlow GUI for tracking!
Run this script to conduct training experiments (runs). If mlflow server is running, the experiment will be tracked as a run"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
import mlflow
from mlflow.models import infer_signature
import io
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.helpers import get_data, DATA_PATH, IMGSIZE


# %%
' ####################### params #######################'
# params for training
BATCHSIZE = 10
CHOSEN_EPOCHS = 20
dense_layer_top_neurons = 128
dense_layer_top_activation = "relu"
dropout_rate_top = 0.5
chosen_loss = "binary_crossentropy"
chosen_optimizer = "adam"
chosen_learning_rate = 0.001
early_stopping = True

# param for base model selection
selected_model = "MobileNet" # "MobileNet
if selected_model == "ResNet":
    tag = "ResNet152V2 with Dense top"
elif selected_model == "MobileNet":
    tag = "MobileNet"
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
          "data": DATA_PATH,
          "tag": tag}
mlflow_tracking = True

# %%
' ######################################### getting training and validation data ################################'

train_data, val_data = get_data(BATCHSIZE, IMGSIZE, selected_data = "train")

 # %%
' ################################################## defining the model #########################'
# base model
if selected_model == "ResNet":
    # ResNet
    base_model = keras.applications.ResNet152V2(
        weights = "imagenet",
        input_shape = (IMGSIZE,IMGSIZE, 3),
        include_top = False)
elif selected_model == "MobileNet":
    # MobileNet
    base_model = keras.applications.MobileNet(
    input_shape = (IMGSIZE,IMGSIZE, 3),
    include_top=False,
    weights="imagenet")

base_model.trainable = False

# complete model setup
inputs = tf.keras.layers.Input(shape = (IMGSIZE, IMGSIZE, 3))
x = keras.layers.Rescaling(scale = 1./255)(inputs)
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(dense_layer_top_neurons, activation=dense_layer_top_activation)(x)
x = layers.Dropout(dropout_rate_top)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)

' ######################################### compile and summary #######################################'
# compile model
model.compile(loss=chosen_loss, 
              optimizer = keras.optimizers.Adam(learning_rate=chosen_learning_rate), 
              metrics=['binary_accuracy'])

# print model summary
model.summary()

# get model summary as string
buffer = io.StringIO()
model.summary(print_fn=lambda x: buffer.write(x + '\n'))
summary_str = buffer.getvalue()

# %%
' ########################################## training #######################'
# define callbacks 
if early_stopping:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.002,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )
    chosen_callbacks = [early_stopping]
else:
    chosen_callbacks = None

# training
start_time = time.time()

history = model.fit(train_data,
          batch_size = BATCHSIZE, epochs = CHOSEN_EPOCHS,
          validation_data=val_data,
          callbacks = chosen_callbacks
          );

end_time = time.time()
training_time = end_time - start_time
print(f"train time:  {training_time:.2f} seconds = {training_time/60:.1f} minutes")

# %%
' ########################################## prediction on validation and test set ########'
# get training data again (generators have been consumed during training and need to be reconstructed)
train_data, val_data = get_data(BATCHSIZE, IMGSIZE, selected_data = "train")
val_loss, val_binary_accuracy = model.evaluate(val_data, verbose = 1)
# get test data
test_data_throwaway, test_data = get_data(BATCHSIZE, IMGSIZE, selected_data = "test")
test_loss, test_binary_accuracy = model.evaluate(test_data, verbose = 1)
print('Val loss:', val_loss)
print('Val binary accuracy:', val_binary_accuracy)
print('test loss:', test_loss)
print('test binary accuracy:', test_binary_accuracy)

# %%
'####################### generate plot of learning curves ################'
# create learning curve (for logging with MLFlow)
fig, ax = plt.subplots()
ax.plot(history.history['binary_accuracy'], label='Train accuracy (binary)')
ax.plot(history.history['val_binary_accuracy'], label='Validation accuracy (binary)')
ax.set_xlabel('Epoch')
ax.set_ylabel('binary accuracy')
ax.legend()
ax.set_title("Training and Validation binary accuracy")

# %%
' ########################### MLFlow model logging #######################'
# Set tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Xray_Pneumonia")

if mlflow_tracking:
    # start logging the run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)
    
        # Log the metrics (validation and test)
        metrics = {"binary accuracy validation data": val_binary_accuracy,
                   "binary accuracy test data": test_binary_accuracy}
        mlflow.log_metrics(metrics)
    
        # log plot of learning curve (and close plt.object afterwards)
        mlflow.log_figure(fig, "learning_curve_bin_acc.png")
        plt.close(fig)
        
        # log model summary as text artifact
        mlflow.log_text(summary_str, "model_summary.txt")
    
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", tag)
    
        # infer model signature
        batch = next(iter(train_data.take(1)))
        single_image = batch[0][0]
        single_image_batch = tf.expand_dims(single_image, axis=0)
        single_image_batch = tf.expand_dims(single_image, axis=0)
        predictions = model.predict(single_image_batch)
        signature = infer_signature(single_image_batch.numpy(), predictions)
    
        # Log the model
        model_info = mlflow.keras.log_model(
            model = model,
            artifact_path = "digits_model",
            signature = signature
        )
