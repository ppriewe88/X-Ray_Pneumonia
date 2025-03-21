import time
import io

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mlflow
from mlflow import MlflowClient
#from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.helpers import get_data, DATA_PATH, IMGSIZE



'''
img = Image.open("./data/train/PNEUMONIA/person2_bacteria_4.jpeg")
img_array = np.array(img) 

print('image shape = ', img_array.shape)
print('max pixel value = ', img_array.max())
'''

# params to log: model_summary (as string), parameters from get_model() function, learning curves + metrics, 
# infer model signature (gives input shape and output shape)

BATCH_SIZE = 600
EPOCHS = 1

loss_func = "binary_crossentropy"
learning_rate = 0.01
momentum = 0.8
optimizer = keras.optimizers.SGD(learning_rate= learning_rate, momentum = momentum)
dropout_rate = 0.3
tag = "Own CNN"

# params for mlflow; will need to add more 
params_dict = {
    "epochs": EPOCHS,
    "loss_function": loss_func,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "momentum": momentum,
    "dropout_rate": dropout_rate,
    "tag": tag
    }
mlflow_logging=False

train_data, val_data = get_data(BATCH_SIZE, IMGSIZE, channelmode = "grayscale", selected_data = "train")

# take an input example out of the train data -> needed for mlflow

batch = train_data.take(1)
batch_as_nparray = list(batch)[0][0].numpy()
input_example = batch_as_nparray[0]
input_example = np.expand_dims(input_example, axis=0)

# print('batch shape = ', list(batch)[0][0].numpy().shape) # (BATCH_SIZE, IMGSIZE, IMGSIZE, 1)


# get class weights

def get_class_weights():

    all_labels = np.array([])
    
    for _, y in train_data:
        y_flat = y.numpy().flatten()
        all_labels = np.concatenate((all_labels,y_flat))

    label_counts = [np.sum(all_labels == i) for i in range(2)]

    class_weights = 0.5 * np.sum(label_counts) / label_counts
    class_weights = np.around(class_weights,2)

    class_weights_dict = dict(enumerate(class_weights))
    
    return class_weights_dict


class_weights_dict = get_class_weights()
params_dict.update({"class_weights": class_weights_dict})


def get_model(dropout_rate): # parameters to be added later: add_dropout = True, dropout_rate = 0.3, add_batch_normalization = True
    
    inputs = keras.layers.Input(shape=(IMGSIZE, IMGSIZE, 1))
    
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
        )(x) # output shape = (IMGSIZE, IMGSIZE, 8)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/4, IMGSIZE/4, 8) 
    
    # Second block 
    x = keras.layers.Conv2D(
        filters=16,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMGSIZE/4, IMGSIZE/4, 16)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/16, IMGSIZE/16, 16) 
    
    # Third block 
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMGSIZE/16, IMGSIZE/16, 32)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/64, IMGSIZE/64, 32) = (4,4,32)
    
    # Head (Flatten + Dense Layer)
    
    x = keras.layers.Flatten()(x) # output shape = 512
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer = None)(x)
    
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    #Final Layer (Output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model


model = get_model(dropout_rate)

metrics = ["accuracy", "f1_score"]

model.compile(
    loss=loss_func, 
    metrics = metrics,
    optimizer = optimizer
    )

# print model summary
model.summary()

# get model summary as string
buffer = io.StringIO()
model.summary(print_fn=lambda x: buffer.write(x + '\n'))
summary_str = buffer.getvalue()



start = time.time()

# 188 sec for 5 epochs => ~ 37.6 s/epoch

history = model.fit(
    train_data, 
    epochs = EPOCHS, 
    verbose = True,
    class_weight = class_weights_dict,
    validation_data = val_data,
    # callbacks = [lr_reduction, model_checkpoint_callback]
    )


print(f'Training time = {time.time() - start} sec')


validation_metrics = ['val_' + metric for metric in metrics]
validation_metrics_values = [history.history['val_' + metric][-1] for metric in metrics] # mlflow logs only numbers/strings for metrics, not lists
metrics_dict = dict(zip(validation_metrics, validation_metrics_values)) # for mlflow logging


def learning_curve_fig(metric):
    # returns learning curve figures -> use for figure logging
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label='Train ' + metric )
    ax.plot(history.history['val_' + metric], label='Validation ' + metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()
    ax.set_title("Training and Validation " + metric)
    plt.close(fig)
    return fig

fig_accuracy = learning_curve_fig(metrics[0])
fig_f1_score = learning_curve_fig(metrics[1])

prediction_example = model(input_example).numpy() # for mlflow logging

   
# First step: run "mlflow server --host 127.0.0.1 --port 8080" in a different terminal to open the server; 
# Make sure that both the python script and rhe mlflow server command are ran from the same folder !!!!!! (in the current case, the folder is "own_model_training") 
# When http://127.0.0.1:8080 displays nonsense, one can try to do a hard refresh while on the webpage with Crtl+Shift+R (worked for me)   

if mlflow_logging: 
    
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Sets the current active experiment to the "own_model_training" experiment and
    # returns the Experiment metadata
    mlflow.set_experiment("convolutional_net_training")


    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.

    run_name = "cnn_train_2"

    # Define an artifact path that the model will be saved to.
    artifact_path = "cnn_artifacts"


    # Start an MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        
        # Log the hyperparameters
        mlflow.log_params(params_dict)

        # Log the metrics
        mlflow.log_metrics(metrics_dict)
        
        # Log figures
        mlflow.log_figure(fig_accuracy, "learning_curve_acc.png")
        mlflow.log_figure(fig_f1_score, "learning_curve_f1.png")

        # log model summary as text artifact
        mlflow.log_text(summary_str, "model_summary.txt")

        # Set a tag that we can use to remind ourselves what this run was for
        
        mlflow.set_tag("Training Info", "2nd run: added class weighting and a dropout layer after the dense layer")
        
        # Infer the model signature
        signature = mlflow.models.infer_signature(input_example, prediction_example)

        # Log the model
        model_info = mlflow.keras.log_model(
                model = model,
                artifact_path = artifact_path,
                signature = signature
            )
