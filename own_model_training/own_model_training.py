import time
import io

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mlflow
from mlflow import MlflowClient
#from PIL import Image




'''
img = Image.open("./data/train/PNEUMONIA/person2_bacteria_4.jpeg")
img_array = np.array(img) 

print('image shape = ', img_array.shape)
print('max pixel value = ', img_array.max())
'''

# params to log: model_summary (as string), parameters from get_model() function, learning curves + metrics, 
# infer model signature (gives input shape and output shape)

BATCH_SIZE = 128
IMG_SIZE = 256
EPOCHS = 200

loss_func = "binary_crossentropy"
learning_rate = 0.005
momentum = 0.8
optimizer = keras.optimizers.SGD(learning_rate= learning_rate, momentum = momentum)
tag = "Own CNN"

# params for mlflow; will need to add more 
params_dict = {
    "epochs": EPOCHS,
    "loss_function": loss_func,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "momentum": momentum,
    "tag": tag }


# images are grayscale, max pixel value is 255

train_data, val_data = keras.utils.image_dataset_from_directory(
    # relative path to images
    r'/home/anandrei90/pneumonia_project/data/train',
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


# take an input example out of the train data -> needed for mlflow

batch = train_data.take(1)
batch_as_nparray = list(batch)[0][0].numpy()
input_example = batch_as_nparray[0]
input_example = np.expand_dims(input_example, axis=0)

# print('batch shape = ', list(batch)[0][0].numpy().shape) # (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)


def get_model(): # parameters to be added later: add_dropout = True, dropout_rate = 0.3, add_batch_normalization = True
    
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

# TO DO:
# add class weights
# add callbacks


start = time.time()

# 188 sec for 5 epochs => ~ 37.6 s/epoch

history = model.fit(
    train_data, 
    epochs = EPOCHS, 
    verbose = True,
    # class_weight = class_weights_dict,
    validation_data = val_data,
    # callbacks = [lr_reduction, model_checkpoint_callback]
    )


print(f'Training time = {time.time() - start} sec')








# validation_metrics_values = [history.history['val_' + metric] for metric in metrics] # mlflow logs only numbers/strings for metrics, not lists
validation_metrics = ['val_' + metric for metric in metrics]
validation_metrics_values = [history.history['val_' + metric][-1] for metric in metrics] 
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






# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This experiment contains the first attempts at training a self-built CNN to detect Pneumonia from chest-xray images."
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "pneumonia-detection",
    "authors": "andrei & patrick",
    "mlflow.note.content": experiment_description,
}

# Sets the current active experiment to the "own_model_training" experiment and
# returns the Experiment metadata
mlflow.set_experiment("convolutional_net_training")
    
# First step: run "mlflow server --host 127.0.0.1 --port 8080" in a different terminal to open the server
# When http://127.0.0.1:8080 displays nonsense, one can try to do a hard refresh while on the webpage with Crtl+Shift+R (worked for me)   
 
mlflow.set_tracking_uri("http://127.0.0.1:8080")    
    


# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
run_name = "cnn_train_1"

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
    mlflow.set_tag("Training Info", "1st run: no batchnorm and no dropout layers")

    # Infer the model signature
    signature = mlflow.models.infer_signature(input_example, prediction_example)

    # Log the model
    model_info = mlflow.keras.log_model(
            model = model,
            artifact_path = artifact_path,
            signature = signature
        )
