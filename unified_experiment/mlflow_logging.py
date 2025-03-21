
import os

import mlflow
import numpy as np


DATA_PATH = os.path.join("..","data/train/")
DATA_PATH = os.path.abspath(DATA_PATH)


def log_mlflow_run(
    model, # keras model to be logged
    run_name, # string that will be displayed as the run title in mlflow GUI
    epochs,
	batch_size,
	loss_function,
	optimizer, # string -> AA: or object, in my case
	learning_rate,
	top_dropout_rate,
	model_summary_string, # string for model summary (comes from a helper function)
    run_tag, # string explaining what this run was for
    signature_batch, # needed for infer_signature
    val_accuracy,
    test_accuracy,
	custom_params, # must be a dictionary (eg for momentum, activation functions in the top layer)
    fig # in case there are more figs
):
    
    params_dict = {
        "epochs": epochs,
        "batch size": batch_size,
        "loss function": loss_function,
        "optimizer": optimizer,
        "learning rate": learning_rate,
        "dense layer dropout rate": top_dropout_rate,
        "dataset": DATA_PATH,
        }
    
    try:
        params_dict.update(custom_params)
    except:
        raise TypeError("custom_params can only be a dictionary.")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Sets the current active experiment to the "own_model_training" experiment and
    # returns the Experiment metadata
    mlflow.set_experiment("X-Ray Pneumonia")


    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        
        # Log the hyperparameters
        mlflow.log_params(params_dict)

        # Log the metrics
        
        metrics_dict = {
            "validation accuracy": val_accuracy,
            "test accuracy": test_accuracy
            }
        
        mlflow.log_metrics(metrics_dict)
        
        # Log figures
        mlflow.log_figure(fig, "learn_curve_accuracy.png")

        # log model summary as text artifact
        mlflow.log_text(model_summary_string, "model_summary.txt")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", run_tag)
        
        # Infer the model signature
        
        batch_as_nparray = list(signature_batch)[0][0].numpy()
        input_example = batch_as_nparray[0]
        input_example = np.expand_dims(input_example, axis=0)
        
        prediction_example = model(input_example).numpy()
                
        signature = mlflow.models.infer_signature(input_example, prediction_example)

        # Log the model
        mlflow.keras.log_model(
            model = model,
            artifact_path = "model_artifact",
            signature = signature
            )	