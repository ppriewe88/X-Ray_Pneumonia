import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from enum import Enum
import mlflow
from api_helpers import resize_image, load_model_from_registry, make_prediction, return_verified_image_as_numpy_arr, get_modelversion_and_tag, get_performance_indicators, save_performance_data_csv, generate_performance_summary, generate_model_comparison_plot
import time
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware # middleware. requirement for frontend-suitable endpoint

import matplotlib.pyplot as plt
import io

""" 
run app by running "fastapi run FastAPIserver.py" in terminal.
Go to localhost = 127.0.0.1. Add "/docs" to url to get to API-frontend and check endpoints!
Works when called from any directory level in project folder. Best, start from subfolder "api". 
Here the explicit call to be run from terminal: uvicorn api_server:app --host 0.0.0.0 --port 8000 (127.0.0.1 for local)
"""


' ############################### helper class for label input #################'
# class for input in uploading-endpoint
class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

' ############################### creating app  ################################'
# make app
app = FastAPI(title = "Deploying an ML Model for Pneumonia Detection")

" ################################ middleware block for frontend-suitable endpoint ###############"
# CORS-Middleware hfor communication with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP-methods
    allow_headers=["*"],  # allow all headers
)

' ############################### root endpoint ###############################'
# root
@app.get("/")
def home():
    return "root of this API"

' ############################### model serving/prediction endpoint ###############################'
# endpoint for uploading image
@app.post("/upload_image")
async def upload_image_and_integer( 
    label: Label,
    file: UploadFile = File(...)
):

    print("label: ", label, "type label: ", type(label))
    # read the uploaded file into memory as bytes
    image_bytes = await file.read()

    # validate image and return as numpy
    img = return_verified_image_as_numpy_arr(image_bytes)

    # set tracking uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # vessel for API-output
    y_pred_as_str = {}
    
    model_name = "Xray_classifier"
    api_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ########################### load, predict, log metric for champion and challenger ################'
    for  alias in ["champion", "challenger", "baseline"]:
        
        # get model and signature
        model, input_shape, input_type  = load_model_from_registry(model_name = model_name, alias = alias)
        
        # get model version and tag for logging
        model_version, model_tag = get_modelversion_and_tag(model_name=model_name, model_alias=alias)

        # resize image according to signature
        formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)
        
        # make prediction
        y_pred = make_prediction(model, image_as_array=formatted_image)
        accuracy_pred = int(label == np.around(y_pred))

        # logging and precalculations in csv-file
        logged_csv_data = save_performance_data_csv(alias = alias, timestamp = api_timestamp, y_true = label.value, y_pred = y_pred, accuracy=accuracy_pred, filename="123.jpeg", model_version=model_version, model_tag=model_tag)

        # set experiment name for model (logging performance for each model in separate experiment)
        mlflow.set_experiment(f"performance {alias}")
    
        # logging of metrics
        with mlflow.start_run():
            
            # log the metrics
            metrics_dict = {
                'log counter': logged_csv_data["log_counter"],
                "y_true": label,
                "y_pred": y_pred,
                "accuracy": accuracy_pred,
                'global accuracy': logged_csv_data["global_accuracy"],
                'floating avg accuracy 50 runs': logged_csv_data["accuracy_last_50_predictions"]
                }
            mlflow.log_metrics(metrics_dict)

            # log model version and tag
            params = {
                'timestamp': logged_csv_data["timestamp"],
                "model version": model_version,
                "model tag": model_tag,
                'image file name': logged_csv_data["filename"],
                }
            mlflow.log_params(params)

        # update dictionary for API-output
        y_pred_as_str.update({f"prediction {alias}": str(y_pred)})
    
    return y_pred_as_str

' ############################### frontend-suitable model serving/prediction endpoint ###############################'
# endpoint for uploading image
@app.post("/upload_image_from_frontend")
async def upload_image_and_integer( 
    label: int = Form(...),
    file: UploadFile = File(...)
):

    print("label: ", label, "type label: ", type(label))
    label = Label(label)
    print("label: ", label, "type label: ", type(label))

    # read the uploaded file into memory as bytes
    image_bytes = await file.read()

    # validate image and return as numpy
    img = return_verified_image_as_numpy_arr(image_bytes)

    # set tracking uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # vessel for API-output
    y_pred_as_str = {}
    
    model_name = "Xray_classifier"
    api_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ########################### load, predict, log metric for champion and challenger ################'
    for  alias in ["champion", "challenger", "baseline"]:
        
        # get model and signature
        model, input_shape, input_type  = load_model_from_registry(model_name = model_name, alias = alias)
        
        # get model version and tag for logging
        model_version, model_tag = get_modelversion_and_tag(model_name=model_name, model_alias=alias)

        # resize image according to signature
        formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)
        
                # make prediction
        y_pred = make_prediction(model, image_as_array=formatted_image)
        accuracy_pred = int(label == np.around(y_pred))

        # logging and precalculations in csv-file
        logged_csv_data = save_performance_data_csv(alias = alias, timestamp = api_timestamp, y_true = label.value, y_pred = y_pred, accuracy=accuracy_pred, filename="123.jpeg", model_version=model_version, model_tag=model_tag)

        # set experiment name for model (logging performance for each model in separate experiment)
        mlflow.set_experiment(f"performance {alias}")
    
        # logging of metrics
        with mlflow.start_run():
            
            # log the metrics
            metrics_dict = {
                'log counter': logged_csv_data["log_counter"],
                "y_true": label,
                "y_pred": y_pred,
                "accuracy": accuracy_pred,
                'global accuracy': logged_csv_data["global_accuracy"],
                'floating avg accuracy 50 runs': logged_csv_data["accuracy_last_50_predictions"]
                }
            mlflow.log_metrics(metrics_dict)

            # log model version and tag
            params = {
                'timestamp': logged_csv_data["timestamp"],
                "model version": model_version,
                "model tag": model_tag,
                'image file name': logged_csv_data["filename"],
                }
            mlflow.log_params(params)

        # update dictionary for API-output
        y_pred_as_str.update({f"prediction {alias}": str(y_pred)})
    
    return y_pred_as_str

' ############################### performance review endpoint ###############################'
# endpoint for uploading image
@app.post("/get_performance_review")
async def get_performance(
    last_n_predictions: int,
    ):
    

    # gets the dictionary for all three model
    start_time1 = time.time()
    perf_dict = get_performance_indicators(num_steps_short_term = last_n_predictions)
    end_time1 = time.time()
    time_old_review = end_time1 - start_time1


    # in addition, show results generated from csv
    start_time2 = time.time()
    csv_perf_dict_champion = generate_performance_summary("champion")
    csv_perf_dict_challenger = generate_performance_summary("challenger")
    csv_perf_dict_baseline = generate_performance_summary("baseline")
    merged_csv_dict = {
    **csv_perf_dict_baseline,
    **csv_perf_dict_challenger,
    **csv_perf_dict_champion,
    }
    end_time2 = time.time()
    time_new_review = end_time2 - start_time2

    # generate response
    response = {
    "old_review": perf_dict,
    "runtime old review": str(time_old_review),
    "new_review": merged_csv_dict,
    "runtime new review": time_new_review
    }

    return response


' ############################### performance review endpoint CSV ###############################'
# endpoint for uploading image
@app.post("/get_performance_review_from_csv")
async def get_performance():

    # get results generated from csv
    start_time2 = time.time()
    csv_perf_dict_champion = generate_performance_summary("champion")
    csv_perf_dict_challenger = generate_performance_summary("challenger")
    csv_perf_dict_baseline = generate_performance_summary("baseline")
    merged_csv_dict = {
    **csv_perf_dict_baseline,
    **csv_perf_dict_challenger,
    **csv_perf_dict_champion,
    }
    end_time2 = time.time()
    time_new_review = end_time2 - start_time2

    response = {
    "new_review": merged_csv_dict,
    "runtime new review": time_new_review
    }
    return response


#####################################################
# endpoint for plot generation
@app.post("/get_comparsion_plot")
async def plot_model_comparison(target = "accuracy_last_50_predictions"):
    
    '''
    Endpoint that displays a plot showing the moving average accuracy
    of the champion and challenger models. 
    '''
    
    # TODO: need to change this to reflect the (potential) updates in the 
    # generate_model_comparison_plot() function
    
    # create the figure
    figure = generate_model_comparison_plot(target, scaling =  "log_counter")
    
    
    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()
    # save the plot in the buffer as a png
    plt.savefig(buffer, format="png")
    # close the fig
    plt.close(figure)
    # move the file pointer back to the start of the buffer so it can be read
    buffer.seek(0)
    
    # extract the binary image from the buffer
    binary_image = buffer.getvalue()
    
    # send the binary image as a png response to the client
    return Response(binary_image, media_type="image/png")



' ################################ host specification ################# '

# my localhost adress
host = "127.0.0.1"

# run server
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=8000, root_path="/")
# GUI at http://127.0.0.1:8000/docs


