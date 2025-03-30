import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query
from enum import Enum
import mlflow
from api_helpers_v2 import resize_image, load_model_from_registry, make_prediction, return_verified_image_as_numpy_arr, get_modelversion_and_tag, get_performance_indicators, save_performance_data, generate_performance_summary
import time

"middleware neu eingebaut für neuen endpoint"
from fastapi.middleware.cors import CORSMiddleware



""" 
run app by running "fastapi run FastAPIserver.py" in terminal.
Go to localhost = 127.0.0.1. Add "/docs" to url to get to API-frontend and check endpoints!
DOES NOT WORK when called from directory of this script! 
Has to be called from parent directory via: uvicorn api_server:app --host 0.0.0.0 --port 8000 (127.0.0.1 for local)
"""


' ############################### helper class for label input #################'
# class for input in uploading-endpoint
class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

' ############################### creating app  ################################'
# make app
app = FastAPI(title = "Deploying an ML Model for Pneumonia Detection")

" ################################ new middleware block for frontend-suitable endpoint ###############"
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

' ############################### frontend-suitable endpoint ###############################'
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
        
        # set experiment name for model (logging performance for each model in separate experiment)
        mlflow.set_experiment(f"performance {alias}")
        
        # logging of metrics
        with mlflow.start_run():
            
            # log the metrics
            metrics_dict = {
                "y_true": label,
                "y_pred": y_pred,
                "accuracy": int(label == np.around(y_pred))
                }
            mlflow.log_metrics(metrics_dict)

            # log model version and tag
            params = {
                "model version": model_version,
                "model tag": model_tag
                }
            mlflow.log_params(params)

        # logging in csv-files
        save_performance_data(alias = alias, y_true = label.value, y_pred = y_pred, accuracy=int(label == np.around(y_pred)), filename="123.jpeg", model_version=model_version, model_tag=model_tag)

        # update dictionary for API-output
        y_pred_as_str.update({f"prediction {alias}": str(y_pred)})
    
    return y_pred_as_str




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
        
        # set experiment name for model (logging performance for each model in separate experiment)
        mlflow.set_experiment(f"performance {alias}")
        
        # logging of metrics
        with mlflow.start_run():
            
            # log the metrics
            metrics_dict = {
                "y_true": label,
                "y_pred": y_pred,
                "accuracy": int(label == np.around(y_pred))
                }
            mlflow.log_metrics(metrics_dict)

            # log model version and tag
            params = {
                "model version": model_version,
                "model tag": model_tag
                }
            mlflow.log_params(params)

        # logging in csv-files
        save_performance_data(alias = alias, y_true = label.value, y_pred = y_pred, accuracy=int(label == np.around(y_pred)), filename="123.jpeg", model_version=model_version, model_tag=model_tag)

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


' ################################ host specification ################# '

# my localhost adress
host = "127.0.0.1"

# run server
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=8000, root_path="/") 
# GUI at http://127.0.0.1:8000/docs


