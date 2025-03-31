import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
import os
from fastapi import HTTPException
from mlflow import MlflowClient
import csv
import os
import datetime
import time


def resize_image(
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
        img_array_tuple = tuple([image_array for i in range(signature_shape[-1])])
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


def load_model_from_registry(model_name, alias):
    
    # start_loading = time.time()
    print("start loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")
    # end_loading = time.time()
    # print("loading time: ", end_loading - start_loading)
    print("model loaded")
    # extract signature
    signature = model.metadata.signature
    input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
    input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']
    
    return model, input_shape, input_type

def make_prediction(model, image_as_array):
    
    prediction = model.predict(image_as_array)
    pred_reshaped = float(prediction.flatten())

    return pred_reshaped


def return_verified_image_as_numpy_arr(image_bytes):
        try: 
            
            # convert bytes to a PIL image, then ensure its integrity
            image = Image.open(io.BytesIO(image_bytes))
            image.verify() # can't be used if i want to process the image
        
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        
        # load image again (as it has been deconstructed by .verify())
        validated_image = Image.open(io.BytesIO(image_bytes))

        # convert the PIL image to np.array
        validated_image_as_numpy = np.asarray(validated_image)
        return validated_image_as_numpy

def get_modelversion_and_tag(model_name, model_alias):

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # get path of model folder and aliases subfolder (both are used later)
    aliases_path = os.path.join(project_folder ,f"unified_experiment/mlruns/models/{model_name}/aliases")
    
    # aliases_path = os.path.abspath(os.path.join("..",f"unified_experiment/mlruns/models/{model_name}/aliases"))
    model_path = os.path.dirname(aliases_path)
    
    
    # find alias (containing version number) in aliases subfolder, read version number from found file
    alias_file = os.path.join(aliases_path, model_alias)
    with open(alias_file, 'r') as file:
        version_number = int(file.read().strip())
        
    # use version number to find folder of model version
    version_dir = os.path.join(model_path, f"version-{version_number}")
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Folder {version_dir} does not exist")
    
    # find subfolder containing tags
    tags_dir = os.path.join(version_dir, "tags")
    if not os.path.exists(tags_dir):
        raise FileNotFoundError(f"folder 'tags' in {version_dir} is missing")
    
    # extract titel from first tag-file found
    tag_files = os.listdir(tags_dir)
    if not tag_files:
        raise FileNotFoundError(f"No tags found in {tags_dir}")
    
    return version_number, tag_files[0].strip()

def get_performance_indicators(num_steps_short_term = 1):

    # setting the uri 
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
        
    # get the three experiments: performance + (baseline, challenger, champion)
    print("getting experiment list")
    experiments = list(client.search_experiments())[:3]
    
    # get experiment names and ids
    print("getting experiment names and ids")
    exp_names = [exp.name for exp in experiments]
    exp_ids = [exp.experiment_id for exp in experiments]
    
    # define an empty dictionary to hold the performance indicators for each experiment
    performance_dictionary = {}
    
    # for loop to calculate perfomance indicators for each experiment/model
    for exp_name, exp_id in zip(exp_names, exp_ids):
        print(f"{exp_name}: getting runs")
        # all runs in the experiment with exp_id, i.e. number of predictions made
        runs = list(client.search_runs(experiment_ids = exp_id))
        
        # run_ids
        run_ids = [run.info.run_id for run in runs]
        
        # extract lists of accuracies, timestamps, and correct prediction labels
        # within the given experiment (0 = no pneumonia, 1 = pneumonia)
        print(f"{exp_name}: starting extraction of accuracies")
        accuracies = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].value for run_id in run_ids]
        print(f"{exp_name}: starting extraction of time stamps")
        timestamps = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].timestamp for run_id in run_ids]
        print(f"{exp_name}: starting extraction of input_labels")
        y_true = [list(client.get_metric_history(run_id = run_id, key = 'y_true'))[0].value for run_id in run_ids]
        
        # 1st row is timestamps, 2nd is accuracies and so on
        values_array = np.array([timestamps, accuracies, y_true])
        # sorts according to the timestamps
        values_array = values_array[:, values_array[0].argsort()]
        # get rid of the timestamps row
        values_array = values_array[1:]
        
        print(f"{exp_name}: calc confusion matrix")
        # calculate confusion matrix elements
        true_positives = np.sum((values_array[0] == 1) & (values_array[1] == 1))
        true_negatives = np.sum((values_array[0] == 1) & (values_array[1] == 0))
        false_negatives = np.sum((values_array[0] == 0) & (values_array[1] == 1))
        false_positives = np.sum((values_array[0] == 0) & (values_array[1] == 0))
        
        # save the experiment information in a dictionary
        exp_dictionary ={
            'all-time average accuracy': str(np.mean(values_array[0])),
            'total number of predictions': str(len(accuracies)),
            f'average accuracy for the last {num_steps_short_term} predictions': str(np.mean(values_array[0,-num_steps_short_term:])),
            'pneumonia true positives': str(true_positives),
            'pneumonia true negatives': str(true_negatives),
            'pneumonia false positives': str(false_positives), 
            'pneumonia false negatives': str(false_negatives),
             
        }
        
        # update the dictionary containing the information from the other experiments
        performance_dictionary.update({exp_name: exp_dictionary})
          
    return performance_dictionary


def save_performance_data_csv(alias, timestamp, y_true, y_pred, accuracy, filename, model_version, model_tag):
    
    # take time
    start_time = time.time()

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path of folder for performance tracking
    tracking_path = os.path.join(project_folder ,f"unified_experiment/performance_tracking")
    
    # make folder, if not existing yet
    os.makedirs(tracking_path, exist_ok=True)
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')
    
    # initializing standard values for cumulative and global values
    log_counter = 1
    cumulative_accuracy = accuracy
    global_accuracy = accuracy
    last_25_accuracy = accuracy
    
    # Count existing rows to calculate log_counter
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if rows:
                last_row = rows[-1]
                # get values of last row to determine cumulations and global accuracy
                log_counter = int(last_row['log_counter']) + 1
                cumulative_accuracy = float(last_row['cumulative_accuracy']) + accuracy
                global_accuracy = cumulative_accuracy / log_counter
                # get last 24 values (or less, if not enough rows available)
                num_previous = min(24, log_counter - 1)
                relevant_rows = rows[-num_previous:]
                relevant_accuracies = [float(row['accuracy']) for row in relevant_rows] + [accuracy]
                last_25_accuracy = sum(relevant_accuracies) / len(relevant_accuracies)

    # prepare data
    data = {
        'log_counter': log_counter,
        'timestamp': timestamp,
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'cumulative_accuracy': cumulative_accuracy,
        'global_accuracy': global_accuracy,
        'accuracy_last_25_predictions': last_25_accuracy,
        'filename': filename,
        'model_version': model_version,
        'model_tag': model_tag,
        "model_alias": alias
    }
    
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    
    # Open file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        # Write header only if file is newly created
        if not file_exists:
            writer.writeheader()
        # Append new row
        writer.writerow(data)
    
    end_time = time.time()
    print("runtime performance logging: ", end_time-start_time)
    print(f"Data has been saved in {file_path}.")

    return data



def generate_performance_summary(alias):
    
    # get path of csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "unified_experiment/performance_tracking")
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')

    if not os.path.exists(file_path):
        return "Error: CSV file not found."

    # read
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        return "Error: CSV file is empty."

    # get values of last prediction (cumulations, averages)
    last_row = rows[-1]
    total_predictions = int(last_row['log_counter'])
    all_time_average = float(last_row['global_accuracy'])
    last_25_average = float(last_row['accuracy_last_25_predictions'])

    # initialize confusion matrix
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert to numpy
    y_true = np.array([float(row['y_true']) for row in rows])
    accuracy = np.array([float(row['accuracy']) for row in rows])
    # calc confusion matrix
    true_positives = np.sum((y_true == 1) & (accuracy == 1))
    true_negatives = np.sum((y_true == 0) & (accuracy == 1))
    false_positives = np.sum((y_true == 0) & (accuracy == 0))
    false_negatives = np.sum((y_true == 1) & (accuracy == 0))

    # generate result dict
    summary = {
        f"performance csv {alias}": {
            "all-time average accuracy": f"{all_time_average:.4f}",
            "total number of predictions": str(total_predictions),
            "average accuracy last 25 predictions": f"{last_25_average:.4f}",
            "pneumonia true positives": str(true_positives),
            "pneumonia true negatives": str(true_negatives),
            "pneumonia false positives": str(false_positives),
            "pneumonia false negatives": str(false_negatives)
        }
    }

    return summary




# if run locally (for tests)
if __name__ == "__main__":
    # modell laden
    model_name_test = "Xray_classifier"  # Small_CNN, MobileNet_transfer_learning, MobileNet_transfer_learning_finetuned
    model_alias = "baseline"
    
    ' ###################### test: extract version via alias ########### '
    print(get_modelversion_and_tag(model_name=model_name_test, model_alias=model_alias))

    ' #################### test: extract tag of model version #############'
    
    # Bild laden
    img = Image.open(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\NORMAL\IM-0001-0001.jpeg")

    # mlflow setting
    "    mlflow server --host 127.0.0.1 --port 8080     "
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    model, input_shape, input_type = load_model_from_registry(model_name = model_name_test, alias=model_alias)

    formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)

    y_pred = make_prediction(model, image_as_array=formatted_image)

    print(y_pred)
    
    print(get_performance_indicators())
