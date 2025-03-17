import os
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np

def generate_data(chosen_test_size):
    """Used for generating test data. Does the download from sklearn, does onehot encoding, scaling, and separates validation data."""
    # get currenttime to save model if necessary
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    #%% import
    digits = load_digits(return_X_y=True)
    data = digits[0]
    targets = digits[1]
    1=2
    # blablabla

    #%%  Onehotencoding
    onehot = OneHotEncoder(sparse_output=False)
    targets = targets.reshape(-1, 1)
    onehot.fit(targets)
    targets = onehot.transform(targets)

    # %%  scaling
    minmax = MinMaxScaler()
    data = minmax.fit_transform(data)

    # %%  train test split
    x_train, x_val, y_train, y_val = train_test_split(data, targets, test_size=chosen_test_size, random_state=0)

    # save data
    folder_name = f"prepared_data"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    file_name = f"data_{current_time}.pkl"

    data_path = os.path.join(folder_name, file_name)
    data = {"x_train": x_train,
            "x_val": x_val,
            "y_train": y_train,
            "y_val": y_val,
            "scaler": minmax}
    
    with open(data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"Data file {file_name} was saved under {data_path}. Contains train and val data, and scaler.")

    return data_path, file_name

def load_training_data(custom_file = None, get_from_location = "data/prepared_data"):
    """Used in train_model.py.
    Loads specific set of training and validation data. location "data" is default. If specific file is desired, specify custom_file name. Else, latest data is taken (via time stamp)"""
    ################################## get data
    # folder name of data
    location = get_from_location

    # If custom_file is given, then load custom file. Else, load latest data (via timestamp)

    # get path of latest file
    if custom_file == None:
        # check data folder for pickle files
        data_files = [f for f in os.listdir(location) if f.endswith("pkl")]
        if not data_files:
            raise FileNotFoundError("No data file found under specified path!")
        # sort found files to get latest (according to timestamp)
        data_files.sort()
        selected_file = data_files[-1]
    else:
        selected_file = custom_file # name of one of the data files. replace by inserting file name, if neccessary
    
    # create absolute path
    data_path = os.path.join(location, selected_file)
    
    # load pickle data from location
    with open(data_path, "rb") as file:
        loaded_data = pickle.load(file)
    
    x_train = loaded_data["x_train"]
    y_train = loaded_data["y_train"]
    x_val = loaded_data["x_val"]
    y_val = loaded_data["y_val"]
    scaler = loaded_data["scaler"]
    
    return x_train, y_train, x_val, y_val, scaler, data_path

def save_digit_as_JPEG(img_array, filename="digit.jpeg"):
    """Used to generate images of validation set (for the sake of testing)"""

    # reshaping from 64 to 8x8
    img_array = img_array.reshape(8,8)

    # scale back to 0 to 255
    if img_array.max() <= 1.0:  # check if normalized (holds true for train/test data!)
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # converting
    img = Image.fromarray(img_array)

    # create output folder
    output_folder = "validation_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # creating file name
    filename = os.path.join(output_folder, filename)
    
    # saving
    img.save(filename, "JPEG")


def convert_JPEG_to_numpy(image_path):
    
    # check path validity
    if not os.path.exists(image_path):
        raise FileNotFoundError("File passed to convert_JPEG_to_numpy not found!")
    
    # open image
    with Image.open(image_path) as img:
        if img.format != "JPEG":
            raise ValueError("image format not supported. Please pass JPEG image!")    
        # convert to numpy array
        image_array = np.array(img)

    if image_array.shape == ():
        raise ValueError(f"Converting failed. Array-Shape: {image_array.shape}")

    return image_array