
from tensorflow import keras
import os

IMGSIZE = 256
DATA_PATH = os.path.join("..","data/train/")
DATA_PATH = os.path.abspath(DATA_PATH)

def get_data(batch_size, imgsize, channelmode = "rgb", selected_data = "train"):
    # for training data, do 20% split
    if selected_data == "train":
        chosen_validation_split=0.2
        subset = "both"
        data_path = DATA_PATH 
    # for test data, do 99,9% split. A little bit hacky, but returns generators, which is important.
    # by splitting this way, we throw away 1 image of the test set, and keep the rest.
    elif selected_data == "test":
        chosen_validation_split = 0.999
        subset = "both"
        data_path = os.path.join("..","data/test/")
    train_data, val_data = keras.utils.image_dataset_from_directory(
        # relative path to images
        data_path,
        labels='inferred',              # labels are generated from the directory structure
        label_mode='binary',            # 'binary' => binary cross-entropy
        color_mode=channelmode,         # alternatives: 'rgb', 'rgba'
        batch_size=batch_size,
        image_size=(imgsize, imgsize),
        shuffle=True,                   # shuffle images before each epoch
        seed=0,                         # shuffle seed
        validation_split = chosen_validation_split,
        subset=subset,                  # return a tuple of datasets (train, val)
        interpolation='bilinear',       # interpolation method used when resizing images
        follow_links=False,             # follow folder structure?
        crop_to_aspect_ratio=False
        )
    return train_data, val_data