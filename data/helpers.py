from tensorflow import keras
import os

IMGSIZE = 256
DATA_PATH_TRAIN = os.path.join("..","data/train/")
DATA_PATH_TRAIN = os.path.abspath(DATA_PATH_TRAIN)

DATA_PATH_TEST = os.path.join("..","data/test/")
DATA_PATH_TEST = os.path.abspath(DATA_PATH_TEST)

def get_train_val_data(batch_size, img_size, channel_mode):
    train_data, val_data = keras.utils.image_dataset_from_directory(
        DATA_PATH_TRAIN,
        labels='inferred',              # labels are generated from the directory structure
        label_mode='binary',            # 'binary' => binary cross-entropy
        color_mode=channel_mode,         # alternatives: 'rgb', 'rgba'
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,                   # shuffle images before each epoch
        seed=0,                         # shuffle seed
        validation_split = 0.2,
        subset="both",                  # return a tuple of datasets (train, val)
        interpolation='bilinear',       # interpolation method used when resizing images
        follow_links=False,             # follow folder structure?
        crop_to_aspect_ratio=False
        )
    return train_data, val_data
    
    
def get_test_data(batch_size, img_size, channel_mode):
    test_data = keras.utils.image_dataset_from_directory(
        DATA_PATH_TEST,
        labels='inferred',              # labels are generated from the directory structure
        label_mode='binary',            # 'binary' => binary cross-entropy
        color_mode=channel_mode,         # alternatives: 'rgb', 'rgba'
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,                   # shuffle images before each epoch
        seed=0,                         # shuffle seed
        validation_split = None,
        subset=None,                   
        interpolation='bilinear',       # interpolation method used when resizing images
        follow_links=False,             # follow folder structure?
        crop_to_aspect_ratio=False
        )
    return test_data