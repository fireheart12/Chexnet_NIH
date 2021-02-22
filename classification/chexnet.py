import os
import glob
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import  confusion_matrix, f1_score, precision_score, recall_score, classification_report
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.utils import plot_model

train = pd.DataFrame(pd.read_csv("/media/HHD_2TB/baurai/chexnet/dataset/train.csv"))
val = pd.DataFrame(pd.read_csv("/media/HHD_2TB/baurai/chexnet/dataset/val.csv"))

def build_decoder(with_labels = True, target_size = (224, 224), ext = "jpg") : 
    def decode(path) : 
        file_bytes = tf.io.read_file(path)
        if ext == "jpg" or ext == "jpeg" : 
            image = tf.image.decode_jpeg(file_bytes, channels = 3)
        elif ext == "png" : 
            image = tf.image.decode_png(file_bytes, channels = 3)
        else:
            raise ValueError("Image extension not supported.")
        
        image = tf.cast(image, tf.float32)/255.0
        image = tf.image.resize(image, target_size)
        return image
    
    def decode_with_labels(path, label) : 
        decoded_image, label = decode(path), label
        return decoded_image, label
    
    
    if with_labels == True : 
        return decode_with_labels
    else:
        return decode

def build_augmenter(with_labels = True) : 
    def augment(image) : 
        image = tf.image.random_flip_left_right(image)
        return image
    
    def augment_with_labels(image, label) : 
        augmented_image, label = augment(image), label
        return augmented_image, label
    
    if with_labels == True : 
        return augment_with_labels
    else:
        return augment

def build_dataset(paths, labels = None, decode_function = None, augment = True, augmenter_function = None, 
                  cache_dir = "", cache = True, shuffle = 1024, repeat = True, batch_size = 16) :
    if cache_dir != "" and cache == True : 
        os.makedirs(cache_dir, exist_ok = True)
    
    if decode_function is None : 
        decode_function = build_decoder(labels is not None)
    
    if augmenter_function is None : 
        augmenter_function = build_augmenter(labels is not None)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    slices = None
    if len(labels) != 0 : 
        slices = (paths, labels)
    else:
        slices = paths
    
    dataset = tf.data.Dataset.from_tensor_slices(slices)
    dataset = dataset.map(decode_function, num_parallel_calls = AUTOTUNE)
    if cache == True : 
        dataset = dataset.cache(cache_dir)
    if augment == True : 
        dataset = dataset.map(augmenter_function, num_parallel_calls = AUTOTUNE)
    if repeat == True : 
        dataset = dataset.repeat()
    if shuffle : 
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    #print(list(dataset.as_numpy_iterator()))
    return dataset

config = {
    "IMAGE_SIZE" : (224, 224),
    "BATCH_SIZE" : 16,
    "CACHE_DIR" : "/media/HHD_2TB/baurai/chexnet/tf_cache/cache"
}

if os.path.exists(config["CACHE_DIR"]) : 
    shutil.rmtree(config["CACHE_DIR"])
os.mkdir(config["CACHE_DIR"])

decoder = build_decoder(with_labels = True, target_size = config["IMAGE_SIZE"])

train_paths = train.image_path.values
train_labels = train.label.values
    
dataset_train = build_dataset(paths = train_paths, labels = train_labels, batch_size = config["BATCH_SIZE"], decode_function = decoder,
                             cache_dir = config["CACHE_DIR"])

val_paths = val.image_path.values
val_labels = val.label.values

dataset_val = build_dataset(paths = val_paths, labels = val_labels, batch_size = config["BATCH_SIZE"], decode_function = decoder,
                           augment = False, shuffle = False, repeat = False, cache_dir = config["CACHE_DIR"])

steps_per_epoch_train = len(train_paths) // config["BATCH_SIZE"]
steps_per_epoch_val = len(val_paths) // config["BATCH_SIZE"]

print(f"Steps per epoch / Train = {steps_per_epoch_train}")
print(f"Steps per epoch / Val = {steps_per_epoch_val}")

# add to config
config["STEPS_PER_EPOCH_TRAIN"] = steps_per_epoch_train
config["STEPS_PER_EPOCH_VAL"] = steps_per_epoch_val

config["EPOCHS"] = 500

print("Configuration file = ", config)

others_count = 70188
nodule_mass_count = 7817

total_cases = others_count + nodule_mass_count

weight_others = (total_cases/others_count)
weight_nodule_mass = (total_cases/nodule_mass_count)


class_weight = {0 : weight_others , 1 : weight_nodule_mass}

print("Weight for other cases = ", class_weight[0])
print("Weight for nodule_mass cases = ", class_weight[1])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("./chexnet/my_chexnet_model.h5", save_best_only = True, monitor = "val_auc", mode = "max")
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True, monitor = "val_auc", mode = "max")
lr_reducer_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_auc", patience = 5, min_lr = 1e-7, mode = "max")

# defining tensorboard callback
root_logdir = "/media/HHD_2TB/baurai/chexnet/my_logs"
if os.path.exists(root_logdir) : 
    shutil.rmtree(root_logdir)
os.mkdir(root_logdir)

def get_run_logdir() : 
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir =  get_run_logdir() 

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

with tf.device("/device:GPU:0"):
    base_model = tf.keras.applications.DenseNet121(weights = "imagenet", include_top = False, 
                                                   input_shape = (config["IMAGE_SIZE"][0], config["IMAGE_SIZE"][1], 3))
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    dropout_1 = tf.keras.layers.Dropout(0.5)(global_pooling)
    output = tf.keras.layers.Dense(units = 1, activation = "sigmoid", name = "predictions")(dropout_1)
    
    model = tf.keras.Model(base_model.input, output)
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = [tf.keras.metrics.AUC(multi_label = False)])
    print(model.summary())
    
    # visualize
    #plot_model(model, to_file = "./models_visuals/my_chexnet_model_plot.png", show_shapes = True, show_layer_names = True)

tf.keras.backend.clear_session()
with tf.device("/device:GPU:0") :
    history = model.fit(dataset_train, steps_per_epoch = config["STEPS_PER_EPOCH_TRAIN"],
                        validation_data = dataset_val, validation_steps = config["STEPS_PER_EPOCH_VAL"],
                        epochs = config["EPOCHS"], verbose = 1,
                        callbacks = [checkpoint_cb, early_stopping_cb, lr_reducer_cb, tensorboard_cb],
                        class_weight = class_weight
                       )