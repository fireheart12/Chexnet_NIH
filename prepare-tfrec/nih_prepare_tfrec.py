import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import albumentations as A
import tensorflow as tf

print(f"Python version = {sys.version}")
print(f"Numpy version = {np.__version__}")
print(f"OpenCV version = {cv2.__version__}")
print(f"Tensorflow version = {tf.__version__}")

df = pd.DataFrame(pd.read_csv("filtered_metadata/metadata.csv"))

IMAGE_SIZE = (512, 512)

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.Rotate(limit = 45, p = 0.5)
])

def apply_albumin_augmentation(image, transform) : 
    augmented_image = transform(image = image)["image"]
    return augmented_image

SIZE = 500

num_of_images = df.shape[0]
image_names = df["image_id"].values

"""
Stratified Sampling : 

Stratified sampling refers to a type of sampling method . With stratified sampling, we divides the population into separate groups, called strata. Then, a simple random sample is drawn from each group.

"""
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 42)
for train_index, val_cum_test_index in split.split(df, df["label"]) : 
    stratified_train_set = df.loc[train_index]
    stratified_val_cum_test_set = df.loc[val_cum_test_index]

stratified_train_set.reset_index(drop = True, inplace = True)
print("Size of Stratified Training set = ", stratified_train_set.shape)

stratified_val_cum_test_set.reset_index(drop = True, inplace = True)
print("Size of Stratified Validation cum Testing set = ", stratified_val_cum_test_set.shape)

"""
We need to further split stratified_val_cum_test_set into **stratified_validation_set and stratified_test_set.
"""

split2 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 42)
for val_index, test_index in split2.split(stratified_val_cum_test_set, stratified_val_cum_test_set["label"]) : 
    stratified_val_set = stratified_val_cum_test_set.loc[val_index]
    stratified_test_set = stratified_val_cum_test_set.loc[test_index]

stratified_val_set.reset_index(drop = True, inplace = True)
print("Size of Stratified Validation set = ", stratified_val_set.shape)
stratified_val_set.head()

stratified_test_set.reset_index(drop = True, inplace = True)
print("Size of Stratified Test set = ", stratified_test_set.shape)
stratified_test_set.head()

# Analyze Label Distribution In Stratified Training/Val/Test Set
def analyze_label_distribution(dataframe, case = "train") : 
    """
    Training, validation and testing frames are passed to this method.
    """
    label_count = dict()
    for label in tqdm(dataframe["label"].values) : 
        if label not in label_count : 
            label_count[label] = 1
        else:
            label_count[label] += 1
    
    print("Healthy cases count = ", label_count[0])
    print("Nodular cases count = ", label_count[1])

analyze_label_distribution(stratified_train_set)
analyze_label_distribution(stratified_val_set)
analyze_label_distribution(stratified_test_set)

# TFRec Preparation
"""
We will also create a metadata file to store all the new info, as we will be expanding the dataset post augmentation too.

Columns : [image_name, label, subset] 

subset will store whether the image is for training/val/testing.
"""
metadata_bin_clf_tfrec = pd.DataFrame(columns = ["image_name", "label", "subset"])

training_image_ids = stratified_train_set["image_id"].values
validation_image_ids = stratified_val_set["image_id"].values
testing_image_ids = stratified_test_set["image_id"].values

print(f"{len(training_image_ids)} training, {len(validation_image_ids)} validation and {len(testing_image_ids)} testing images located!")

PATH = "/media/HHD2/NIH/tflow_obj_detection/images/"

def bytes_features(value) : 
    if isinstance(value, type(tf.constant(0))) : 
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def float_features(value) : 
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def int64_features(value) : 
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
"""
A TFRecord files is a tensor of records, which are represented using Example data structure. **This data structure is nothing more than a compilation of faetures which map the OG string features to TF compatible feature format**.

This is what we are gonna do now. We define a set of features and encapsulate them in an Example data structure.
"""

def serialize_example(feature_list) : 
    # feature_list = [image, image_name, label]
    feature = {
        "image" : bytes_features(feature_list[0]),
        "image_name" : bytes_features(feature_list[1]),
        "label" : int64_features(feature_list[2])
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

total_chunks = len(training_image_ids)//SIZE + int(len(training_image_ids)%SIZE != 0)

print("Total training record to be prepared = ", total_chunks)

dataframe_index = 0 # for appending entry to the meta dataframe

# Train
count_healthy = 0 
for j in tqdm(range(total_chunks)) : 
    print("\nWriting Training TFRecord %i of %i"%(j+1, total_chunks))
    count = min(SIZE, len(training_image_ids) - (j * SIZE))
    with tf.io.TFRecordWriter("./tfrec_v3/train%.2i.tfrec"%(j)) as writer : 
        c = 0 
        for k in range(count) : 
            image_id = training_image_ids[(SIZE * j) + k]
            image_path = PATH + image_id
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[0]), interpolation = cv2.INTER_NEAREST)
            
            
            if stratified_train_set.loc[stratified_train_set.image_id == image_id].values[0][1] == 1 :
                t = 0 
                for t in range(2) : 
                    image_copy = image.copy()
                    image_copy = apply_albumin_augmentation(image_copy, transform)
                    image_copy = cv2.imencode(".jpg", image_copy, (cv2.IMWRITE_JPEG_QUALITY, 95))[1].tostring()
                    name_with_extension = image_id
                    name_without_extension = name_with_extension.split(".")[0] + "_" + str(t)
                    row = stratified_train_set.loc[stratified_train_set.image_id == name_with_extension]
                    feature_list = [image_copy, str.encode(name_without_extension), row.label.values[0]]
                    example = serialize_example(feature_list)
                    writer.write(example)
                    
                    metadata_bin_clf_tfrec.loc[dataframe_index] = [name_without_extension + ".png", row.label.values[0], "training"]
                    dataframe_index = dataframe_index + 1
                    
                    c += 1
                    
                    if c %100 == 0 :
                        print(c, ",", end = " ")
            else:
                count_healthy = count_healthy + 1
                if count_healthy % 2 == 0 : 
                    image = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 95))[1].tostring()
                    name_with_extension = image_id
                    name_without_extension = name_with_extension.split(".")[0]
                    row = stratified_train_set.loc[stratified_train_set.image_id == name_with_extension]
                    feature_list = [image, str.encode(name_without_extension), row.label.values[0]]
                    example = serialize_example(feature_list)
                    writer.write(example)
                
                    metadata_bin_clf_tfrec.loc[dataframe_index] = [name_without_extension + ".png", row.label.values[0], "training"]
                    dataframe_index = dataframe_index + 1
                    
                    c += 1
                
                    if c %100 == 0 :
                        print(c, ",", end = " ")

# Validation
total_chunks = len(validation_image_ids)//SIZE + int(len(validation_image_ids)%SIZE != 0)

print("Total val record to be prepared = ", total_chunks)

for j in tqdm(range(total_chunks)) : 
    print("\nWriting Validation TFRecord %i of %i"%(j+1, total_chunks))
    count = min(SIZE, len(validation_image_ids) - (j * SIZE))
    with tf.io.TFRecordWriter("./tfrec_v3/val%.2i.tfrec"%(j)) as writer : 
        c = 0 
        for k in range(count) : 
            image_id = validation_image_ids[(SIZE * j) + k]
            image = cv2.imread(PATH + image_id)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[0]), interpolation = cv2.INTER_NEAREST)
            image = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 95))[1].tostring()
            name_with_extension = image_id
            name_without_extension = name_with_extension.split(".")[0]
            row = stratified_val_set.loc[stratified_val_set.image_id == name_with_extension]
            feature_list = [image, str.encode(name_without_extension), row.label.values[0]]
            example = serialize_example(feature_list)
            writer.write(example)
                
            metadata_bin_clf_tfrec.loc[dataframe_index] = [name_without_extension + ".png", row.label.values[0], "validation"]
            dataframe_index = dataframe_index + 1
                
            c += 1
                
            if c %100 == 0 :
                print(c, ",", end = " ")

# Test
total_chunks = len(testing_image_ids)//SIZE + int(len(testing_image_ids)%SIZE != 0)

print("Total test records to be prepared = ", total_chunks)

for j in tqdm(range(total_chunks)) : 
    print("\nWriting Testing TFRecord %i of %i"%(j+1, total_chunks))
    count = min(SIZE, len(testing_image_ids) - (j * SIZE))
    with tf.io.TFRecordWriter("./tfrec_v3/test%.2i.tfrec"%(j)) as writer : 
        c = 0 
        for k in range(count) : 
            image_id = testing_image_ids[(SIZE * j) + k]
            image = cv2.imread(PATH + image_id)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[0]), interpolation = cv2.INTER_NEAREST)
            image = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 95))[1].tostring()
            name_with_extension = image_id
            name_without_extension = name_with_extension.split(".")[0]
            row = stratified_test_set.loc[stratified_test_set.image_id == name_with_extension]
            feature_list = [image, str.encode(name_without_extension), row.label.values[0]]
            example = serialize_example(feature_list)
            writer.write(example)
                
            metadata_bin_clf_tfrec.loc[dataframe_index] = [name_without_extension + ".png", row.label.values[0], "testing"]
            dataframe_index = dataframe_index + 1    
                
            c += 1
                
            if c %100 == 0 :
                print(c, ",", end = " ")

# Save metadata file
metadata_bin_clf_tfrec.to_csv(r"tfrec_v3/metadata_bin_clf_tfrec.csv", index = False)