# Tensorflow Records Preparation : 

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/thumbnail_tf.png)

**Note that this should be considered more of a tangent, rather than a mandatory approach. For the final cut, we didn't use TFRecords. It's upto the user, in which format he/she wants to prepare the data, that is either as tensor slices, or batches from Image Data Generator of Keras, or TFRecord files**.

I sincerely recommend to go for either tensor slices or TFRecord files. Image Data Generator is very simple to use, however it doesn't utilizes the hardware accelerator(GPU/TPU) efficiently.

## TFRecords - A Quick Insight
A Tensorflow Record (TFRecord for short) is Tensorflow's binary storage format for your data. These are essentially a tensor of records, which are represented using Example data structure. **This data structure is nothing more than a compilation of faetures which map the OG string features to TF compatible feature format**.

As stated previously, a TFRecord file contains an array of **Examples**. **Example is a data structure for representing a record**, like an observation in a training or test dataset. **A record is represented as a set of features, each of which has a name and can be an array of bytes, floats, or 64-bit integers**.

To summarize :

* A TFRecord is a collection of Examples.
* An Example contains Features.
* Features is a mapping from the feature names stored as strings to Features.

This is what we are gonna do now. We define a set of features and encapsulate them in an Example data structure.

The serialization sceme inside a TFRecord file is actually based on **Protocol Buffers**.

## Protocol Buffers : 

Google’s **Protocol buffer is a serialization scheme for structured data**. In other words, protocol buffers are used for serializing structured data into a byte array, so that they can be sent over the network or stored as a file. *In this sense, it is similar to JSON, XML*.

Protocol buffers can offer a lot faster processing speed compared to text-based formats like JSON or XML.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/protocolbuffers.jpg)

Using protocol buffers has many advantages over plain text serializations like JSON and XML:

* Very dense data which result in very small output and therefore less network overhead
* Declared schema makes parsing from most languages very straightforward with less boilerplate parsing code
* Very fast processing
* Binary encoded and hard to decode without knowledge of the schema.

# Stratified Sampling :
As we saw previously during EDA, our dataset turned out to be massively skewed. So, in order to make the distribution fair during training, validation and testing, a stratified sampling approach was implemented and is encoded in the Python3 file available in this subdirectory. Stratified sampling refers to a type of sampling method . With stratified sampling, we divides the population into separate groups, called strata. Then, a simple random sample is drawn from each group.

# Metadata For Logging Info : 
An additional metadata file was also created, and was later exported as .csv
The format of the file was : 

Columns : [image_name, label, subset] 
* subset will store whether the image is for training/val/testing.

This file will store the name and label of the image, along with whether the image belongs to training, validation or testing category. This logging of information can come in handy as one might augment the images during preparation of tfrec files, in order to expand the dataset.

# Augmentations
In the script, I did cut down the images for selection. Since healthy count was way too more than nodular, so the way the script is written : 
* It will pick every second healthy image, and encode it in .tfrec file.
* As for nodular cases, each image will be augmented once. So, original as well as augmented version (rotation/horizontal flip) will be appended in the TFRecords. Hence, we are essentially doubling the nodular images. One can go for more augmentation count( *that is instead of once, perform augmentation multiple times*). However, too much can lead to overfitting too. 

For validation and testing TFRecords, there wasn't any need of augmentations as they will simply be used for evaluating and testing the trained model. 

*This script is made more as a reference. I strongly encourage to tinker with it, by adding more augmentations, or altering sampling techniques. There is potential for improvement of this script*. 

-------------------------------------------------------------------------------------------------------
This ends this section of the project. 

**Back to homepage (main)** : **https://github.com/CodingWitcher/NIH_Chest_X_Ray**

