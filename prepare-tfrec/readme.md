# Tensorflow Records Preparation : 

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/thumbnail_tf.png)

**Note that this should be considered more of a tangent, rather than a mandatory approach. For the final cut, we didn't use TFRecords. It's upto the user, in which format he/she wants to prepare the data, that is either as tensor slices, or batches from Image Data Generator of Keras, or TFRecord files**.

I sincerely recommend to go for either tensor slices or TFRecord files. Image Data Generator is very simple to use, however it doesn't utilizes the hardware accelerator(GPU/TPU) efficiently.

## TFRecords - A Quick Insight
A Tensorflow Record (TFRecord for short) is Tensorflow's binary storage format for your data.

A TFRecord file contains an array of **Examples**. **Example is a data structure for representing a record**, like an observation in a training or test dataset. **A record is represented as a set of features, each of which has a name and can be an array of bytes, floats, or 64-bit integers**.

To summarize :

* A TFRecord is a collection of Examples.
* An Example contains Features.
* Features is a mapping from the feature names stored as strings to Features.

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
