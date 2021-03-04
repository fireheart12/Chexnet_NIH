# NIH_Chest_X_Ray

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/1.jpg)

The chest X-ray is one of the most commonly accessible radiological examinations for screening and diagnosis of many lung diseases. This project revolves around application of deep learning inspired computer vision for identifying the presence of nodular-mass abnormality, from the chest X-ray scans of the patient.

# Dataset Used : 
The dataset used for this project comes from National Center for Biotechnology Information, National Library of Medicine, and National Institutes of Health, Bethesda. It's christened as ChestX-ray8 dataset. During the entire project, we refer to it as NIH dataset. The providers compiled the dataset of scans from more than 30,000 patients, including many with advanced lung disease. For creating labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate.

The dataset is freely available at : **https://www.kaggle.com/nih-chest-xrays/data**, and the paper associated with this dataset can be found at : **https://arxiv.org/pdf/1705.02315.pdf**. 

# Libraries Used : 
* **os** : TO interact with the underlying Operating System.
* **glob** : UNIX like module for searching patterns.
* **shutil** : This module helps in automating process of copying and removal of files and directories. 
* **Numpy** : Numerical Python3 computations.
* **Pandas** : For handling csv files, in the dataset.
* **Matplotlib and Seaborn** : Plotting libraries in Python3.
* **Scikit-learn** : Contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering, and many more.
* **tqdm** : For visualizing progress.
* **Tensorflow 2x** : Deep learning framework by Google, for designing machine learning pipelines (*from training to production*).

# Exploratory Data Analysis : 
A separate section of this repository contains the work done in EDA. Please navigate to the link down below to reach the EDA sub-section.

EDA : **https://github.com/CodingWitcher/NIH_Chest_X_Ray/tree/main/Exploratory%20Data%20Analysis**

# Tensorflow Records Preparation [OPTIONAL] : 
A separate section of this repository contains the script and necessary info on TFRecord creation. Please navigate to the link down below to reach it.

Create TFRec files : **https://github.com/CodingWitcher/NIH_Chest_X_Ray/tree/main/prepare-tfrec**

# Classification Pipeline - Using CheXNet Architecture
This is available at the link down below, along with it's own readme and code file/s.

Classification model : **https://github.com/CodingWitcher/NIH_Chest_X_Ray/tree/main/classification**

-----------------------------------------------------------------------------------------------------------------------------------------------

**Thank you for reading this far. For any query, feel free to ping at : bauraiaditya7@gmail.com**

-----------------------------------------------------------------------------------------------------------------------------------------------



