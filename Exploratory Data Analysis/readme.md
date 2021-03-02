# Exploratory Data Analysis (EDA) : 

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/magnifying%20glass.jpg)

Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions. The main purpose of EDA is to help look at data before making any assumptions. It can help identify obvious errors, as well as better understand patterns within the data, detect outliers or anomalous events, find interesting relations among the variables. 

Upon loading the official metadata file, we observed : 
* Images were NLP labeled, with multiple labels separated by |. This had to be broken down into individual labels, which we performed down the line.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/head.png)
(*Metadata csv provided with the image data by NIH*.)

* There were no missing values.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/missing%20values.png)

## Helper Methods : 
**Image loading** and **resizing** functions are defined to reduce the OG 1024 X 1024 chest X-ray scans to smaller 224 X 224 or 512 X 512 dimensions. Using images of 1024 will not only lead to significant increase in the input vector size, but also can lead to memory exhaustion during training.

The following image shows randomly selected Chest X-ray scans available in our dataset -

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/samples.png)
(*Taken from ChestX-Ray NIH dataset*)

## View Positions : 

Upon closer inspection, we found that we don't have any lateral images in our dataset.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/lat_vs_front.jpg)
(*We only have images as shown on the left - the frontal/back view. The side view X-rays( figure II) are often found difficult to interpret anyways..*)

## Extracting Nodular and Healthy Images

Our end-goal is identification of nodule-mass/nodule images from healthy ones. Hence, for purpose of EDA, we will be extracting all the nodular and healthy images.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/nodule_v_healthy.png)

(*This confirms that dataset is highly skewed when it comes to nodule images. A small fraction of overall images actually have nodules*).

## Age Analysis
Upon age analysis we found that A good fraction of our patients are b/w 40-60 years of age. 

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/age.png)

(*Age analysis of patients - healthy, nodular abnormality*.)

Next, have a look separately at age of the patients having nodules.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/age_2.png)

(*Mean age of the distribution of diagnosed patients= 49.25  Standard Deviation age of the distribution of diagnosed patients= 15.34*.)

For healthy cases : 

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/age_3.png)

(*Mean age of the distribution of healthy patients= 45.48  Standard Deviation age of the distribution of healthy patients= 16.53*.)

**Inference** : The mean and standard deviation for both healthy and diagnosed patients is comparable and within each other's hair. Hence, considering this as a feature to distinguish between them might not give the best results.

## Gender Analysis

Following distribution b/w diagnosed males : diagnosed females was observed. Once again, it's comparable and not a very strong basis for identification. Both genders seem almost equally prone to the infection.

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/gender.png)

-------------------------------------------------------------------------------------

This all concludes our initial EDA. Navigate back to the main page, from where you can access the data creation/classification pipeline created post this step.

Back to main : **https://github.com/CodingWitcher/NIH_Chest_X_Ray**
