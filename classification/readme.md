# Classification Architecture - CheXNet
![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/xray%20heatmap.webp)

CheXNet(by Stanford) is a **121-layer dense convolutional neural network** trained on ChestX-ray14, currently the largest publicly available chest X-ray dataset, containing over 100,000 frontal-view X-ray images with 14 diseases. 

DenseNets improve flow of information and gradients through the network, making the optimization of very deep networks tractable. The weights of the network are initialized with weights from a model pretrained on ImageNet (Deng et al.,2009). The network is trained end-to-end using Adam optimization, with standard parameter values. Link to paper on Densenet is available in the **refrenced_papers** section of this repository.

The model is trained using **Google Cloud Compute Engine, harnessing power of 30 GB RAM, of an octacore CPU, and a Tesla V100 GPU**. With alterations in batch size, this could be trained using Google Colab/Kaggle kernels too.

## Results ~ Part One Of Two : 

For nodule_mass classification, we obtained a **validation AUC ROC score of nearly 0.82**, after assigning class weights during training. In terms of ROC AUC model performed amazingly well (training ROC AUC of nearly 0.94). As far as confusion matrix is concerned, we saw low count of false positives, but a slightly higer count of false negatives. 

With little bit of tinkering in the architecture, and probably increasing image size to 600X600 for training(*rather than 224 X 224*), we believe that those False negatives can go down. 

**Why?** 

It can be argued that nodules are generally tiny specks in an entire chest X-ray scan. Resizing to 224, might cause them to become so miniscule that their detection may become excruciatingly difficult by the neural net. Again, going with 1k X 1K images is not feasible too, as the size of input vector will increase dramatically, and that could lead to RAM exhaustion. 600 X 600 is a good option, with obviously low batch sizes (*can go for higher if you have colossal CPU memory*). 

Following images show the detected nodules in a typical chest X-ray (*note that detection is not covered in this repository, as we used an external dataset which could not be made public due to Non-Disclosure Agreement(NDA). Neverthess, this should convey the message that nodules are super small in size, due to which they may essentially vanish during resizing to 224 or 128 dimensions*).

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/nodule_1)
(*Detected nodule. Compare the size wrt the full image*).

![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/nodule_2)
(*Detected nodule*).

**Also, it is interesting to note how alike TP(True positive) nodules and FP(False Positive) nodules appear. In the first image, we have a TP and in second FP. Even from naked eyes, it's exruciatingly difficult to tell them apart as real/fake. Interestingly, most research papers leave it after ROC AUC, whereas we feel the true picture is never completed without the Confusion matrix**.

## Results ~ Part Two Of Two
