# Classification Architecture - CheXNet
![](https://github.com/CodingWitcher/NIH_Chest_X_Ray/blob/main/images_for_readme/xray%20heatmap.webp)

CheXNet(by Stanford) is a **121-layer dense convolutional neural network** trained on ChestX-ray14, currently the largest publicly available chest X-ray dataset, containing over 100,000 frontal-view X-ray images with 14 diseases. 

DenseNets improve flow of information and gradients through the network, making the optimization of very deep networks tractable. The weights of the network are initialized with weights from a model pretrained on ImageNet (Deng et al.,2009). The network is trained end-to-end using Adam optimization, with standard parameter values. Link to paper on Densenet is available in the **refrenced_papers** section of this repository.

