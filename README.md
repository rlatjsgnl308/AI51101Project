#  Comparison between Adam amd AdamW in simple tasks

This repository compares two optimizer, Adam and AdamW, based on pytorch implementations.

For comparison, this code conducts two tasks:

* Grid test

  * On a grid of learning rate and weight decay rate, a model is trained with both Adam and AdamW
  * learning rate = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
  * weight decay rate = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02], multiplied by 0.0001
    
* Training test

  * With a fixed learning rate, a model is trained with both Adam and AdamW under various weight decay rate setting
  * The fixed learing rate = 0.001
  * weight decay rate : same with the grid test

For these two test, this code expeiments two tasks

* Image classification

  * Model : [VGG11](https://pytorch.org/hub/pytorch_vision_vgg/) with batch normalization
  * Dataset : [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
    
* Image generation

  * Model : [VAE](https://github.com/ANLGBOY/VAE-with-PyTorch) with dense layers
  * Dataset : [Fashion MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

## Requirements

* pytorch >= 1.11.0
* torchvision >= 0.12.0
  
To install requirements, 

```setup
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Or you can download any proper [version](https://pytorch.org/get-started/previous-versions/).

## Testing

To proceed above tests, run this command:

```train
./run.sh
```

Additionally, in run.sh file, hyperparameters, including batch size, epoch, model seeds and designated learning rate and weight decay rate, can be adjusted.

## Visualizeion

Visualization folder contains files to visualize obtained results

* Visualize VGG grid test

```bash
python3 
```

* Visualize VGG training test
```bash
python3 
```

* Visualize VAE grid test
```bash
python3
```

* Visualize VAE grid test qualitatively
```bash
python3
```

* Visualize VAE training test
```bash
python3 
```

* Visualize VAE training test qualitatively
```bash
python3 
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
