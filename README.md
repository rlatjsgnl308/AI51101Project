#  Comparison between Adam amd AdamW in simple tasks

This repository compares two optimizers, Adam and AdamW, based on pytorch implementations.

For comparison, this code conducts two tests:

* Grid test

  * On a grid of learning rate and weight decay rate, a model is trained with both Adam and AdamW
  * learning rate = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
  * weight decay rate = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02], multiplied by 0.001
    
* Training test

  * With a fixed learning rate, a model is trained with both Adam and AdamW under various weight decay rate settings
  * The fixed learing rate = 0.0001
  * weight decay rate : same with the grid test

For these two tests, this code expeiments two tasks. 

* Image classification

  * Model : [VGG11](https://pytorch.org/hub/pytorch_vision_vgg/) with batch normalization
  * Dataset : [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
    
* Image generation

  * Model : [VAE](https://github.com/ANLGBOY/VAE-with-PyTorch) with dense layers
  * Dataset : [Fashion MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

Each hyperlink gives where the implementation is from.

## Requirements

* pytorch >= 1.11.0
* torchvision >= 0.12.0
  
To install requirements, 

```setup
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
,or you can download any proper [version](https://pytorch.org/get-started/previous-versions/).

## Test

To proceed above tests, run this command:

```train
./run.sh
```

The bash file runs all tests sequentially. However, because each task is totally seperable with independent loop and functions, indicated by comments, one can extract some desirable tests by simply making unwanted test loops comments.

The training process is stored in each txt file whose name format is '{model} _ {test_name} _ {optimizer} _ {learning_rate} _ {weight_decay_rate} _ {epoch} _ {seed}.txt'. 

Each trained weight is stored at the last train epoch for each learning rate and weight decay rate combination. Its name format is same with txt file, but the extension is '.pth'. (For VAE, _{epoch} is added between {seed} and .pth.) 

There should be folders to contains each txt and pth file for each task (,or model) folder. Each txt file will be saved in '{model}/Stats', and pth file in '{model}/Weight'.

For the image classification task, each txt file consists of train loss, train accuracy, test loss, and test accuracy.

For the image generation, each txt file consists of total train loss, reconstruction train loss, KL divergence train loss, and corresponding test statistics. The total loss means reconstruction loss + KL loss.

Additionally, in run.sh file, hyperparameters, including batch size, epoch, model seeds and the designated learning rate and weight decay rate set, can be adjusted.

## Visualization

Visualization folder contains files to visualize the obtained statistics. 

If there are multiple trials with various model seed, this code averages all the results. 

Because qualitative results are also important for image generation tasks, there are also codes to draw sampled images with various learning rate and weight decay rate. 

It is highly recommended to run these commands in the Visualization folder.

* Visualize VGG test (First : Grid, Second : Training)

```bash
python3 VGG_grid_stats.py
```
```bash
python3 VGG_train_stats.py
```

* Visualize VAE test (First : Grid, Second : Training)
```bash
python3 VAE_grid_stats.py
```
```bash
python3 VAE_train_stats.py
```

* Visualize VAE test qualitatively (First : Grid, Second : Training)
```bash
python3 VAE_grid_image.py
```
```bash
python3 VAE_train_image.py
```

## Results

The result figures are in the Visualization folder.

The train statistics recorded and pretrained weights are in [Onedrive](https://1drv.ms/f/s!AkRfaKwbeLASxBcdFWQx5SdNPf7m?e=6EVFHP).

To run visualization code with these zip files, each uncompressed file should be located in the rigith folder.

VAE_stats.zip -> VAE/Stats

VAE_weights.zip -> VAE/Weight

VGG_stats.zip -> VGG/Stats

(There are no pretrained weights of VGG, because its zip file has too large size and these weight are not required for visualization.)

## Conclusion

After the training enters in overfit status, the generalization of AdamW get degraded faster than Adam.

The overfit situation doesn't change the parameter dependency which Adam shows.

If the training itself includes regularization (not L2 regularization) such as VAE, Adam becomes less sensitive to the weight decay rate change.

Detailed explanation is on report pdf.