# AI51101Project
Compare Adam and AdamW

AdamW : Decoupled Weight Decay Regularization, Ilya Loshchilov & Frank Hutter, ICRL 2019

: in the simple envrionment, verify effect of AdamW.
: Different from the paper, AdamW is compared in an unsupervised learning.

Model + Dataset
1. VGG + CIFAR 10
	https://github.com/chengyangfu/pytorch-vgg-cifar10
	https://pytorch.org/hub/pytorch_vision_vgg/ [model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)]

	-> For a grid test : Epoch 100, batch size 128 -> Final Test Accuracy
	-> For training / test loss : Epoch 200, batch size 128 -> Traning Loss + Test Accuracy

2. VAE with dense layers + Fashion MNIST
	https://github.com/ANLGBOY/VAE-with-PyTorch/blob/master/main.py
	
	-> For a grid test : Epoch 20, batch size 128
	-> For training / test loss : Epoch 25, batch size 128

Model seed : 7, 77, 777
	
Grid test
Learning rate + Weight deacy
[0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] * [10, 5, 1, 0.5, 0.1, 0.05]*0.001

Training / Test loss
[0.0001] * [10, 5, 1, 0.5, 0.1, 0.05]*0.001

-Visualization
1. Accuracy grid map for learning rate and weight decay
2. training process for each weight deacy
3. For VAE only, sample grid map for learnign rate and weight decay with same latent
4. For VAE only, sample change for each weight decay
