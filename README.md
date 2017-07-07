## Dependency
* Python3.6(numpy, scipy, pickle, h5py, Pillow),
* Keras2.02,
* Tensorflow v1.1 backend, (Not Theano backend)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
Super-Resolution GAN was firstly raised in the [paper](https://arxiv.org/pdf/1609.04802). The main idea is using a discriminator instead of the "mse" loss function. The scaled image is fuzzy when scaling image using the neural network trained with "mse" loss function. Thanks to the idea of GAN, the scaling effects are much better. 

We use cifar10 dataset (only 32x32 pixels) because training is easier with small dimensional datasets. The generator tries to generate enlarged images as the input images are resized to 16x16 by BICUBIC method. The discriminator tries to distinguish generated images from raw images. The discriminator is trained by a larger learning rate, hence the discriminator is always better trained than the generator. After training for 80 epochs, the generator can generate scaled image with a high quality. Since our dataset is small, the training time is short and 80 epochs is enough. 

Some comparisons between the generated images and the NEAREST-resized images are presented below 
<p align="center">
  <img src="https://github.com/liangstein/SRGAN-Keras/blob/master/comparison.png" width="350"/>
</p>

Training GAN is sensitive to network structures and hyperparameters, there may be a better hyperparameter than that in this implementation. 

## Authors
liangstein (lxxhlb@gmail.com, lxxhlb@mail.ustc.edu.cn) 

