# cnn_cats_vs_dogs
a basic image classifier using convolutional neural network to differentiate between cats and dogs  

## model structure
- convolutional layer with same padding (64 kernels)
- max pool layer with kernel size of 2*2 and strides of 2
- convolutional layer with same padding (64 kernels)
- max pool layer with kernel size of 2*2 and strides of 2
- flatten layer
- fully connected layer (64 nodes)
- fully connected layer (32 nodes)
- output layer

## training info
- dataset link: [kaggle_dogs_vs_cats](https://www.kaggle.com/chetankv/dogs-cats-images#dog%20vs%20cat.zip)
- training dataset: 9000 images (4500 each)
- testing dataset: 1000 images(500 each)
- epochs: 4
- validation accuracy: .8344 (83%)
- validation loss: .3912

## how to run
- clone this repo
- save image you want to calssify in the cloned folder (only jpg image)
- run predict.py and enter the full name of image
- accuracy of the model is 83% thus prediction may be wrong so try with a different image in that case
