# densenet_1d

This repository contains a Keras implementation of the DenseNet paper (Huang et al, "Densely Connected Convolutional Networks", CVPR 2017). This implementation will focus on use-cases where the inputs are 1D sequences. 

# Setup
To install densenet, simply clone this repository, and run
```
python setup.py install
```

# Usage

The classifiers directory contains classifiers implemented as subclasses of keras.models.Model classes. This means that once a densenet.classifier is instantiated, it contains all of the usual methods of keras.models.Model, such as fit, predict, evaluate, summary, etc.

Here is an instantiation of the model that matches the original Huang et al. paper, except using a one-dimensional input rather than a two-dimensional input:

```python

from densenet.classifiers.one_d import DenseNet121
model = DenseNet121((224, 10), 1000, 5, 3, 4, 2, 2, .5, 7, 2, 10, 3, 2)
print(model.summary())
```
Upon running those lines, you should see an extensive summary indicating the layers in the model.

# References

- [Original paper](https://arxiv.org/abs/1608.06993): Huang et al. "Densely Connected Convolutional Networks", CVPR 2017. 
- Another great implementation of DenseNets in Keras, although not one that uses 1D sequences. https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
- An implementation of Resnet in Keras whose directory structure was loosely used for this implementation: https://github.com/broadinstitute/keras-resnet


