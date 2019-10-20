# Face Detector using CNN

A simple face detector (MD) trained from convolutional neural network (CNN).
Accuracy is as high as 98%

## Prerequisites
* [Python3](https://www.python.org/) (tested on Python 3.7.4)
* [TensorFlow 2.0](https://www.tensorflow.org/) (tested on 2.0.0)
* [NumPy](https://numpy.org/) (tested on 1.17.3)
* [SciPy](https://www.scipy.org/) (tested on 1.3.1)
* [Matplotlib](https://matplotlib.org/) (tested on 3.1.1)
* Datasets (see DATA.SOURCE.md)

## Usage
Run `./run.sh` to learn a face detector

## Details
* [UTKFace Dataset](https://susanqq.github.io/UTKFace/) was used to train face
images
* Random scenery pictures were used to train non-face images
* An independent [LHI-Animal-Faces](http://www.stat.ucla.edu/~zzsi/HiT/exp5.html) dataset was used as part of the test set
* In the test set, non-face images are more than images with faces to simulate
a actual picture where most of the windows do no contain a face

Note that the sample/test division is random, so the accuracy could fluctuate
