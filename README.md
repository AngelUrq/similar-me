
# SimilarMe

This is an implementation of the FaceNet paper in Tensorflow 2. The paper can be found [here](https://arxiv.org/abs/1503.03832).

![](https://i.postimg.cc/JhksT45B/download.jpg)

## Requirements

* Tensorflow
* Numpy

## Usage

To train the model, you will need a dataset of images of faces. I have used the VGGFace2 dataset. Once you have downloaded the dataset, place it in data/train and data/test, create a folder for each identity.

You can then train the model by running `python train.py`.
