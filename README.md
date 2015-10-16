Hetero
======
Heterogenous Computing Project using Halide to optimize the execution of Convolutional Neural Networks (CNNs).
See README files in each suybdirectory for more instructions. If this does not help please ask someone on the project to update the README.

Environment Setup
=================
Source the file "hc_env_setup.sh" in the top level directory. This should enable Halide, OpenCV, and Caffe.

Contents
=======
  * halide - halide programs
  * caffe - cnn architecture files, training and testing scripts, trained models
  * data - data, extracted parameters, scripts to build datastores, precprocess images, and extract images
  * opencv - opencv programs (for runtime comparison to Halide)
  * utils - utility code

Results
=======
GATES CLASSIFIER
Goal: Create a classifier for digital logic gates
Implementation Details: Uses the lenet architecutre, and the same hyperparameters as those used in the mnist project. Uses a mean file. Images were first cropped, and various programs were written to conduct preprocessing proucedures include initial resizing (the print_image_stats tool can be used to get median sizes for all the images), centering on a background, resizing, and augmentation(8 different versions using rotations and reflections). 

Accuracy Results: 85% accuracy after 1000 training iterations

MNIST CLASSIFIER
Goal: Test the image classification capabilities of Halide

Implementation Details: Uses caffe's cpp benchmark tool ("caffe time" command to determine the time required for caffe's forward phase. Uses batch-lenet to determine the time required for the forward phase using halide. Performace on the zhang-01 server and my local machine show that haldie is superior to caffe (TODO check the last few layers). A gpu implementation of the halide classifer was also produced.

Additional Impelementations: Python scripts for caffe classification and for viewing activation maps. Also created lenet_cnn, another halide-based classifier (batch-lenet is superior).

Performance Results
caffe on zhang-01: 6827, 7147, 6821
batch-lenet on zhnag-01: 3833, 3832, 3830
(local is 2.9 GHz Dual Core Intel Core i7 w/ hyperthreading)
caffe on local: 3683, 3697, 3586
batch-lenet on local: 1922, 1974, 1991

LATTE
Goal: Use lenet_train_test.prototxt and .caffemodel to do image classification with halide. 

Purpose: We would like to have an easy flow from caffe to halide so that OpenCL pipelines can be generated. This project would make that as easy as running one command. The best alternative is using espresso, but for each new pipline, that would require 120 lines of cpp using the espresso API and 100 lines of harness code using the espresso API.

Features:
+ Do not have to independently extract caffe parameters
+ Compatibility with all image databases and formats accepted by caffe
+ Does not require ANY extra code

Incomplete: 
Debug layers:
      softmaxloss layer, inner product and convolution layer have bugs. The correct function definition is likely not used in softmaxloss layer. In inner product and convolution layer, the weights are not properly extracted and used in the weights/bias function.
Implement remaining layers
Find a scheduling solution (prewritten schedules, autotuner intergration)