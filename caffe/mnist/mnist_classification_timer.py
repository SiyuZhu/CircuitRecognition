import numpy as np
import cPickle, gzip, sys, caffe
import time

# -------------------------------------------------------------------
# set up the trained caffe network
# -------------------------------------------------------------------
CAFFE_ROOT = '/work/zhang/common/tools/caffe/'
DATA_DIR = '../../data/mnist'
PROJ_DIR = '.'
MODEL_FILE = PROJ_DIR + '/lenet.prototxt'
PRETRAINED = PROJ_DIR + '/lenet_iter_10000.caffemodel'

caffe.set_phase_test()
caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED)

# -------------------------------------------------------------------
# read the input mnist data
# -------------------------------------------------------------------
f = gzip.open(DATA_DIR + '/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# these are the images we want to classify
count = 9984

# -------------------------------------------------------------------
# reshape test images
# -------------------------------------------------------------------
# test_images is a 10000 x 28 x 28 x 1 ndarray, each pixel is in [0,1]
test_images = test_set[0]
test_images = test_images.reshape(test_images.shape[0], 28,28,1)
test_images = test_images[0:count,:,:,:]
# test_labels is a 10000 ndarray
test_labels = test_set[1]

# -------------------------------------------------------------------
# perform classification
# -------------------------------------------------------------------
print "\n"

right = 0
total = 0
predictions = [0] * (count)

# classifies the images in batches
start_time = time.time()
predictions = net.predict(test_images, False)
end_time = time.time();

# prints the result
for i in range(count):
  predict = predictions[i].argmax()
#  print "Index ", repr(i).rjust(3), ":\tPredict ", predict, "for ", test_labels[i]
  if predict == test_labels[i]:
    right += 1
  total += 1
print "Total Accuracy: ", ((float) (right))/total
print "Total Runtime: ", (end_time - start_time)
