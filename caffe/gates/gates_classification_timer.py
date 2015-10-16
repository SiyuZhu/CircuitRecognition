import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle, gzip, sys, caffe
from matplotlib.backends.backend_pdf import PdfPages
from scipy import misc
import time

# -------------------------------------------------------------------
# caffe network configuration information
# -------------------------------------------------------------------
DATA_DIR = '../../data/gates'
PROJ_DIR = '.'
MODEL_FILE = PROJ_DIR + '/lenet.prototxt'
PRETRAINED = PROJ_DIR + '/lenet_iter_500.caffemodel'

# -------------------------------------------------------------------
# set up the mean file as numpy array
# https://github.com/BVLC/caffe/issues/808
# -------------------------------------------------------------------

# blob = caffe.proto.caffe_pb2.BlobProto()
# data = open( MEAN_FILE , 'rb' ).read()
# blob.ParseFromString(data)
# arr = np.array( caffe.io.blobproto_to_array(blob) )
# MEAN_NPARRAY = arr[0]

# -------------------------------------------------------------------
# set up the trained caffe network
# -------------------------------------------------------------------
caffe.set_phase_test()
caffe.set_mode_cpu()

#net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=MEAN_NPARRAY)
net = caffe.Classifier(MODEL_FILE, PRETRAINED)

# -------------------------------------------------------------------
# input data configuration
# -------------------------------------------------------------------
# images to be tested are stored in ./val_data, and their filenames and
# labels are written in ./val_data/val.txt as expected by caffe
VAL_DATA = DATA_DIR + '/test_data'
VAL_TXT = VAL_DATA + '/test.txt'
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1

# -------------------------------------------------------------------
# load test images and labels
# -------------------------------------------------------------------
# parses test image filenames and labels from VAL_TXT and loads the
# images. 
USING_GRAYSCALE = (IMG_CHANNELS == 1);
val_txt = open(VAL_TXT, "r");
val_line_tokens = val_txt.readline().split(' ');
labels_temp = [];
images_temp = [];
max_label = 0;
while len(val_line_tokens) >= 2:
  image = misc.imread(VAL_DATA + '/' + val_line_tokens[0].strip(),
                      USING_GRAYSCALE)
  images_temp.append(image.reshape(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS));
  label = int(val_line_tokens[1].strip());
  labels_temp.append(label);
  max_label = (label if label > max_label else max_label)
  val_line_tokens = val_txt.readline().split(' ');
val_txt.close();

# shuffles images before classification
labels = []
images = []
shuffle_order = range(len(labels_temp))
random.shuffle(shuffle_order)
for i in shuffle_order:
  labels.append(labels_temp[i])
  images.append(images_temp[i])

# the number of images to classify
indices = xrange(len(labels));

# -------------------------------------------------------------------
# perform classification
# -------------------------------------------------------------------
right = 0
total = 0;
headers = ["Index", "Test Label", "Prediction Result"]
predictions = [0] * (len(indices))

# runs the caffe classifier using batch inputs
start_time = time.time()
predictions = net.predict(images, False)
end_time = time.time();

# prints results
for i in indices:
  predict = predictions[i].argmax()
  print "Index", repr(i).rjust(3), ":\tPredict ", predict, "for ", labels[i]
  if predict == labels[i]:
    right += 1
  total += 1
print "Total Accuracy: ", ((float) (right))/total
print "Total Runtime: " , (end_time - start_time)
