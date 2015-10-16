import numpy as np
import matplotlib.pyplot as plt
import cPickle, gzip, sys, caffe
from matplotlib.backends.backend_pdf import PdfPages

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
indices = range(100)

# -------------------------------------------------------------------
# reshape test images
# -------------------------------------------------------------------
# test_images is a 10000 x 28 x 28 x 1 ndarray, each pixel is in [0,1]
test_images = test_set[0]
test_images = test_images.reshape(test_images.shape[0], 28,28,1)
# test_labels is a 10000 ndarray
test_labels = test_set[1]

# -------------------------------------------------------------------
# perform classification
# -------------------------------------------------------------------
print "\n"
plot = True
if(plot):
  pp = PdfPages('plots.pdf')
right = 0
total = 0

for i in indices:
  # runs the classifer and prints results
  prediction = net.predict([test_images[i]])
  predict = prediction[0].argmax()
  print "Index ", repr(i).rjust(3), ":\tPredict ", predict, "for ", test_labels[i]

  if predict == test_labels[i]:
    right += 1
  total += 1

  # ignore everything below if not plotting
  if not plot:
    continue

  # extract the feature maps in each layer
  # k is the name
  # v.data is the layer data
  for k,v in net.blobs.items():
    #print k, v.data.shape
    s = v.data.shape

    # ignore the layer if it is only 1 pixel
    if (s[2:4] == (1,1)):
      continue

    plt.figure(1)
    plt.title(k)
    ploth = s[1] / 5 if s[1] >= 5 else 1
    plotw = s[1] / ploth

    for h in range(ploth):
      for w in range(plotw):
        print "plotting", h, w, h*plotw+w+1
        plt.subplot(ploth, plotw, h*plotw+w+1)
        plt.imshow(v.data[0,h*plotw+w], cmap='Greys')
    plt.savefig(pp, format='pdf')
    plt.close()

if(plot):
  pp.close()

# print results
print "Accuracy = ", float(right)/total*100, "%"
