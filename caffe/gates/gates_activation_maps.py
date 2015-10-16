import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle, gzip, sys, caffe
from matplotlib.backends.backend_pdf import PdfPages
from scipy import misc

# -------------------------------------------------------------------
# caffe network configuration information
# -------------------------------------------------------------------
CAFFE_ROOT = '/work/zhang/common/tools/caffe/'
DATA_DIR = '../../data/gates'
PROJ_DIR = '.'
MODEL_FILE = PROJ_DIR + '/lenet.prototxt'
PRETRAINED = PROJ_DIR + '/lenet_iter_500.caffemodel'
MEAN_FILE = DATA_DIR + '/mean.binaryproto'

# -------------------------------------------------------------------
# set up the mean file as numpy array
# https://github.com/BVLC/caffe/issues/808
# -------------------------------------------------------------------

#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open( MEAN_FILE , 'rb' ).read()
#blob.ParseFromString(data)
#arr = np.array( caffe.io.blobproto_to_array(blob) )
#MEAN_NPARRAY = arr[0]

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
VAL_TXT = VAL_DATA + '/' + 'test.txt'
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
#t -------------------------------------------------------------------
print "\n"
plot = True #False
if plot:
  pp1 = PdfPages('plots_correct.pdf')
  pp2 = PdfPages('plots_incorrect.pdf');

# Keeps track of total accuracy
right = 0
total = 0

# Keeps track of accuracy for each class
class_right = [0] * (max_label+1);
class_total = [0] * (max_label+1);

for i in indices:
  prediction = net.predict([images[i]], False)
  predict = prediction[0].argmax()
  print "Index ", repr(i).rjust(3), ":\tPredict ", predict, "for ", labels[i]

  if predict == labels[i]:
    right += 1
    class_right[labels[i]] += 1
  total += 1
  class_total[labels[i]] += 1

  # ignore the rest of this for loop if not plotting
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
    if predict == labels[i]: plt.savefig(pp1, format='pdf')
    else: plt.savefig(pp2, format='pdf')
    plt.close()

if plot:
  pp1.close()
  pp2.close()

# -------------------------------------------------------------------
# print results to console
# -------------------------------------------------------------------
print "Total Accuracy = ", float(float(right)/total*100), "%"
for i in xrange(len(class_total)):
  print i, " Accuracy = " , float(class_right[i])/class_total[i]*100, "%"

# -------------------------------------------------------------------
# print results to results.txt
# -------------------------------------------------------------------
# f = open('results.txt' , 'w');
# f.write("Total Accuracy = " + str(float(right)/total*100) + "%\n");
# for i in xrange(len(class_total)):
#   f.write(str(i) + " Accuracy = " + str(float(class_right[i])/class_total[i]*100) + "%\n");
# f.close();
