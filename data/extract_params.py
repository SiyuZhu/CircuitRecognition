#!/work/zhang/common/usr/bin/python
#import numpy as np
import cPickle, gzip, sys, caffe
import sys, getopt

# -------------------------------------------------------------------
# configuration
# -------------------------------------------------------------------
PROJ = sys.argv[1].strip('/')
CAFFE_PROJ_DIR = '../caffe/' + PROJ
DATA_DIR = './' + PROJ
MODEL_FILE = CAFFE_PROJ_DIR + '/lenet.prototxt'
# MEAN_FILE = PROJ + '/mean.binaryproto'
PRETRAINED = CAFFE_PROJ_DIR + '/lenet_iter_500.caffemodel'
DST_PREFIX = DATA_DIR + '/layer_params'
DST_POSTFIX = '.dat'

# ------------------------------------------------------------------- 
# setup the mean file as numpy array
# https://github.com/BVLC/caffe/issues/808
# ------------------------------------------------------------------- 
# blob = caffe.proto.caffe_pb2.BlobProto()
# data = open(MEAN_FILE,'rb').read()
# blob.ParseFromString(data) 
# arr = np.array(caffe.io.blobproto_to_array(blob))
# MEAN_NPARRAY = arr[0]

# ------------------------------------------------------------------- 
# setup caffe net
# ------------------------------------------------------------------- 
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
#net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=MEAN_NPARRAY)

# -------------------------------------------------------------------
# collect parameter data
# -------------------------------------------------------------------
conv1_weight = net.params['conv1'][0].data
conv1_bias = net.params['conv1'][1].data
conv2_weight = net.params['conv2'][0].data
conv2_bias = net.params['conv2'][1].data
ip1_weight = net.params['ip1'][0].data
ip1_bias = net.params['ip1'][1].data
ip2_weight = net.params['ip2'][0].data
ip2_bias = net.params['ip2'][1].data

# -------------------------------------------------------------------
# write parameters to files
# -------------------------------------------------------------------
layer_count = 0;
# the first convolution layer
f = open(DST_PREFIX + str(layer_count) + DST_POSTFIX, "w")
layer_count += 1
f.write("weights\n")
f.write("20, 1, 5, 5, \n")
for i in range(20):
    for j in range(1):
        for k in range(5):
            for l in range(5):
                f.write(str(conv1_weight[i][j][k][l]) + ", ")
        f.write("\n")
f.write("biases\n")
f.write("20,\n")
for i in range(20):
    f.write(str(conv1_bias[0][0][0][i]) + ", ")
f.close()

# the second convolution layer
f = open(DST_PREFIX + str(layer_count) + DST_POSTFIX, "w")
layer_count += 1
f.write("weights\n")
f.write("50, 20, 5, 5, \n")
for i in range(50):
    for j in range(20):
        for k in range(5):
            for l in range(5):
                f.write(str(conv2_weight[i][j][k][l]) + ", ")
        f.write("\n")
f.write("biases\n")
f.write("50,\n")
for i in range(50):
    f.write(str(conv2_bias[0][0][0][i]) + ", ")
f.close()

# the first inner product layer
f = open(DST_PREFIX + str(layer_count) + DST_POSTFIX, "w")
layer_count += 1
f.write("weights\n")
f.write("800, 500, \n")
for i in range(1):
    for j in range(1):
        for k in range(800):
            for l in range(500):
                f.write(str(ip1_weight[i][j][l][k]) + ", ")
            f.write("\n")
f.write("biases\n")
f.write("500,\n")
for i in range(500):
    f.write(str(ip1_bias[0][0][0][i]) + ", ")
f.close()

# the second inner product layer
f = open(DST_PREFIX + str(layer_count) + DST_POSTFIX, "w")
layer_count += 1
f.write("weights\n")
f.write("500, 10, \n")
for i in range(1):
    for j in range(1):
        for k in range(500):
            for l in range(10):
                f.write(str(ip2_weight[i][j][l][k]) + ", ")
            f.write("\n")
f.write("biases\n")
f.write("10,\n")
for i in range(10):
    f.write(str(ip2_bias[0][0][0][i]) + ", ")
f.close()
