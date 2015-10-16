#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
# argument 1 is the project directory

# parses first argument, which should specify the project
if [ -z "$1" ]; then
    echo "Please specify a project directory as an argument"
    exit
fi
DIR="$( cd "$(dirname "$0")" ; pwd -P)"
PROJ=$DIR/${1%/}
if [ ! -d "$PROJ" ]; then
    echo "Error: $PROJ directory does not exist."
    echo "Please use a valid project directory located in hetero/data"
    exit 1
fi

CAFFE_ROOT=/work/zhang/common/tools/caffe
CAFFE_TOOLS=$CAFFE_ROOT/build/tools

TOOLS=$DIR/tools
BACKEND=lmdb # lmdb/leveldb
TRAIN_DB=$PROJ/train_$BACKEND
TEST_DB=$PROJ/test_$BACKEND
USE_MEAN=false
MEAN_FILE=$PROJ/mean.binaryproto

# runs the script that retrieves the data required for the project
$PROJ/get_data.sh

rm -rf $TRAIN_DB
rm -rf $TEST_DB
rm -rf $MEAN_FILE

echo "Creating train ${BACKEND}..."

GLOG_logtostderr=1 $CAFFE_ROOT/examples/mnist/convert_mnist_data.bin $PROJ/train-images-idx3-ubyte \
  $PROJ/train-labels-idx1-ubyte $TRAIN_DB --backend=${BACKEND}

if $USE_MEAN; then
echo "Creating train mean..."

GLOG_logtostderr=1 $CAFFE_TOOLS/compute_image_mean \
    $TRAIN_DB \
    $MEAN_FILE \
    $BACKEND
fi

echo "Creating test ${Backend}..."

GLOG_logtostderr=1 $CAFFE_ROOT/examples/mnist/convert_mnist_data.bin $PROJ/t10k-images-idx3-ubyte \
  $PROJ/t10k-labels-idx1-ubyte $TEST_DB --backend=${BACKEND}

echo "Done."
