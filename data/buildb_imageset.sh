#!/usr/bin/env sh
# This script can be used to convert the gates data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND

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

CAFFE_ROOT=/usr/local/caffe
CAFFE_TOOLS=$CAFFE_ROOT/build/tools

BACKEND="lmdb" # lmdb/leveldb
TRAIN_DATA=$PROJ/train_data/
TRAIN_TXT=$TRAIN_DATA/train.txt
TRAIN_DB=$PROJ/train_$BACKEND
TEST_DATA=$PROJ/test_data/
TEST_TXT=$TEST_DATA/test.txt
TEST_DB=$PROJ/test_$BACKEND
USE_MEAN=true
MEAN_FILE=$PROJ/mean.binaryproto

# runs the script that retrieves the data required for the project
$PROJ/get_data.sh

# Set RESIZE=true to resize the images to 28x28. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=28
  RESIZE_WIDTH=28
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

# check that the required directories exist
if [ ! -d "$TRAIN_DATA" ]; then
  echo "Error: TRAIN_DATA is not a path to a directory: $TRAIN_DATA"
  echo "Set the TRAIN_DATA variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi
if [ ! -d "$TEST_DATA" ]; then
  echo "Error: TEST_DATA is not a path to a directory: $TEST_DATA"
  echo "Set the TEST_DATA variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

rm -rf $TRAIN_DB
rm -rf $TEST_DB
rm -rf $MEAN_FILE

echo "Creating train $BACKEND..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --backend=$BACKEND \
    --gray=true \
    --shuffle \
    $TRAIN_DATA \
    $TRAIN_TXT \
    $TRAIN_DB

if $USE_MEAN; then
echo "Creating train mean..."

GLOG_logtostderr=1 $CAFFE_TOOLS/compute_image_mean \
    --backend=$BACKEND\
    $TRAIN_DB \
    $MEAN_FILE
fi

echo "Creating test $BACKEND..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --backend=$BACKEND \
    --gray=true \
    --shuffle \
    $TEST_DATA \
    $TEST_TXT \
    $TEST_DB

echo "Done."
