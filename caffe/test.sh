#!/usr/bin/env sh

if [ -z "$1" ]; then
    echo "Please specify a project directory as an argument"
    exit 1
fi

DIR="$( cd "$(dirname "$0")" ; pwd -P)"
PROJ=$DIR/${1%/}

if [ ! -d "$PROJ" ]; then
    echo "Error: $PROJ directory does not exist."
    echo "Please use a valid project directory located in hetero/caffe"
    exit 1
fi

cd $PROJ

CAFFE_ROOT=/work/zhang/common/tools/caffe
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe test --model=lenet_train_test.prototxt --weights=./lenet_iter_10000.caffemodel --iterations=156
#$TOOLS/caffe time --model=lenet_train_test.prototxt --weights=./lenet_iter_10000.caffemodel --iterations=156
# $TOOLS/caffe time --weights=./lenet_iter_10000.caffemodel --model=lenet.prototxt --iterations=156