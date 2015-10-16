#!/usr/bin/env sh

# parses user input for the project name
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

# trains the given caffe network
cd $PROJ
CAFFE_ROOT=/work/zhang/common/tools/caffe
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe train --solver=./solver.prototxt

# Use below to resume from a solver state
#$TOOLS/caffe train --solver=./solver.prototxt --snapshot=./lenet_iter_500.solverstate

# Use below for fine tuning
# $TOOLS/caffe train --solver=./finetune_solver.prototxt --weights=./lenet_iter_10000.caffemodel