#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips if the unzipped version does not exist.
# An argument of "force" will force the script to download and unzip all mnist images

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if ( !(test -d "$DIR/train_data" ) || !(test -f "$DIR/train_data/train.txt") || \
	!(test -d "$DIR/test_data") || !(test -f "$DIR/test_data/test.txt") || \
	    [ "$1" == "force" ] ); then
    rm -rf $DIR/train_test_data
    echo "Uncompressing..."
    tar -xzvf $DIR/train_test_data.tar.gz
    echo "Preprocessing..."
    $DIR/preprocess_images.sh
    rm -rf $DIR/train_test_data
fi