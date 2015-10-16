#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips if the unzipped version does not exist.
# An argument of "force" will force the script to download and unzip all mnist images

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if ( !(test -f "$DIR/train-images-idx3-ubyte" )  ||  [ "$1" == "force" ] ); then
    if ( !(test -f "$DIR/train-images-idx3-ubyte.gz") || [ "$1" == "force" ] ); then
	echo "Downloading..."
	wget -P $DIR --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz;
    fi
    echo "Unzipping..."
    gunzip $DIR/train-images-idx3-ubyte.gz
fi
if ( !(test -f "$DIR/train-labels-idx1-ubyte" ) ||  [ "$1" == "force" ] ); then
    if ( !(test -f "$DIR/train-labels-idx1-ubyte.gz" ) || [ "$1" == "force" ] ); then
	echo "Downloading..."
        wget -P $DIR --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz;
    fi
    echo "Unzipping..."
    gunzip $DIR/train-labels-idx1-ubyte.gz
fi
if ( !(test -f "$DIR/t10k-images-idx3-ubyte" ) || [ "$1" == "force" ] ); then
    if ( !(test -f "$DIR/t10k-images-idx3-ubyte" ) || [ "$1" == "force" ] ); then
	if ( !(test -f "$DIR/t10k-images-idx3-ubyte.gz" ) || [ "$1" == "force" ] ); then
	    echo "Downloading..."
            wget -P $DIR --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz;
	fi
	echo "Unzipping..."
	gunzip $DIR/t10k-images-idx3-ubyte.gz
    fi
fi
if ( !(test -f "$DIR/t10k-labels-idx1-ubyte" ) || [ "$1" == "force" ] ); then
    if ( !(test -f "$DIR/t10k-labels-idx1-ubyte") || [ "$1" == "force" ] ); then
	if ( !(test -f "$DIR/t10k-labels-idx1-ubyte.gz") || [ "$1" == "force" ] ); then
	    echo "Downloading..."
            wget -P $DIR --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz;
	fi
	echo "Unzipping..."
	gunzip $DIR/t10k-labels-idx1-ubyte.gz
    fi
fi