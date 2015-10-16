Contents
========
Caffe architectures, models, and training tools. Requires training and testing data be built to lmdb/leveldb
databases.
Training a model while in caffe directory:

```sh
../data/buildb_idx.sh mnist
# use buildb_idx.sh because mnist images are stored in idx format.
# for gates, use ../data/buildb_imageset.sh gates
./train.sh mnist
```

Training snapshots will be stored in the respective project directory (ex hetero/caffe/mnist).
To create a new architectures for training, make a directory with the project name, and include the
a solver file named "solver.prototxt", and include the necessary testing and training architecture files.
To swap the training and testing data used for each architecutre, modify the "source" tag in the net protobuf
definition, and make sure that the lmdb/leveldb datastores pointed to by those sources are
appropriately built (see hetero/data/buildb_idx.sh and hetero/data/buildb_imageset.sh)

Each project directory also contains python scripts to rest the resulting model. Adjust the model file and data
fields of the python scripts if necessary. Each project directory has a working script to print the activation
maps to pdf(s) and a scipt to time the classifer.

BENCHMARK TESTING
Caffe offers a time command to do official time benchmarks for their system. Running the time script will
run the benchmark script for the given project:

```sh
./time.sh mnist
```

Likewise, testing the accuraccy of a trained model can be easily done by using caffe's train command.
An associated script is provided:

```sh
./train.sh mnist
```