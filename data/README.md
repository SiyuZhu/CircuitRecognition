Contents
========
Data files, extracted parameters, scripts to build datastores, precprocess images, and extract images.
The testing and training data used in each project is located in the associated project directory (ex 
hetero/data/mnist).

To fetch the data used in each project, run the get_data.sh script in the associated project directory.
The script will fetch data only if it is not already present. In order to force it to fetch the data,
pass "force" as the first argument when executing the script. The data in the gates project must first
undergo a preprocessing step, and to build the tools necessary for the preprocessing, run make in the
tools directory (hetero/data/tools).
Fetching gates data while in the hetero/data directory:

```sh
cd tools
make all
cd ../gates
./get_data.sh
```

To build the data from a project directory into a datastore usable by caffe, run buildb_idx.sh or
buildb_imageset.sh (as deemed appropriate) with the project name (ex mnist) as the first argument. Use the
buildb_idx.sh script if the data being built is in idx format or buildb_imageset.sh if the data is stored in
a directory with a txt file with containing the data file names and the associated labels. Note that for now,
the data directories to be used are hard coded in the buildb_idx.sh and buildb_imageset.sh scripts.

To extract the parameters from a caffemodel file, run the extract_params.py script with the project name as
the first argument. If necessary, modify the caffemodel name in extract_params.py. Note that for now,
the script can only properly extract the parameters for a lenet model.

Note: convert_images_idx.sh has not yet been tested. 