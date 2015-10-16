#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <algorithm>
#include "common.hpp"

using namespace std;
namespace fs = boost::filesystem;

// TODO the floating point multiplication for allocation leads to some images
// not being used.

// configure gflags
// TODO fill in default strings
// TODO in the code, check if arguments are valid - do not assume
DEFINE_string(imgdir_args, "", "directories containing target image files");
DEFINE_string(traindir_arg, "", "destination for training images");
DEFINE_string(traintxt_arg, "", "path to train.txt");
DEFINE_string(valdir_arg, "", "destination for validation images");
DEFINE_string(valtxt_arg, "", "path to val.txt");
DEFINE_bool(shuffle_arg, true, "shuffles files in image directory before allocating them");
DEFINE_bool(sort_arg, false, "shuffles files in image directory before allocating them");
DEFINE_double(train_percent_arg, 0.8, "percentage of images to use for training");
DEFINE_double(val_percent_arg, 0.2, "percentage of iamges to use for validation");

// allocates (count) number of images to dstdir, and then writes the image
// filename and label to record
void allocate_images(vector<fs::path> * const images, const fs::path& dstdir,
		     const fs::path& record, const int& label, const int& count) {
  fstream recordfile;
  recordfile.open(record.string(), fstream::out | fstream::app);
  for(int i = 0; i < count ; i++) {
    string filename = images->back().filename().string();
    fs::path dst(dstdir.string() + "/" + filename);
    fs::copy(images->back(),dst);
    recordfile << filename << " " << label << endl;
    images->pop_back();
  }
  recordfile.close();
  return;
}

// distributes images from the given source directories to the two destination
// directories
int main(int argc, char **argv) {
  // setup gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // parse names of image directories from gflags argument (see common.cpp)
  vector<string> *imgdir_names = split(FLAGS_imgdir_args, ',');
  // initiate paths for the image destinations and records
  fs::path train_data(FLAGS_traindir_arg), train_txt(FLAGS_traintxt_arg),
    val_data(FLAGS_valdir_arg), val_txt(FLAGS_valtxt_arg);

  // delete and then create destination directories (see common.cpp)
  recreate(train_data);
  recreate(val_data);
  
  // iterate through each image directory, distributing images from each directory
  // to the destination directories
  int label = 0;
  for(vector<string>::const_iterator imgdir_names_it = imgdir_names->begin();
      imgdir_names_it != imgdir_names->end(); imgdir_names_it++) {
    // initialize image directory path
    fs::path imgdir_path(*imgdir_names_it);
    if(!fs::exists(imgdir_path)) continue;
    if(!fs::is_directory(imgdir_path)) continue;

    // collect all image paths in the image directory
    vector<fs::path> *img_paths = new vector<fs::path>;
    for(fs::directory_iterator imgdir_it(imgdir_path);
	imgdir_it != fs::directory_iterator(); imgdir_it++) {
      if(!fs::exists(imgdir_it->path())) continue;
      // TODO instead of below, check if it is a valid image file
      if(!imgdir_it->path().filename().string().compare(".DS_Store")) continue;
      img_paths->push_back(imgdir_it->path());      
    }

    // shuffle or sort the image paths based on the arguments
    if(FLAGS_shuffle_arg) random_shuffle(img_paths->begin(), img_paths->end());
    if(FLAGS_sort_arg) sort(img_paths->begin(), img_paths->end());

    // calculate the number of images to be used for training and for validation
    int train_count = (int) (FLAGS_train_percent_arg*(img_paths->size()));
    int val_count = (int) (FLAGS_val_percent_arg*(img_paths->size()));

    // allocate images to each of the destination directories
    allocate_images(img_paths, train_data, train_txt, label, train_count);
    allocate_images(img_paths, val_data, val_txt, label++, val_count);

    delete img_paths;
    img_paths = NULL;
  }

  return 0;
}
