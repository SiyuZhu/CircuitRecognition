#include <gflags/gflags.h>
#include <boost/filesystem.hpp>
#include <string>
#include "common.hpp"

using namespace std;
namespace fs = boost::filesystem;

// configure gflags
// TODO fill in the default strings
// TODO in the code, check if the arguments are valid
DEFINE_string(imgdir_args, "", "directories containing target image files");

// renames images in the given directories to the format 
// prepend_dirname_postpend.extension
// where prepend is different for each image in a directory, but postpend
// is common for all images in the directoy but different among images from
// different directories. the prepend was used specifically so that images in
// the same directory can be consistenly sorted, but the remaining part of the
// name is somewhat arbitrary. this was largely implemented because the image
// samples are all apple screenshots, and the naming scheme for these screenshots
// is not compatible with caffe.
// TODO introduce randomization after prepend to prevent overwriting
// TODO always check before renaming to avoid overwriting
int main(int argc, char **argv) {
  // setuo gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // parse names of image directories from gflags argument (see common.cpp)
  vector<string> *imgdir_names = split(FLAGS_imgdir_args, ',');

  // iterate through given directories, renaming every images according to the 
  // naming scheme
  unsigned int postpend_int = 0;
  for(vector<string>::const_iterator imgdir_names_it = imgdir_names->begin();
      imgdir_names_it != imgdir_names->end(); imgdir_names_it++) {
    // initialize directory path
    fs::path imgdir_path(*imgdir_names_it);
    if(!fs::exists(imgdir_path)) continue;
    if(!fs::is_directory(imgdir_path)) continue;

    // iteratively rename every image
    unsigned int prepend_int = 0;
    string imgdir_name = imgdir_path.filename().string();
    string imgdir_parent = imgdir_path.string();
    for(fs::directory_iterator imgdir_it(imgdir_path);
	imgdir_it != fs::directory_iterator(); imgdir_it++) {
      if(!fs::exists(imgdir_it->path())) continue;
      // TODO instead of below, check if it is a valid image format
      if(!imgdir_it->path().filename().string().compare(".DS_Store")) continue;
      // create new name and rename image
      fs::path dst(imgdir_parent + "/" +
		   to_string((long long unsigned int) prepend_int++) + "_" +
		   imgdir_name + "_" +
		   to_string((long long unsigned int) postpend_int) +
		   fs::extension(imgdir_it->path()));
      fs::rename(imgdir_it->path(), dst);
    }

    postpend_int++;
  }

  return 0;
}
