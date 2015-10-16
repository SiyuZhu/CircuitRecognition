#include <cv.h>
#include <highgui.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <gflags/gflags.h>
#include "common.hpp"

using namespace std;
namespace fs = boost::filesystem;

// configure gflags
// TODO fill in the default string
// TODO in the code, check that the arguments are valid - do not assume
DEFINE_string(imgdir_args, "", "directories containing target image files");
DEFINE_int32(img_width_arg, 28, "desired width of images");
DEFINE_int32(img_height_arg, 28, "desired height of images");

// resize every image in the given directories. side-effect of converting
// every image to grayscale
int main(int argc, char **argv) {
  // setup gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // parse names of image directories from gflags argument (see common.cpp)
  vector<string> *imgdir_names = split(FLAGS_imgdir_args, ',');

  // iterate through each directory
  cv::Mat img;
  for(vector<string>::const_iterator imgdir_names_it = imgdir_names->begin();
      imgdir_names_it != imgdir_names->end(); imgdir_names_it++) {
    // initialize directory path
    fs::path imgdir_path(*imgdir_names_it);
    if(!fs::exists(imgdir_path)) continue;
    if(!fs::is_directory(imgdir_path)) continue;

    // resize each image in the directory
    for(fs::directory_iterator imgdir_it(imgdir_path);
	imgdir_it != fs::directory_iterator(); imgdir_it++) {
      if(!fs::exists(imgdir_it->path())) continue;
      // TODO instead of below, should check if it is a valid image file
      if(!imgdir_it->path().filename().string().compare(".DS_Store")) continue;

      // read image in grayscale
      img = cv::imread(imgdir_it->path().string(), CV_LOAD_IMAGE_GRAYSCALE);
      // resize image and write
      cv::resize(img, img, cv::Size(FLAGS_img_width_arg, FLAGS_img_height_arg));
      cv::imwrite(imgdir_it->path().string(), img);
    }
  }

  return 0;
}
