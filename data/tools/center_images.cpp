#include <iostream>
#include <cv.hpp>
#include <highgui.h>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include "common.hpp"

using namespace std;
namespace fs = boost::filesystem;

// configure gflags
// TODO fill in default parameters
// TODO make sure in code that arguments are valid
DEFINE_string(imgdir_args, "", "directories containing target image files");
DEFINE_bool(choose_size_args, false,
	    "set to specify desired image dimensions in arguments");
DEFINE_int32(img_width_args, 0, "desired width of images");
DEFINE_int32(img_height_args, 0, "desired height of images");

// ceneter images from the source directory onto a white background
int main(int argc, char **argv) {
  // setup gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // parse names of image directories from gflags argument (see common.cpp)
  vector<string> *imgdir_names = split(FLAGS_imgdir_args, ',');
  
  // iterate through each image directory, centering every image in the directory
  for(vector<string>::const_iterator imgdir_names_it = imgdir_names->begin();
      imgdir_names_it != imgdir_names->end(); imgdir_names_it++) {
    // initialize image directory path
    fs::path imgdir_path(*imgdir_names_it);
    if(!fs::exists(imgdir_path)) continue;
    if(!fs::is_directory(imgdir_path)) continue;

    // center every image in the directory
    for(fs::directory_iterator imgdir_it(imgdir_path);
	imgdir_it != fs::directory_iterator(); imgdir_it++) {
      if(!fs::exists(imgdir_it->path())) continue;
      // TODO instead of below, should check if it is a valid image file
      if(!imgdir_it->path().filename().string().compare(".DS_Store")) continue;

      // read image
      cv::Mat img_src = cv::imread(imgdir_it->path().string(), CV_LOAD_IMAGE_GRAYSCALE);

      // set desired height and width
      int desired_height = 0;
      int desired_width = 0;
      if(FLAGS_choose_size_args) {
	desired_height = FLAGS_img_height_args;
	desired_width = FLAGS_img_width_args;
      } else {
	desired_height = (img_src.rows > img_src.cols) ? img_src.rows : img_src.cols;
	desired_width = (img_src.rows > img_src.cols) ? img_src.rows : img_src.cols;
      }

      // center image onto a white background of the desired dimensions
      cv::Mat img_dst(cv::Size(desired_width, desired_height), CV_8UC1);
      img_dst = cv::Scalar(255);
      int start_row = (int) ((desired_height-img_src.rows)/2);
      int start_col = (int) ((desired_width-img_src.cols)/2);
      img_src.copyTo(img_dst.rowRange(start_row, start_row + img_src.rows).
		     colRange(start_col, start_col + img_src.cols));
      cv::imwrite(imgdir_it->path().string(), img_dst);
      // TODO assumes src image can fit in destination if user specifies height
    }
  }

  return 0;
}
