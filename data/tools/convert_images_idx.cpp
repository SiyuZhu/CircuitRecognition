#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <fstream>
#include <gflags/gflags.h>
#include <vector>
#include "stdint.h"
#include <algorithm>
#include <cv.hpp>
#include <highgui.h>
#include "common.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

DEFINE_string(valdir_arg, "", "directory containing validation images");
DEFINE_string(valtxt_arg, "", "path of val.txt");
DEFINE_string(imgidx_arg, "", "target file for images IDX file");
DEFINE_string(lblidx_arg, "", "target file for labels IDX file");
DEFINE_int32(img_width_arg, 28, "width of image files");
DEFINE_int32(img_height_arg, 28, "height of image files");

void write_to_idx(uint32_t value, fstream& file) {
  std::reverse((char *) &value, (char *) (&value + 1));
  file.write((char *) &value, 4);
}

int main(int argc, char **argv) {
  // setup gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // setup paths
  fs::path valdir_path(FLAGS_valdir_arg), valtxt_path(FLAGS_valtxt_arg),
    imgidx_path(FLAGS_imgidx_arg), lblidx_path(FLAGS_lblidx_arg);
  fs::remove_all(imgidx_path);
  fs::remove_all(imgidx_path);

  // setup destination file streams
  fstream imgidx, lblidx;
  imgidx.open(imgidx_path.string(), ios_base::out | ios_base::binary | ios_base::trunc);
  lblidx.open(lblidx_path.string(), ios_base::out | ios_base::binary | ios_base::trunc);

  // write magic number
  uint32_t imgmagic = 2051;
  uint32_t lblmagic = 2049;
  write_to_idx(imgmagic, imgidx);
  write_to_idx(lblmagic, lblidx);

  // write count to imgidx and lblidx
  ifstream valtxt (valtxt_path.string());
  string line;
  uint32_t count = 0;
  while(getline(valtxt, line)) {
    vector<string> *values = split(line, ' ');
    if(values->size() < 2) {
      delete values;
      continue;
    }
    count++;
    delete values;
  }
  valtxt.close();
  write_to_idx(count, imgidx);
  write_to_idx(count, lblidx);

  // write width and height to imgidx
  uint32_t rows = (uint32_t) FLAGS_img_height_arg;
  uint32_t cols = (uint32_t) FLAGS_img_width_arg;
  write_to_idx(rows, imgidx);
  write_to_idx(cols, imgidx);

  // write images and labels
  valtxt.open(valtxt_path.string());
  while(getline(valtxt, line)) {
    vector<string> *values = split(line, ' ');
    if(values->size() < 2) {
      delete values;
      continue;
    }
    string img_path_string = valdir_path.string() + "/" + (*values)[0];
    // NOTE: mnist has grayscale range of 0 (white) to 255 (black).
    // gate recognition uses the reverse (same as opencv)
    Mat img = imread(img_path_string, CV_LOAD_IMAGE_GRAYSCALE);
    for(uint32_t r = 0; r < rows; r++) {
      for(uint32_t c = 0; c < cols; c++)
	imgidx.put(img.at<char>(r, c));
    }
    char label = (char) stoi((*values)[1]);
    lblidx.put(label);
    delete values;
  }
  valtxt.close();
  imgidx.close();
  lblidx.close();

  return 0;  
}
