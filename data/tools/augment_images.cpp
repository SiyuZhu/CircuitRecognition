#include <boost/filesystem.hpp>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <gflags/gflags.h>
#include "common.hpp"

using namespace cv;
namespace fs = boost::filesystem;

// TOOD since this may operate on train_data and val_data, it makes sure that train.txt and 
// val.txt are not augmented. but it should really be checking if each file is an image file

DEFINE_string(imgdir_args, "", "directories containing images to be augmented");
DEFINE_bool(aug8_args, false, "make 8 versions of each image (using 90 degree rotations and flips about its major axis");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  vector<string> *imgdir_names = split(FLAGS_imgdir_args, ',');

  for(vector<string>::const_iterator imgdir_names_it = imgdir_names->begin();
      imgdir_names_it != imgdir_names->end(); imgdir_names_it++) {
    fs::path imgdir_path(*imgdir_names_it);
    if(!fs::exists(imgdir_path)) continue;
    if(!fs::is_directory(imgdir_path)) continue;
    
    for(fs::directory_iterator imgdir_it(imgdir_path);
	imgdir_it != fs::directory_iterator(); imgdir_it++) {
      if(!fs::exists(imgdir_it->path())) continue;
      if(!imgdir_it->path().filename().string().compare(".DS_Store")) continue;
      if(!imgdir_it->path().filename().string().compare("train.txt")) continue;
      if(!imgdir_it->path().filename().string().compare("val.txt")) continue;
      
      fs::path src_path = imgdir_it->path();
      Mat src = imread(src_path.string());
      char label = 'a';
      if(FLAGS_aug8_args) {
	bool loop_flag(false);
	do {
	  Mat dst;
	  
	  flip(src, dst, 0);
	  imwrite(src_path.parent_path().string() + "/" + label +
		  src_path.filename().string(), dst);
	  label++;
	  
	  flip(src, dst, 1);
	  imwrite(src_path.parent_path().string() + "/" + label +
		  src_path.filename().string(), dst);
	  label++;

	  flip(src, dst, -1);
	  imwrite(src_path.parent_path().string() + "/" + label +
		  src_path.filename().string(), dst);
	  label++;

	  if(loop_flag) break;
	  
	  transpose(src, src);
	  flip(src, src, 0);
	  src_path = fs::path(src_path.parent_path().string() + "/" +
			      label + src_path.filename().string());
	  imwrite(src_path.string(), src);
	  label++;
	  loop_flag = !loop_flag;
	} while (loop_flag);
      } else {
	flip(src, src, 0);
	imwrite(src_path.parent_path().string() + "/" + label + 
		src_path.filename().string(), src);
      }
    }
  }

  return 0;
}
