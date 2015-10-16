#include "common.hpp"
#include <boost/filesystem.hpp>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <algorithm>

#define IMG_DIR "../images/selected"

using namespace std;
namespace fs = boost::filesystem;

template <typename T>
void print_quartiles(vector<T> * const& values) {
  vector<T> *quartiles = get_quartiles(values);
  cout.unsetf(ios::floatfield);
  cout.precision(2);
  cout << (*quartiles)[0] << " , "
       << (*quartiles)[1] << " , "
       << (*quartiles)[2] << " , "
       << (*quartiles)[3] << " , "
       << (*quartiles)[4] << endl;
}

// print quartile values of width, height, w/h ratio
int main(int argc, char ** argv) {
  fs::path img_dir_path(IMG_DIR);
  for(fs::directory_iterator dir_it = fs::directory_iterator(img_dir_path);
      dir_it != fs::directory_iterator(); dir_it++) {
    if(!dir_it->path().filename().string().compare(".DS_Store")) continue;
    cv::Mat image;
    vector<int> *img_widths = new vector<int>();
    vector<int> *img_heights = new vector<int>();
    vector<float> *img_wh_ratios = new vector<float>();
    for(fs::directory_iterator subdir_it = fs::directory_iterator(dir_it->path());
	subdir_it != fs::directory_iterator() ; subdir_it++) {
      if(!subdir_it->path().filename().string().compare(".DS_Store")) continue;
      image = cv::imread(subdir_it->path().string(), CV_LOAD_IMAGE_GRAYSCALE);
      // cv::imwrite(subdir_it->path().string(), image); // write greyscale image
      img_widths->push_back(image.cols);
      img_heights->push_back(image.rows);
      img_wh_ratios->push_back(((float) image.cols)/((float) image.rows));
    }
    cout << dir_it->path().filename().string() << endl;
    cout << "Width: "; print_quartiles(img_widths);
    delete img_widths;
    cout << "Height: "; print_quartiles(img_heights);
    delete img_heights;
    cout << "WH Ratio: "; print_quartiles(img_wh_ratios);
    delete img_wh_ratios;
  }

  return 0;
}
