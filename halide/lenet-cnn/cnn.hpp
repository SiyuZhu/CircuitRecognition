#ifndef CNN_HPP
#define CNN_HPP

#include "Halide.h"
#include "../../utils/read_params.h"
#include <utility>
#include <vector>
#include <string>

using namespace std;
using namespace Halide;

pair<Image<float>, Image<float>> read_dat_params(string filepath,
						 string name_prefix,
						 int dense_flag) {
  // weight_dim stores dimensional extent in the same way as halide. The
  // diemnsion at index 0 of weight_dim is the dimension of which the index
  // changes fastest during a c-style array iteration
  vector<int> weight_dim {0, 0, 0, 0};
  int bias_size;
  float *weight_data, *bias_data;

  // parses the parameters from the given file
  if(dense_flag == 1) {
    read_params(filepath, weight_data, bias_data,
		weight_dim[1], weight_dim[0], weight_dim[2],
		weight_dim[3], dense_flag);
    bias_size = weight_dim[0];
  } else {
    read_params(filepath, weight_data, bias_data,
		weight_dim[2], weight_dim[3], weight_dim[1],
		weight_dim[0], dense_flag);
    bias_size = weight_dim[3];
  }

  // creates halide images using the given weight and bias data
  Image<float> weight_image(Buffer(type_of<float>(), weight_dim,
				   (uint8_t*) weight_data, name_prefix + 
				   "_weight"));
  Image<float> bias_image(Buffer(type_of<float>(), bias_size, 0, 0, 0,
				 (uint8_t*) bias_data, name_prefix +
				 "_bias"));
  pair<Image<float>, Image<float>> image_pair(weight_image, bias_image);
  return image_pair;
}

#endif
