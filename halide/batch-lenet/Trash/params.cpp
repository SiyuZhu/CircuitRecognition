#include "params.hpp"
#include "Halide.h"
#include <string>
#include "../../utils/read_params.h"

using namespace Halide;
using namespace std;

Parameters<float> *read_params_dat(string filename, int dense_flag) {
  Parameters<float>* params = new Parameters<float>;
  read_params(filename, params->weight_data, params->bias_data,
	      params->depth, params->count, params->height,
	      params->width, dense_flag);
  return params;
}
