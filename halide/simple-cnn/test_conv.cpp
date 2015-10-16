#include <Halide.h>
#include <stdio.h>

using namespace Halide;
using Halide::Image;

#include <fstream>
#include <string>

#include "common.h"
#include "convolution.h"
#include "layers.h"
#include "../support/image_io.h"
#include "../../utils/read_mnist.h"
#include "../../utils/read_params.h"

int main (int argc, char** argv) {
  Var x("x"), y("y"), z("z"), k("k");
 
  // fake input data: 4 x 4 x 1
  float input_data[4*4];
  for (int i = 0; i < 4*4; ++i) {
    input_data[i] = static_cast<float>(i)-7.5;
  }

  // input image
  Func input = Halide_func_from_buffer(4, 4, 1, input_data, "input");
  print_func<float>(input, 4, 4, 1, "input");

  Func pooled = maxpool<float>(input, 2, 2, 1, "pooled");
  pooled.compile_to_c("maxpool.cpp", std::vector<Argument>(), "maxpool");
  print_func<float>(pooled, 2, 2, 1, "output");

  return 0;
}

