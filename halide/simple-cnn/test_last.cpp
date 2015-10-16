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
  int numin = 500, numout = 10;
  int kw3=0, kh3=0;

  // fake input data
  float input_data[500];
  for (int i = 0; i < 500; ++i) {
    input_data[i] = 0;
  }
  input_data[0] = 1;

  // input image
  Func input = Halide_func_from_buffer(500, 1, 1, input_data, "input");

  // read data
  float *weight3_data=NULL, *bias3_data=NULL;
  read_params("layer3_params.dat", weight3_data, bias3_data, numin, numout, kw3, kh3, 1);
  assert (numout == 10);
  assert (numin == 500);
  
  // weights
  Func weight3 = Halide_func_from_buffer(numout, numin, 1, 1, weight3_data, "weight3");
  Func bias3 = Halide_func_from_buffer(1, 1, 1, numout, bias3_data, "bias3");

  Func layer3 = last_layer<float>(
                    input,
                    weight3,
                    bias3,
                    "layer3",
                    numout, numin,
                    0//DEBUG_LAST_OUTPUT
  );
  printf ("Layer 3: %d inputs, %d outputs\n", numin, numout);
  
  print_func<float>(layer3, numout, 1, 1, "output");

  printf("Success!\n");
  
  return 0;
}

