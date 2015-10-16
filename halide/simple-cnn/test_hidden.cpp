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
 
  // fake input data: 4 x 4 x 50
  float input_data[50*4*4];
  for (int i = 0; i < 50*4*4; ++i) {
    input_data[i] = 0;
    if (i < 1)
      input_data[i] = 1;
  }

  // input image
  Func layer2_input = Halide_func_from_buffer(4, 4, 50, input_data, "input");
  print_func<float>(layer2_input, 4, 4, 50, "input");
  
  // read layer2 params
  int numin2=0, numout2=0, kw2 = 0, kh2 = 0;
  float *weight2_data=NULL, *bias2_data=NULL;
  read_params("layer2_params.dat", weight2_data, bias2_data, numin2, numout2, kw2, kh2, 1);
  assert (numin2 == 800);
  assert (numout2 == 500);
  printf ("Layer 2: %d inputs, %d outputs, %dx%d\n", numin2, numout2, kw2, kh2);

  // weights: 500 x 4 x 4 x 50
  Func weight2 = Halide_func_from_buffer(numout2, 4, 4, 50, weight2_data, "weight2");
  // biases: 1 x 1 x 1 x 500
  Func bias2;// = Halide_func_from_buffer(1, 1, 1, numout2, bias2_data, "bias2");
  bias2(x,y,z,k) = cast<float>(0);

  print_func<float>(weight2, 500, 4, 4, 50, "weight2");
  print_func<float>(bias2, 1, 1, 1, 500, "bias2");
  
  // layer2: 500 x 1 x 1
  Func layer2 = hidden_layer<float>(
                    layer2_input,
                    weight2,
                    bias2,
                    "layer2",
		    4, 4,
                    50, 
                    numout2,
                    0//DEBUG_HIDDEN_OUTPUT
      );

  print_func<float>(layer2, numout2, 1, 1, "output");

  printf("Success!\n");

  return 0;
}

