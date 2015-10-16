#include <Halide.h>
#include <stdio.h>

using namespace Halide;
using Halide::Image;

#include <fstream>
#include <string>

#include "common.h"
#include "convolution.h"
#include "../../utils/read_mnist.h"
#include "layers.h"

#include "../support/image_io.h"

#define COMPILE_TO_FILE


//-------------------------------------------------------------------
// This main function creates the CNN and compiles it to file
//-------------------------------------------------------------------
int main() {
  Var x("x"), y("y"), z("z"), k("k");
  
#ifdef COMPILE_TO_FILE
  // define the input parameter and its expected size
  int rows=28, cols=28;
  ImageParam input_image(type_of<uint8_t>(), 3);
  Func input("input");
  input(x,y,z) = cast<float>(input_image(x,y,z)) / 255;
#else
  int rows, cols, num_images, num_labels;
  uint8_t* digits = read_mnist_test_images (num_images, rows, cols);
  //uint8_t* labels = read_mnist_test_labels (num_labels);
  //assert (num_images == num_labels);
  Image<uint8_t> input_img = Halide_image_from_buffer(cols, rows, 1, digits, "input_img");
  Func input("input");
  input(x,y,z) = cast<float>(input_img(x,y,z)) / 255;
  print_image_by_layer (input_img, "input.dat");
#endif

  // -----------------------------------------------------------------------------------------------
  // read layer0 params
  
  Parameters p0;
  p0.read_from_file("../../data/mnist/layer_params0.dat", 0);

  // weights : 5 x 5 x 1 x 20
  Func kernel = p0.weight("weight0");
  // bias : 1 x 1 x 1 x 20
  Func bias0 = p0.bias("bias0");
  //printf("%d x %d x %d x %d\n", p0.width, p0.height, p0.layers, p0.count);
  // layer 1: 12 x 12 x 20
  int layer0_conv_x = 0;
  int layer0_conv_y = 0;
  int layer1_x = 0;
  int layer1_y = 0;
  Func layer0_conv = convolutional_layer<float>(
                    input, 
                    kernel, 
                    bias0,
                    "layer0_conv",
                    cols, rows, 
                    p0,
                    layer0_conv_x, layer0_conv_y,
                    0//DEBUG_CONV_OUTPUT | DEBUG_POOL_OUTPUT | DEBUG_CONVRESULT_OUTPUT
      );

  Func layer0 = maxpool_layer<float>(
                    layer0_conv,
                    "layer0_pool",
                    layer0_conv_x, layer0_conv_y, p0.count,
                    2, 2,
                    layer1_x, layer1_y,
                    0
    );

  printf ("Layer 0: %d inputs, %d outputs, %dx%d\n", p0.layers, p0.count, layer1_x, layer1_y);
  printf("Success0!\n");
  
  // ------------------------------------------------------------------------------------------------------------------
  // read layer1 params
  Parameters p1;
  p1.read_from_file("../../data/mnist/layer_params1.dat", 0);
  assert (p1.layers = p0.count);

  //  weights : 5 x 5 x 20 x 50
  Func kernel1 = p1.weight("weight1");
  // bias : 1 x 1 x 1 x 50
  Func bias1 = p1.bias("bias1");

  // layer 2: 4 x 4 x 50
  int layer1_conv_x = 0;
  int layer1_conv_y = 0;
  int layer2_x = 0;
  int layer2_y = 0;
  Func layer1_conv = convolutional_layer<float>(
                    layer0,
                    kernel1,
                    bias1,
                    //bias1,
                    "layer1_conv",
                    layer1_x, layer1_y, 
                    p1,
                    layer1_conv_x, layer1_conv_y,
                    0//DEBUG_CONV_OUTPUT
      );

  Func layer1 = maxpool_layer<float>(
                    layer1_conv,
                    "layer1_pool",
                    layer1_conv_x, layer1_conv_y, p1.count,
                    2, 2,
                    layer2_x, layer2_y,
                    0
    );

  printf ("Layer 1: %d inputs, %d outputs, %dx%d\n", p1.layers, p1.count, layer2_x, layer2_y);
  printf("Success1!\n");

  // ------------------------------------------------------------------------------------------------------------------
  // read layer2 params
  Parameters p2;
  p2.read_from_file("../../data/mnist/layer_params2.dat", 1);
  assert (p2.layers == p1.count * 4 * 4);
  printf ("Layer 2: %d inputs, %d outputs, %dx%d\n", p2.layers, p2.count, p2.width, p2.height);
  
  //Modify the 2 dimension of weights parameter into 4 dimension
  p2.width = p2.count;
  p2.height = layer2_x;
  p2.layers = layer2_y;
  p2.count = p1.count;
  // weights: 500 x 4 x 4 x 50
  Func weight2 = p2.weight("weight2");

  //Convert the parameter of number of outputs back for bias
  p2.count = p2.width;
  // biases: 1 x 1 x 1 x 500
  Func bias2 = p2.bias("bias2");

  // layer2: 800 x 1 x 1
  Func layer2 = hidden_layer<float>(
                    layer1,
                    weight2,
                    bias2,
                    "layer2",
	                  layer2_x,  layer2_y,
                    p1.count,  p2.count,
                    0//DEBUG_HIDDEN_OUTPUT
      );
  printf("Success2!\n");

  // ------------------------------------------------------------------------------------------------------------------
  // read layer3 params  
  Parameters p3;
  p3.read_from_file("../../data/mnist/layer_params3.dat", 1);
  assert (p3.layers = p2.width);
  assert (p3.count = 10);

  //Change the 2 dimension of weights parameter into 4 dimension
  p3.width = p3.count;
  p3.height = p3.layers;
  p3.layers = 1;
  p3.count = 1;
  // weights: 10 x 500 x 1 x 1
  Func weight3 = p3.weight("weight3");

  //change the number of output back for bias
  p3.count = p3.width;
  // biases: 1 x 1 x 1 x 10
  Func bias3 = p3.bias("bias3");

  // layer 3: 10 x 1 x 1
  Func layer3 = dense_layer<float>(
                    layer2,
                    weight3,
                    bias3,
                    "layer3",
                    p3,
                    0//DEBUG_LAST_OUTPUT
  );
  printf ("Layer 3: %d inputs, %d outputs\n", p3.height, p3.count);
  printf("Success3!\n");
  
  // ---------------------------------------------------------------------------------------------------------------------
  //layer 4: 10 x 1 x 1
  Func layer4 = softmax_layer<float>(
                    layer3,
                    10,
                    "layer4"
  );
  printf ("Layer 4: %d inputs, %d outputs\n", 10, 10);
  
  Var xi("xi"), yi("yi");

  layer0_conv.compute_root().vectorize(x,8);
  layer0.compute_root().vectorize(x,8);
  layer1_conv.compute_root().vectorize(x,8);
  layer1.compute_root().vectorize(x,8);
  layer2.compute_root().vectorize(x,8);
  layer3.compute_root().vectorize(x,8);
  layer4.compute_root().vectorize(x,8);
  
#ifdef COMPILE_TO_FILE
  std::vector<Argument> args;
  args.push_back(input_image);
  layer4.compile_to_file("Compiled_Pipeline", args);
  printf("Successfully compiled pipeline\n");
#else
  print_func<float>(layer4, 10, 1, 1, "output");
  printf("Successfully generated classification\n");
#endif

  return 0;
}

